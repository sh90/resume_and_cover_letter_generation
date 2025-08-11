# scripts/ab_test.py
# A/B test UI + CLI for Few-shot vs Fine-tuned models.
# Run UI:    streamlit run scripts/ab_test.py
# Run CLI:   python scripts/ab_test.py --baseline-model gpt-4o-mini --tuned-model "$(cat data/tuned_model.txt)"

from __future__ import annotations
import os, sys, argparse, glob, pathlib, datetime, json, io, zipfile
import pandas as pd

# Allow "from app.xxx import ..." when running from scripts/
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.prompts import RESUME_BULLETS_TMPL, COVER_LETTER_TMPL
from app.llm import generate_text, GenConfig
from app.eval import compute_metrics, composite_score

# Optional imports for UI features
def _try_imports():
    st = None
    docx = None
    pypdf = None
    try:
        import streamlit as st  # type: ignore
    except Exception:
        pass
    try:
        from docx import Document as _Doc  # type: ignore
        docx = _Doc
    except Exception:
        pass
    try:
        import PyPDF2 as _PyPDF2  # type: ignore
        pypdf = _PyPDF2
    except Exception:
        pass
    return st, docx, pypdf

def _read_text_filelike(uploaded_file, docx_cls=None, pypdf=None) -> str:
    if uploaded_file is None:
        return ""
    name = getattr(uploaded_file, "name", "")
    mime = getattr(uploaded_file, "type", "")
    # TXT
    if mime == "text/plain" or (name and name.lower().endswith(".txt")):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    # DOCX
    if (mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document") or (name and name.lower().endswith(".docx")):
        if docx_cls is None:
            return "[Unsupported DOCX: python-docx not installed]"
        doc = docx_cls(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    # PDF
    if (mime == "application/pdf") or (name and name.lower().endswith(".pdf")):
        if pypdf is None:
            return "[Unsupported PDF: PyPDF2 not installed]"
        reader = pypdf.PdfReader(uploaded_file)
        pages = []
        for pg in reader.pages:
            try:
                t = pg.extract_text() or ""
            except Exception:
                t = ""
            pages.append(t)
        return "\n".join(pages)
    return "[Unsupported file type]"

def _read_text(path: str) -> str:
    p = pathlib.Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""

def _build_prompt(task: str, jd: str, resume: str, examples: str) -> str:
    if task == "bullets":
        return RESUME_BULLETS_TMPL.render(jd=jd, resume=resume, examples=examples)
    elif task == "cover_letter":
        highlights = "\n".join([l for l in resume.splitlines() if l.strip().startswith("-")][:6])
        return COVER_LETTER_TMPL.render(jd=jd, highlights=highlights, examples=examples)
    else:
        raise ValueError(f"Unknown task: {task}")

def _run_ab_once(samples_dir: str, baseline_model: str, tuned_model: str | None, fewshot_text: str,
                 tasks: list[str], limit: int, out_dir: pathlib.Path, progress_cb=None):
    sample_dirs = sorted([p for p in glob.glob(os.path.join(samples_dir, "*")) if os.path.isdir(p)])
    if limit:
        sample_dirs = sample_dirs[:limit]
    rows = []
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    total = len(sample_dirs) * len(tasks)
    done = 0
    for sdir in sample_dirs:
        sid = pathlib.Path(sdir).name
        jd = _read_text(os.path.join(sdir, "jd.md"))
        resume = _read_text(os.path.join(sdir, "profile.md"))
        if not jd or not resume:
            continue

        for task in tasks:
            prompt = _build_prompt(task, jd, resume, fewshot_text)

            # Baseline
            try:
                out_base = generate_text(prompt, GenConfig(model=baseline_model))
            except Exception as e:
                out_base = f"[ERROR baseline] {e}"
            m_base = compute_metrics(jd, out_base, task)
            m_base["composite_score"] = composite_score(jd, out_base, task)
            rows.append({
                "sample_id": sid, "task": task, "model_type": "baseline",
                "model_name": baseline_model, "output": out_base, **m_base
            })
            (raw_dir / f"{sid}_{task}_baseline.txt").write_text(out_base or "", encoding="utf-8")

            # Tuned
            if tuned_model:
                try:
                    out_tuned = generate_text(prompt, GenConfig(model=tuned_model))
                except Exception as e:
                    out_tuned = f"[ERROR tuned] {e}"
                m_tuned = compute_metrics(jd, out_tuned, task)
                m_tuned["composite_score"] = composite_score(jd, out_tuned, task)
                rows.append({
                    "sample_id": sid, "task": task, "model_type": "tuned",
                    "model_name": tuned_model, "output": out_tuned, **m_tuned
                })
                (raw_dir / f"{sid}_{task}_tuned.txt").write_text(out_tuned or "", encoding="utf-8")

            done += 1
            if progress_cb:
                progress_cb(done / max(1, total))

    df = pd.DataFrame(rows)
    results_csv = out_dir / "results.csv"
    df.to_csv(results_csv, index=False)

    metrics = ["keyword_coverage", "quantify_score", "composite_score"]
    summary = df.groupby(["task", "model_type", "model_name"])[metrics].mean().reset_index()
    summary_csv = out_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False)

    # package zip for download
    memfile = io.BytesIO()
    with zipfile.ZipFile(memfile, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.csv", results_csv.read_text(encoding="utf-8"))
        zf.writestr("summary.csv", summary_csv.read_text(encoding="utf-8"))
        for p in raw_dir.glob("*.txt"):
            zf.writestr(f"raw/{p.name}", p.read_text(encoding="utf-8"))
    memfile.seek(0)

    return df, summary, memfile

# ----------------------------
# Streamlit UI
# ----------------------------
def run_ui():
    st, docx_cls, pypdf = _try_imports()
    if st is None:
        print("Streamlit not installed. Install with: pip install streamlit", file=sys.stderr)
        sys.exit(1)

    st.set_page_config(page_title="A/B Test — Few-shot vs Fine-tuned", page_icon="🧪", layout="wide")
    st.title("🧪 A/B Test — Few-shot vs Fine-tuned")

    # Environment hints
    api_ok = os.getenv("OPENAI_API_KEY")
    tuned_default = os.getenv("GEN_MODEL", "")

    if not api_ok:
        st.error("OPENAI_API_KEY is missing. Add it to your environment or .env and restart.")
    else:
        st.success("OPENAI_API_KEY detected.")

    with st.sidebar:
        st.header("Configuration")
        samples_dir = st.text_input("Samples directory", "data/samples")
        baseline_model = st.text_input("Baseline model", "gpt-4o-mini")
        tuned_model = st.text_input("Tuned model (ft:...)", tuned_default)
        tasks = st.multiselect("Tasks", ["bullets", "cover_letter"], ["bullets", "cover_letter"])
        limit = st.number_input("Limit samples (0 = all)", min_value=0, step=1, value=0)

        st.markdown("---")
        st.caption("Few-shot examples (optional)")
        few_file = st.file_uploader("Upload TXT/PDF/DOCX", type=["txt", "pdf", "docx"])
        few_text_from_file = _read_text_filelike(few_file, docx_cls=docx_cls, pypdf=pypdf) if few_file else ""
        few_text_area = st.text_area("Or paste examples here", value=few_text_from_file, height=160)

        st.markdown("---")
        run_btn = st.button("Run A/B test", type="primary")

    # Main area
    if run_btn:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = pathlib.Path(f"results/ab_run_{ts}")
        out_dir.mkdir(parents=True, exist_ok=True)

        st.info(f"Running on **{samples_dir}** → results in `{out_dir}`")
        prog = st.progress(0.0)
        status = st.empty()

        def _prog(pct):
            prog.progress(min(1.0, pct))

        try:
            df, summary, zip_mem = _run_ab_once(
                samples_dir=samples_dir,
                baseline_model=baseline_model,
                tuned_model=tuned_model.strip() or None,
                fewshot_text=few_text_area or "",
                tasks=tasks,
                limit=limit,
                out_dir=out_dir,
                progress_cb=_prog
            )
        except Exception as e:
            st.error(f"Run failed: {e}")
            return

        st.success("Done!")

        # Show summary
        st.subheader("Summary (mean scores by task & model)")
        st.dataframe(summary, use_container_width=True)

        # Simple chart of composite_score
        try:
            pivot = summary.pivot(index="model_type", columns=["task", "model_name"], values="composite_score")
            st.bar_chart(pivot)
        except Exception:
            pass

        # Show sample of detailed results
        st.subheader("Detailed results (sample)")
        st.dataframe(df.head(20), use_container_width=True)

        # Downloads
        results_csv = out_dir / "results.csv"
        summary_csv = out_dir / "summary.csv"
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Download results.csv", results_csv.read_bytes(), file_name="results.csv", mime="text/csv")
        with col2:
            st.download_button("Download summary.csv", summary_csv.read_bytes(), file_name="summary.csv", mime="text/csv")
        with col3:
            st.download_button("Download all (zip)", data=zip_mem, file_name="ab_run.zip", mime="application/zip")

        st.caption(f"Raw outputs saved under `{out_dir}/raw/`.")

# ----------------------------
# CLI fallback
# ----------------------------
def main_cli():
    ap = argparse.ArgumentParser(description="A/B test few-shot vs fine-tuned and store results.")
    ap.add_argument("--samples-dir", default="data/samples")
    ap.add_argument("--baseline-model", default=os.getenv("BASELINE_MODEL", "gpt-4o-mini"))
    ap.add_argument("--tuned-model", default=os.getenv("GEN_MODEL", ""))
    ap.add_argument("--fewshot", default="")
    ap.add_argument("--tasks", default="bullets,cover_letter")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    fewshot_text = _read_text(args.fewshot) if args.fewshot else ""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.out or f"results/ab_run_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ab] Baseline: {args.baseline_model}")
    print(f"[ab] Tuned:    {args.tuned_model or '(none)'}")
    print(f"[ab] Tasks:    {tasks}")
    print(f"[ab] Samples:  {args.samples_dir}")
    if args.fewshot:
        print(f"[ab] Few-shot: {args.fewshot}")

    df, summary, _ = _run_ab_once(
        samples_dir=args.samples_dir,
        baseline_model=args.baseline_model,
        tuned_model=args.tuned_model or None,
        fewshot_text=fewshot_text,
        tasks=tasks,
        limit=args.limit,
        out_dir=out_dir,
        progress_cb=None
    )
    print("\n[ab] Summary (means):")
    print(summary.to_string(index=False))
    print(f"\n[ab] Wrote results to: {out_dir}")

if __name__ == "__main__":
    # If launched via Streamlit, just render UI. Otherwise, use CLI entry.
    try:
        import streamlit as _st  # noqa
        # If streamlit is present AND we are launched via "streamlit run", render the UI
        if any("streamlit" in a for a in sys.argv):
            run_ui()
        else:
            # If someone runs "python scripts/ab_test.py", try to render UI anyway if streamlit is installed
            run_ui()
    except Exception:
        main_cli()
