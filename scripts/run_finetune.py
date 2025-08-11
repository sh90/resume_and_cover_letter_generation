# python scripts/run_finetune.py
# Fine-tune helper for your workshop (Python 3.12)

import os, time, sys, pathlib
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

DATA_PATH = os.getenv("FT_DATA_PATH", "data/finetune.jsonl")
# IMPORTANT: use the exact snapshot name for FT
BASE_MODEL = os.getenv("BASE_MODEL", "gpt-4o-mini-2024-07-18")
N_EPOCHS = int(os.getenv("FT_N_EPOCHS", "3"))
SUFFIX = os.getenv("FT_SUFFIX", "resume-cover-ft")

ENV_FILE = pathlib.Path(".env")
TUNED_ID_FILE = pathlib.Path("data/tuned_model.txt")

def fail(msg: str, code: int = 1):
    print(f"[FT] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def update_env(var: str, value: str):
    """Set or replace VAR=value in .env."""
    lines = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    out = []
    found = False
    for ln in lines:
        if ln.strip().startswith(f"{var}="):
            out.append(f"{var}={value}")
            found = True
        else:
            out.append(ln)
    if not found:
        out.append(f"{var}={value}")
    ENV_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")

def pretty_status(s: str) -> str:
    return {
        "queued": "⏳ queued",
        "running": "🏃 running",
        "succeeded": "✅ succeeded",
        "failed": "❌ failed",
        "cancelled": "🚫 cancelled",
    }.get(s, s)

def main():
    # Basic validations
    p = pathlib.Path(DATA_PATH)
    if not p.exists():
        fail(f"Training file not found: {DATA_PATH}")
    if p.stat().st_size == 0:
        fail(f"Training file is empty: {DATA_PATH}")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        fail("OPENAI_API_KEY is missing. Add it to your environment or .env")

    client = OpenAI(api_key=api_key)

    # Upload training file
    print(f"[FT] Uploading {DATA_PATH} …")
    try:
        f = client.files.create(file=open(DATA_PATH, "rb"), purpose="fine-tune")
    except OpenAIError as e:
        fail(f"Upload failed: {e}")

    # Create FT job
    try:
        print(f"[FT] Creating job on base model: {BASE_MODEL}")
        job = client.fine_tuning.jobs.create(
            training_file=f.id,
            model=BASE_MODEL,
            hyperparameters={"n_epochs": N_EPOCHS},
            suffix=SUFFIX[:18],  # API restricts suffix length; keep it short
        )
    except OpenAIError as e:
        # Common cause: wrong model name (e.g., using 'gpt-4o-mini' without snapshot date)
        fail(f"Job creation failed: {e}")

    print(f"[FT] Job ID: {job.id}")
    # Poll
    while True:
        time.sleep(5)
        try:
            j = client.fine_tuning.jobs.retrieve(job.id)
        except OpenAIError as e:
            fail(f"Polling failed: {e}")
        print(f"[FT] Status: {pretty_status(j.status)}")
        if j.status in ("succeeded", "failed", "cancelled"):
            break

    if j.status != "succeeded":
        fail(f"Fine-tune ended with status: {j.status}")

    tuned = j.fine_tuned_model
    if not tuned:
        fail("Job succeeded but no fine-tuned model ID returned.")

    # Persist tuned model ID
    print(f"[FT] Fine-tuned model: {tuned}")
    update_env("GEN_MODEL", tuned)
    TUNED_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    TUNED_ID_FILE.write_text(tuned + "\n", encoding="utf-8")
    print(f"[FT] Saved to .env (GEN_MODEL) and {TUNED_ID_FILE}")
    print("[FT] Done. Restart your app or `source .venv/bin/activate && streamlit run app/app.py`")

if __name__ == "__main__":
    main()
