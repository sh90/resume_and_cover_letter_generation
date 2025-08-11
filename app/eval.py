# app/eval.py
# Lightweight evaluation helpers for resume bullets & cover letters

from __future__ import annotations
from typing import Dict

def _keywords_from_jd(jd: str) -> set:
    # crude keyword pick: long words, TitleCase tokens, and common skill tokens
    jd = jd or ""
    words = [w.strip(".,:;()[]{}") for w in jd.split()]
    skills = {"python","sql","tableau","power","bi","ml","machine","learning","analytics","django","postgresql"}
    keys = set(
        w.lower()
        for w in words
        if (w.istitle() and len(w) >= 3) or len(w) >= 7 or w.lower() in skills
    )
    return {k for k in keys if k.isalpha()}

def keyword_coverage(jd: str, output: str) -> float:
    keys = _keywords_from_jd(jd)
    if not keys:
        return 0.0
    out = (output or "").lower()
    hits = sum(1 for k in keys if k in out)
    return hits / len(keys)

def quantify_score(output: str) -> float:
    """
    Count presence of quantitative cues per line.
    Higher is better; ~0.3–1.0 is typical for decent resumes.
    """
    if not output:
        return 0.0
    cues = ["%", "increased", "reduced", "cut", "grew", "decreased",
            "saved", "roi", "uplift", "improved", "revenue", "cost", "conversion"]
    lines = [ln for ln in output.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    total = 0
    for ln in lines:
        l = ln.lower()
        total += sum(l.count(c) for c in cues)
        # numeric tokens
        total += sum(ch.isdigit() for ch in l) * 0.05
    return total / max(1, len(lines))

def length_ok(output: str, min_words=120, max_words=200) -> bool:
    n = len((output or "").split())
    return min_words <= n <= max_words

def compute_metrics(jd: str, output: str, task: str) -> Dict[str, float | bool]:
    kc = keyword_coverage(jd, output)
    qs = quantify_score(output)
    lo = length_ok(output) if task == "cover_letter" else None
    return {
        "keyword_coverage": round(kc, 4),
        "quantify_score": round(qs, 4),
        "length_ok": bool(lo) if lo is not None else None,
    }

def composite_score(jd: str, output: str, task: str) -> float:
    """
    Simple weighted score:
      - bullets: 60% keyword coverage, 40% quantification (capped)
      - cover letter: 50% keyword, 40% quantification, +0.1 bonus if length_ok
    """
    m = compute_metrics(jd, output, task)
    kc = m["keyword_coverage"]
    qs = min(1.0, m["quantify_score"] / 2.0)  # normalize quantification roughly into [0,1]
    if task == "cover_letter":
        base = 0.5 * kc + 0.4 * qs
        bonus = 0.1 if m["length_ok"] else 0.0
        return round(min(1.0, base + bonus), 4)
    else:
        return round(min(1.0, 0.6 * kc + 0.4 * qs), 4)
