# scripts/prep_dataset.py

import json, glob, pathlib, re

def read(p): return pathlib.Path(p).read_text(encoding="utf-8")

def make_example(jd_text, profile_text, out_text, task):
    system = "You are a writing assistant that tailors resumes and cover letters to job descriptions."
    user = f"""Task: {task}
    Job Description:
    {jd_text}
    
    Candidate Profile:
    {profile_text}
    
    Write concise, quantified, impact-oriented content."""
    assistant = out_text.strip()
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]}

def main():
    rows = []
    # Example pattern: for each sample folder, read jd, profile, outputs
    for sample_dir in glob.glob("data/samples/*"):
        try:
            jd = read(f"{sample_dir}/jd.md")
            profile = read(f"{sample_dir}/profile.md")
            bullets = read(f"{sample_dir}/out_resume_bullets.md")
            cover = read(f"{sample_dir}/out_cover_letter.md")
            rows.append(make_example(jd, profile, bullets, "Generate tailored resume bullets"))
            rows.append(make_example(jd, profile, cover, "Generate a tailored cover letter"))
        except FileNotFoundError:
            continue
    with open("data/finetune.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
