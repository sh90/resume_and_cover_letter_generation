# few-shot prompting + templates
# app/prompts.py
from jinja2 import Template

RESUME_BULLETS_TMPL = Template("""
You are a resume rewrite assistant.
Given a Job Description and a base resume, output 4-6 bullets for the target role.
Rules:
- Each bullet: <action verb> + <what you did> + <impact> + <metric>.
- Mirror relevant keywords from the JD.
- Keep to one line each, no pronouns, no fluff.

Job Description:
{{ jd }}

Base Resume:
{{ resume }}

Few-shot examples:
{{ examples }}

Now write the bullets:
""")

COVER_LETTER_TMPL = Template("""
You are a cover letter writer. 130–180 words.
- First line: align to the company’s mission/problem from the JD.
- Middle: 2 achievements mapped to JD’s must-haves; quantify impact.
- Close: show enthusiasm + availability.

Job Description:
{{ jd }}

Candidate Highlights:
{{ highlights }}

Few-shot examples:
{{ examples }}

Now write the cover letter:
""")
