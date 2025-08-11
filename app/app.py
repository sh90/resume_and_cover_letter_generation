import streamlit as st
from prompts import RESUME_BULLETS_TMPL, COVER_LETTER_TMPL
from llm import generate_text
from docx import Document
import PyPDF2

def read_file_contents(uploaded_file):
    if uploaded_file is None:
        return ""
    file_type = uploaded_file.type
    if file_type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif file_type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return f"[Unsupported file type: {file_type}]"

st.set_page_config(page_title="Resume & Cover Letter Generator", page_icon="🧰", layout="wide")
st.title("🎯 Personalized Resume & Cover Letter")

# --- Job Description input ---
jd_file = st.file_uploader("Upload Job Description (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
jd_text = read_file_contents(jd_file)
jd = st.text_area("Paste or Edit Job Description", value=jd_text, height=220)

# --- Base Resume input ---
resume_file = st.file_uploader("Upload Base Resume (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
resume_text = read_file_contents(resume_file)
base_resume = st.text_area("Paste or Edit Base Resume", value=resume_text, height=220)

# --- Few-shot Examples input ---
examples_file = st.file_uploader("Upload Few-shot Examples (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
examples_text = read_file_contents(examples_file)
fewshot = st.text_area("Few-shot examples (optional)", value=examples_text, height=150)

col1, col2 = st.columns(2)
with col1:
    if st.button("Generate Tailored Bullets"):
        prompt = RESUME_BULLETS_TMPL.render(jd=jd, resume=base_resume, examples=fewshot)
        out = generate_text(prompt)
        st.subheader("Tailored Bullets")
        st.write(out)

with col2:
    if st.button("Generate Cover Letter"):
        highlights = "\n".join([l for l in base_resume.splitlines() if l.strip().startswith("-")][:6])
        prompt = COVER_LETTER_TMPL.render(jd=jd, highlights=highlights, examples=fewshot)
        out = generate_text(prompt)
        st.subheader("Cover Letter")
        st.write(out)
