from __future__ import annotations

import io
import re

import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader

from matcher import extract_keywords, extract_skills, normalize_text, rank_resumes


st.set_page_config(page_title="Talent Lens", page_icon="TL", layout="wide")

SKILLS = [
    "Python", "SQL", "Java", "JavaScript", "TypeScript", "React", "Node.js", "Django",
    "FastAPI", "Flask", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "Linux",
    "Pandas", "NumPy", "scikit-learn", "TensorFlow", "PyTorch", "NLP", "machine learning",
    "deep learning", "REST API", "GraphQL", "PostgreSQL", "MongoDB", "Spark", "Tableau",
    "Power BI", "Agile", "product management", "communication", "leadership",
]

SAMPLE_RESUMES = [
    {
        "name": "Maya Chen",
        "text": "Machine learning engineer with 5 years of experience building NLP systems in Python. "
        "Developed production APIs with FastAPI and Docker, trained models with scikit-learn and PyTorch, "
        "and deployed services on AWS. Strong SQL, NLP, and stakeholder communication skills.",
    },
    {
        "name": "Andre Silva",
        "text": "Data analyst with 4 years of experience using SQL, Pandas, Tableau, and Power BI. "
        "Built dashboards for product teams and automated reporting workflows with Python. Experienced in "
        "A/B testing, data storytelling, and cross-functional communication.",
    },
    {
        "name": "Priya Nair",
        "text": "Full-stack developer working with TypeScript, React, Node.js, PostgreSQL, and GraphQL. "
        "Led Agile delivery for customer-facing platforms, containerized applications with Docker, and "
        "mentored engineers through code reviews and technical planning.",
    },
]


def read_upload(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    extension = uploaded_file.name.lower().rsplit(".", 1)[-1]
    if extension == "pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if extension == "docx":
        document = Document(io.BytesIO(data))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    return data.decode("utf-8", errors="ignore")


def badge(label: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{label}</span>'


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#17202a; --muted:#64707d; --mint:#b8f2d0; --orange:#ff8a5b; --paper:#f7f5ef; --line:#c9c5ba; }
    .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans', sans-serif; }
    .block-container { max-width:1400px; padding-top:2rem; padding-bottom:3rem; }
    .stMarkdown, .stTextArea, .stFileUploader, .stCheckbox { color:var(--ink); }
    label, [data-testid="stWidgetLabel"] p { color:var(--ink) !important; font-weight:700 !important; }
    h1,h2,h3 { font-family:'Space Grotesk', sans-serif !important; letter-spacing:0 !important; }
    .hero { padding: 2.5rem 0 1.2rem; border-bottom:1px solid #d8d5cc; margin-bottom:1.5rem; }
    .eyebrow { color:#e45d2e; font-weight:700; font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; }
    .hero h1 { font-size:3.2rem; line-height:1; margin:.4rem 0 .8rem; max-width:750px; }
    .hero p { color:var(--muted); font-size:1.05rem; max-width:680px; }
    .panel-title { background:#17202a; color:#fff; border-radius:8px; padding:.7rem 1rem; margin:0 0 .8rem; font-family:'Space Grotesk'; font-weight:700; }
    .metric { background:#fff; border:1px solid var(--line); border-radius:8px; padding:1rem; min-height:92px; }
    .metric-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { font-family:'Space Grotesk'; font-size:1.8rem; font-weight:700; margin-top:.25rem; }
    .badge { border-radius:999px; padding:.22rem .6rem; margin:.15rem .2rem .15rem 0; display:inline-block; color:#17202a; font-size:.78rem; font-weight:700; }
    .match-card { background:#fff; border:1px solid var(--line); border-left:5px solid var(--orange); border-radius:8px; padding:1.1rem; margin-bottom:.7rem; }
    .match-score { color:#e45d2e; font-family:'Space Grotesk'; font-size:2rem; font-weight:700; }
    .upload-guide { background:#fff7f2; border:2px solid var(--orange); border-radius:8px; padding:1rem; margin:.7rem 0 .5rem; }
    .upload-guide strong { color:#c84e24; font-size:1rem; }
    .upload-guide span { color:var(--muted); display:block; margin-top:.25rem; font-size:.9rem; }
    [data-testid="stFileUploader"] { background:#fff; border:2px dashed #e45d2e; border-radius:8px; padding:1rem; margin-bottom:.6rem; }
    [data-testid="stFileUploaderDropzone"] { background:#fffaf7; min-height:135px; }
    [data-testid="stFileUploaderDropzoneInstructions"] { color:var(--ink) !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] small { color:var(--muted) !important; }
    [data-testid="stFileUploader"] button { background:#e45d2e !important; color:#fff !important; border:0 !important; font-weight:700 !important; }
    .upload-status { color:#176b42; font-weight:700; padding:.35rem 0 .8rem; }
    </style>
    """, unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><div class="eyebrow">Talent intelligence / NLP screening</div>'
    '<h1>Find the signal in every resume.</h1>'
    '<p>Talent Lens ranks candidates against a role using explainable semantic similarity, skill overlap, and evidence snippets.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Talent Lens")
    st.caption("A lightweight, explainable screening workspace")
    st.divider()
    st.markdown("**Scoring model**")
    st.caption("65% semantic similarity · 35% required skill coverage")
    st.markdown("**Supported files**")
    st.caption("TXT, PDF, and DOCX resumes")

left, right = st.columns([1.05, 1.4], gap="large")
with left:
    st.markdown('<div class="panel-title">1 &nbsp; Define the role</div>', unsafe_allow_html=True)
    job_text = st.text_area(
        "Job description",
        value="We are looking for a Machine Learning Engineer to build NLP products. The role requires Python, SQL, machine learning, NLP, scikit-learn, PyTorch, FastAPI, Docker, AWS, and strong communication.",
        height=190,
        label_visibility="collapsed",
    )
    job_skills = extract_skills(job_text, SKILLS)
    st.markdown("**Detected requirements**")
    st.markdown(" ".join(badge(skill, "#b8f2d0") for skill in job_skills) or "No catalog skills detected", unsafe_allow_html=True)
    st.markdown('<div class="panel-title">2 &nbsp; Add candidate resumes</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-guide"><strong>Upload resumes here</strong><span>Choose one or more TXT, PDF, or DOCX files. Your files are processed locally by this app.</span></div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Select resume files", type=["txt", "pdf", "docx"], accept_multiple_files=True, help="Supported formats: TXT, PDF, DOCX")
    if uploaded_files:
        st.markdown(f'<div class="upload-status">Loaded {len(uploaded_files)} resume file(s)</div>', unsafe_allow_html=True)
    else:
        st.caption("No files uploaded yet. Demo candidates are shown below when enabled.")
    use_demo = st.checkbox("Include demo candidates", value=not bool(uploaded_files))

resumes = []
if use_demo:
    resumes.extend(SAMPLE_RESUMES)
if uploaded_files:
    for uploaded_file in uploaded_files:
        resumes.append({"name": uploaded_file.name, "text": read_upload(uploaded_file)})

with right:
    st.markdown('<div class="panel-title">Screening overview</div>', unsafe_allow_html=True)
    if not job_text.strip() or not resumes:
        st.info("Add a job description and at least one resume to see ranked matches.")
    else:
        ranked = rank_resumes(resumes, job_text, SKILLS)
        top_score = ranked[0]["result"].score
        average_score = sum(item["result"].score for item in ranked) / len(ranked)
        metric_cols = st.columns(3)
        for column, label, value in zip(metric_cols, ["Candidates", "Top match", "Average fit"], [len(ranked), f"{top_score:.1f}%", f"{average_score:.1f}%"]):
            column.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
        st.write("")
        for index, candidate in enumerate(ranked, start=1):
            result = candidate["result"]
            with st.container():
                st.markdown(f'<div class="match-card"><span class="match-score">{result.score:.1f}%</span> &nbsp; <strong>{index}. {candidate["name"]}</strong></div>', unsafe_allow_html=True)
                detail_cols = st.columns([1, 1])
                with detail_cols[0]:
                    st.markdown("**Matched skills**")
                    st.markdown(" ".join(badge(skill, "#b8f2d0") for skill in result.matched_skills) or "None", unsafe_allow_html=True)
                with detail_cols[1]:
                    st.markdown("**Skill gaps**")
                    st.markdown(" ".join(badge(skill, "#ffd3c2") for skill in result.missing_skills) or "None", unsafe_allow_html=True)
                if result.evidence:
                    with st.expander("View evidence"):
                        for snippet in result.evidence:
                            st.write(f"“{snippet}”")

    if job_text.strip() and resumes:
        st.subheader("Keyword intelligence")
        keyword_data = pd.DataFrame({"Keyword": extract_keywords(job_text), "Frequency": [len(re.findall(rf"\b{re.escape(word)}\b", job_text.lower())) for word in extract_keywords(job_text)]})
        st.bar_chart(keyword_data.set_index("Keyword"), color="#ff8a5b", height=220)
