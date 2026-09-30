# Talent Lens

An explainable AI-based resume screener and job matcher built with Python, Streamlit, and NLP.

## What it does

- Accepts job descriptions and TXT, PDF, or DOCX resumes.
- Uses TF-IDF n-gram similarity to compare resume language with role requirements.
- Detects a configurable skill catalog and calculates required skill coverage.
- Blends semantic similarity (65%) and skill coverage (35%) into a ranking score.
- Shows matched skills, skill gaps, keyword intelligence, and resume evidence snippets.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, file parsing, and presentation |
| `matcher.py` | Explainable NLP scoring and ranking logic |
| `requirements.txt` | Runtime dependencies |

## Responsible screening notes

This tool is a decision-support prototype, not an autonomous hiring decision-maker. Review evidence manually, remove protected or irrelevant attributes from inputs, audit score behavior on representative data, and keep a human in the loop.
