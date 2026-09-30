"""Explainable NLP utilities for resume screening and job matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with", "will",
}


@dataclass
class MatchResult:
    score: float
    matched_skills: list[str]
    missing_skills: list[str]
    evidence: list[str]


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving readable phrases for evidence."""
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def extract_skills(text: str, skill_catalog: list[str]) -> list[str]:
    normalized = normalize_text(text).lower()
    found = []
    for skill in skill_catalog:
        pattern = rf"(?<![a-z0-9+#.]){re.escape(skill.lower())}(?![a-z0-9+#.])"
        if re.search(pattern, normalized):
            found.append(skill)
    return found


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z+#.-]{2,}", normalize_text(text).lower())
    counts: dict[str, int] = {}
    for word in words:
        if word not in STOP_WORDS:
            counts[word] = counts.get(word, 0) + 1
    return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _evidence_snippets(resume_text: str, skills: list[str]) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", normalize_text(resume_text))
    snippets = []
    for sentence in sentences:
        if any(skill.lower() in sentence.lower() for skill in skills):
            snippets.append(sentence.strip())
    return snippets[:4]


def calculate_match(resume_text: str, job_text: str, skill_catalog: list[str]) -> MatchResult:
    """Calculate a blended semantic and skill overlap score with evidence."""
    resume = normalize_text(resume_text)
    job = normalize_text(job_text)
    if not resume or not job:
        return MatchResult(0.0, [], extract_skills(job, skill_catalog), [])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([resume, job])
    semantic_score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])

    job_skills = extract_skills(job, skill_catalog)
    resume_skills = extract_skills(resume, skill_catalog)
    matched = [skill for skill in job_skills if skill in resume_skills]
    missing = [skill for skill in job_skills if skill not in resume_skills]
    skill_score = len(matched) / len(job_skills) if job_skills else semantic_score
    blended_score = (semantic_score * 0.65) + (skill_score * 0.35)

    return MatchResult(
        score=round(min(blended_score, 1.0) * 100, 1),
        matched_skills=matched,
        missing_skills=missing,
        evidence=_evidence_snippets(resume, matched),
    )


def rank_resumes(resumes: list[dict[str, str]], job_text: str, skill_catalog: list[str]) -> list[dict]:
    ranked = []
    for resume in resumes:
        result = calculate_match(resume["text"], job_text, skill_catalog)
        ranked.append({**resume, "result": result})
    return sorted(ranked, key=lambda item: item["result"].score, reverse=True)
