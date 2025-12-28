# agents/gap_agent.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------- LOAD MODEL ONCE ----------------
model = SentenceTransformer("all-MiniLM-L6-v2")


def _normalize(text: str) -> str:
    """
    Light normalization to reduce noise.
    """
    return text.lower().strip()


def gap_agent(
    jd_requirements,
    resume_evidence,
    inferred_skills=None,
    base_threshold=0.65
):
    """
    Improved semantic JD vs Resume comparison.

    Fixes:
    - Long JD sentence vs short resume skill mismatch
    - Uses inferred (parent) skills if available
    - Adaptive similarity threshold
    """

    if not jd_requirements:
        return [], [], 0

    if not resume_evidence:
        return [], jd_requirements, 0

    # ---------------- NORMALIZE INPUTS ----------------
    jd_reqs = [_normalize(r) for r in jd_requirements]
    resume_skills = [_normalize(r) for r in resume_evidence]

    # Add inferred / parent skills if available
    if inferred_skills:
        resume_skills.extend([_normalize(s) for s in inferred_skills])

    # Remove duplicates
    resume_skills = list(set(resume_skills))

    # ---------------- EMBEDDINGS ----------------
    jd_embeddings = model.encode(jd_reqs, convert_to_numpy=True)
    resume_embeddings = model.encode(resume_skills, convert_to_numpy=True)

    matched = []
    missing = []

    # ---------------- MATCHING ----------------
    for i, jd_req in enumerate(jd_reqs):
        similarities = cosine_similarity(
            jd_embeddings[i].reshape(1, -1),
            resume_embeddings
        )[0]

        max_sim = float(np.max(similarities))

        # ---------------- ADAPTIVE THRESHOLD ----------------
        # Short skills (python, sql) → lower threshold
        # Conceptual requirements → slightly higher
        threshold = base_threshold
        if len(jd_req.split()) <= 3:
            threshold -= 0.05  # allow looser match for skill names

        if max_sim >= threshold:
            matched.append(jd_requirements[i])  # original text
        else:
            missing.append(jd_requirements[i])

    match_percentage = int((len(matched) / len(jd_requirements)) * 100)

    return matched, missing, match_percentage
