import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------ LOAD EMBEDDING MODEL ------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ LOAD SKILL ONTOLOGY SAFELY ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_PATH = os.path.join(BASE_DIR, "skill_ontology.json")

with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
    SKILL_DB = json.load(f)

# ------------------ FLATTEN SKILLS ------------------
PARENT_SKILLS = list(SKILL_DB.keys())
CHILD_SKILLS = [child for children in SKILL_DB.values() for child in children]

ALL_SKILLS = PARENT_SKILLS + CHILD_SKILLS

# ------------------ BUILD FAISS INDEX (ONCE) ------------------
skill_vectors = model.encode(ALL_SKILLS, convert_to_numpy=True)
index = faiss.IndexFlatL2(skill_vectors.shape[1])
index.add(skill_vectors)

# ------------------ INFER PARENT SKILLS ------------------
def infer_parent_skills(resume_skills, threshold=0.65):
    """
    Infer high-level (parent) skills from low-level resume skills
    using semantic similarity + ontology mapping.
    """

    if not resume_skills:
        return []

    inferred_parents = set()

    resume_vectors = model.encode(resume_skills, convert_to_numpy=True)

    for i in range(len(resume_skills)):
        distances, indices = index.search(resume_vectors[i:i+1], k=3)

        for idx, dist in zip(indices[0], distances[0]):
            similarity = 1 / (1 + dist)

            if similarity >= threshold:
                matched_skill = ALL_SKILLS[idx]

                # Map child → parent
                for parent, children in SKILL_DB.items():
                    if matched_skill in children:
                        inferred_parents.add(parent)

    return sorted(list(inferred_parents))
