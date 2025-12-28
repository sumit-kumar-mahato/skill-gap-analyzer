import json
from utils.prompts import RECOMMENDATION_PROMPT
from utils.groq_client import groq_call


def recommendation_agent(missing_skills, role, api_key):
    """
    Generates learning recommendations for missing skills
    using Groq (LLaMA-3) instead of Gemini.
    """

    if not missing_skills:
        return {"recommendations": []}

    prompt = f"""
{RECOMMENDATION_PROMPT}

TARGET ROLE:
{role}

MISSING SKILLS:
{missing_skills}
"""

    raw = groq_call(prompt, api_key)

    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean)
    except Exception:
        return {"recommendations": []}
