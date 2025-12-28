import json
from utils.prompts import RESUME_PROMPT
from utils.groq_client import groq_call


def resume_agent(resume_text: str, api_key: str):
    """
    Extracts explicit skills, tools, technologies, and experience
    mentioned in a resume using Groq (LLaMA-3).
    """

    prompt = RESUME_PROMPT + "\n\nRESUME:\n" + resume_text

    raw = groq_call(prompt, api_key)

    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean)
    except Exception:
        return {"evidence": []}
