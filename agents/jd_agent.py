import json
from utils.prompts import JD_PROMPT
from utils.groq_client import groq_call


def jd_agent(jd_text: str, api_key: str):
    """
    Extracts explicit job requirements from a job description.
    """

    prompt = JD_PROMPT + "\n\nJOB DESCRIPTION:\n" + jd_text

    raw = groq_call(prompt, api_key)

    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean)
    except Exception:
        return {"requirements": []}
