# agents/evaluation_agent.py

import json
from typing import List, Dict, Any
from utils.groq_client import groq_call


EVALUATION_PROMPT = """
You are a strict skill-evaluation agent.

Your task:
- Compare JOB REQUIREMENTS against the RESUME
- Judge whether each requirement is:
  - MET (clearly demonstrated)
  - PARTIALLY_MET (mentioned but proficiency unclear)
  - MISSING (no convincing evidence)

IMPORTANT RULES:
- Do NOT assume proficiency unless evidence exists
- Projects, tools, and applied usage count as evidence
- Certifications and degrees count as supporting evidence
- Be conservative but fair

Return ONLY valid JSON in the following format:

{
  "met": [
    {
      "requirement": "<requirement>",
      "reason": "<short justification>"
    }
  ],
  "partially_met": [
    {
      "requirement": "<requirement>",
      "reason": "<why it is partial>"
    }
  ],
  "missing": [
    {
      "requirement": "<requirement>",
      "reason": "<why it is missing>"
    }
  ]
}
"""


def evaluate_constraints(
    jd_requirements: List[str],
    resume_text: str,
    api_key: str
) -> Dict[str, Any]:
    """
    LLM-based reasoning agent.
    Overrides embedding-based decisions when necessary.
    """

    if not jd_requirements:
        return {
            "met": [],
            "partially_met": [],
            "missing": []
        }

    requirements_block = "\n".join(f"- {r}" for r in jd_requirements)

    prompt = f"""
{EVALUATION_PROMPT}

JOB REQUIREMENTS:
{requirements_block}

RESUME:
{resume_text}
"""

    raw = groq_call(prompt, api_key)

    clean = (
        raw.replace("```json", "")
           .replace("```", "")
           .strip()
    )

    try:
        parsed = json.loads(clean)

        # ---- SAFETY NORMALIZATION ----
        return {
            "met": parsed.get("met", []),
            "partially_met": parsed.get("partially_met", []),
            "missing": parsed.get("missing", [])
        }

    except Exception:
        # Conservative fallback
        return {
            "met": [],
            "partially_met": [],
            "missing": [
                {
                    "requirement": r,
                    "reason": "Unable to confidently evaluate this requirement"
                }
                for r in jd_requirements
            ]
        }
