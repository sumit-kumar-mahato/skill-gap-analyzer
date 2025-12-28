# agents/orchestrator.py

import json
from typing import Dict, Any
from utils.groq_client import groq_call


# =========================================================
# ORCHESTRATOR PROMPT
# =========================================================

ORCHESTRATOR_PROMPT = """
You are the ORCHESTRATOR of an AGENTIC AI system called
"Skill Gap Analyzer (Resume vs Job Description)".

Your responsibility:
- Observe the CURRENT STATE
- Decide the NEXT_ACTION
- Explain WHY this action is chosen
- Explain WHY other actions are NOT chosen
- You DO NOT execute agents
- You ONLY decide which agent/tool should act next

--------------------------------------------------
AVAILABLE AGENTS / TOOLS
--------------------------------------------------

CORE ANALYSIS AGENTS:
- JD_AGENT: extract job requirements from job description
- RESUME_AGENT: extract skills, tools, and evidence from resume
- SKILL_RAG: infer high-level skills from low-level resume evidence
- GAP_AGENT: compute semantic match and missing skills
- EVALUATION_AGENT: explain WHY requirements are met or missing
- RECOMMENDATION_AGENT: suggest learning paths for missing skills

INTERACTION AGENTS:
- CHAT_AGENT: answer user questions using the current analysis state
- HUMAN: ask the user for clarification when ambiguity exists

CONTROL ACTION:
- DONE: stop when analysis is sufficient

--------------------------------------------------
IMPORTANT RULES
--------------------------------------------------
- Do NOT assume any fixed order of agents
- Choose agents only if required information is missing
- Do NOT repeat an agent if its output already exists and is sufficient
- Prefer SKILL_RAG if resume evidence is too low-level or abstract
- Choose HUMAN if:
  - skill evidence is ambiguous
  - confidence is borderline (e.g. 40–60%)
  - JD requirements are unclear
- Choose CHAT_AGENT ONLY when:
  - a user question exists in the state (chat_question)
- Choose DONE ONLY when:
  - skill gaps are identified
  - reasoning is complete
  - recommendations (if needed) are generated

--------------------------------------------------
OUTPUT FORMAT (STRICT JSON)
--------------------------------------------------
{
  "next_action": "<ONE_OF_THE_ALLOWED_ACTIONS>",
  "reason": "<short explanation of why this action was chosen>",
  "rejected_actions": {
    "<ACTION_NAME>": "<short reason for not choosing this action>"
  }
}

--------------------------------------------------
ALLOWED ACTIONS
--------------------------------------------------
["JD_AGENT", "RESUME_AGENT", "SKILL_RAG", "GAP_AGENT",
 "EVALUATION_AGENT", "RECOMMENDATION_AGENT",
 "CHAT_AGENT", "HUMAN", "DONE"]

--------------------------------------------------
CURRENT STATE
--------------------------------------------------
"""


# =========================================================
# ORCHESTRATOR DECISION FUNCTION
# =========================================================

def decide_next_action(state: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    LLM-driven planner.
    Decides which agent/tool should act next
    and explains why other agents were rejected.
    """

    prompt = ORCHESTRATOR_PROMPT + json.dumps(state, indent=2)

    raw = groq_call(prompt, api_key)

    try:
        decision = json.loads(raw)
    except Exception:
        # Fail-safe fallback
        return {
            "next_action": "HUMAN",
            "reason": "Failed to parse orchestrator output",
            "rejected_actions": {}
        }

    allowed_actions = {
        "JD_AGENT",
        "RESUME_AGENT",
        "SKILL_RAG",
        "GAP_AGENT",
        "EVALUATION_AGENT",
        "RECOMMENDATION_AGENT",
        "CHAT_AGENT",
        "HUMAN",
        "DONE"
    }

    action = decision.get("next_action")

    if action not in allowed_actions:
        return {
            "next_action": "HUMAN",
            "reason": f"Invalid action chosen: {action}",
            "rejected_actions": {}
        }

    return {
        "next_action": action,
        "reason": decision.get("reason", ""),
        "rejected_actions": decision.get("rejected_actions", {})
    }
