# agents/actions.py

from typing import Dict, Any

from agents.jd_agent import jd_agent
from agents.resume_agent import resume_agent
from agents.gap_agent import gap_agent
from agents.evaluation_agent import evaluate_constraints
from agents.recommendation_agent import recommendation_agent
from agents.chat_agent import chat_agent
from rag.skill_rag import infer_parent_skills


def execute_action(action: str, state: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    Executes ONE agent/tool chosen by the orchestrator.
    This function does NOT decide order.
    It only mutates and enriches the shared state.
    """

    # ---------------- JD AGENT ----------------
    if action == "JD_AGENT":
        jd_data = jd_agent(state["jd_text"], api_key)
        state["jd_requirements"] = jd_data.get("requirements", [])

    # ---------------- RESUME AGENT ----------------
    elif action == "RESUME_AGENT":
        resume_data = resume_agent(state["resume_text"], api_key)
        state["resume_evidence"] = resume_data.get("evidence", [])

    # ---------------- SKILL RAG AGENT ----------------
    elif action == "SKILL_RAG":
        state["inferred_skills"] = infer_parent_skills(
            state.get("resume_evidence", [])
        )

    # ---------------- GAP AGENT ----------------
    elif action == "GAP_AGENT":
        matched, missing, match_pct = gap_agent(
            jd_requirements=state.get("jd_requirements", []),
            resume_evidence=state.get("resume_evidence", []),
            inferred_skills=state.get("inferred_skills", [])
        )
        state["matched"] = matched
        state["missing"] = missing
        state["confidence"] = match_pct / 100

    # ---------------- EVALUATION + MERGE AGENT ----------------
    elif action == "EVALUATION_AGENT":

        evaluation = evaluate_constraints(
            state.get("jd_requirements", []),
            state.get("resume_text", ""),
            api_key
        )

        # GAP results
        gap_matched = set(state.get("matched", []))
        gap_missing = set(state.get("missing", []))

        final_met = {}
        final_partial = {}
        final_missing = {}

        # -------- Evaluation MET --------
        for item in evaluation.get("met", []):
            final_met[item["requirement"]] = item["reason"]

        # -------- Evaluation PARTIAL --------
        for item in evaluation.get("partially_met", []):
            if item["requirement"] not in final_met:
                final_partial[item["requirement"]] = item["reason"]

        # -------- Evaluation MISSING --------
        for item in evaluation.get("missing", []):
            if (
                item["requirement"] not in final_met
                and item["requirement"] not in final_partial
            ):
                final_missing[item["requirement"]] = item["reason"]

        # -------- GAP FALLBACK LOGIC --------
        for r in gap_matched:
            if r not in final_met and r not in final_partial:
                final_partial[r] = (
                    "Semantically matched, but explicit proficiency or depth is unclear"
                )

        for r in gap_missing:
            if r not in final_met and r not in final_partial:
                final_missing[r] = (
                    "No strong semantic or contextual evidence found in resume"
                )

        # -------- SAVE FINAL MERGED RESULT --------
        state["final_evaluation"] = {
            "met": [
                {"requirement": k, "reason": v}
                for k, v in final_met.items()
            ],
            "partially_met": [
                {"requirement": k, "reason": v}
                for k, v in final_partial.items()
            ],
            "missing": [
                {"requirement": k, "reason": v}
                for k, v in final_missing.items()
            ]
        }

    # ---------------- RECOMMENDATION AGENT ----------------
    elif action == "RECOMMENDATION_AGENT":
        recs = recommendation_agent(
            missing_skills=[
                item["requirement"]
                for item in state.get("final_evaluation", {}).get("missing", [])
            ],
            role="Target Role",
            api_key=api_key
        )
        state["recommendations"] = recs.get("recommendations", [])

    # ---------------- CHAT AGENT ----------------
    elif action == "CHAT_AGENT":
        state["chat_answer"] = chat_agent(
            question=state.get("chat_question", ""),
            context=state,
            api_key=api_key
        )

    # ---------------- BOOKKEEPING ----------------
    state["last_action"] = action

    return state
