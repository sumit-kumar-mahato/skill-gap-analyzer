# agents/lg_nodes/gap_node.py

from agents.gap_agent import gap_agent


def gap_node(state: dict) -> dict:
    """
    LangGraph node: Semantic gap analysis.
    Preserves full graph state and fails fast if inputs are missing.
    """

    if "jd_requirements" not in state:
        raise KeyError("jd_requirements missing from state")

    if "resume_evidence" not in state:
        raise KeyError("resume_evidence missing from state")

    matched, missing, match_pct = gap_agent(
        jd_requirements=state.get("jd_requirements", []),
        resume_evidence=state.get("resume_evidence", []),
        inferred_skills=state.get("inferred_skills", [])
    )

    return {
        **state,  # 🔥 preserve everything
        "matched": matched,
        "missing": missing,
        "confidence": round(match_pct / 100, 2),
        "last_action": "GAP_AGENT"
    }
