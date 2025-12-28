# agents/lg_nodes/recommendation_node.py

from agents.recommendation_agent import recommendation_agent


def recommendation_node(state: dict) -> dict:
    """
    LangGraph node: Generate recruiter-focused recommendations.
    Requires GAP + EVALUATION to have completed.
    Preserves full LangGraph state.
    """

    # ---------------- STATE GUARDS ----------------
    if "missing" not in state:
        raise KeyError("missing skills not found — GAP_AGENT must run first")

    if "final_evaluation" not in state:
        raise KeyError("final_evaluation missing — EVALUATION_AGENT must run first")

    if "api_key" not in state:
        raise KeyError("api_key missing from state")

    missing_skills = [
        m["requirement"] if isinstance(m, dict) else m
        for m in state.get("missing", [])
    ]

    # ---------------- RECOMMENDATION ----------------
    recs = recommendation_agent(
        missing_skills=missing_skills,
        role="Target Role",
        api_key=state["api_key"]
    )

    return {
        **state,  # 🔥 preserve everything
        "recommendations": recs.get("recommendations", []),
        "last_action": "RECOMMENDATION_AGENT"
    }
