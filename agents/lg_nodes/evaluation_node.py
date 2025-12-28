# agents/lg_nodes/evaluation_node.py

from agents.evaluation_agent import evaluate_constraints


def evaluation_node(state: dict) -> dict:
    """
    LangGraph node: Explain WHY requirements are met / partial / missing.
    Requires GAP_AGENT to have run.
    Preserves full LangGraph state.
    """

    # ---------------- STATE GUARDS ----------------
    if "jd_requirements" not in state:
        raise KeyError("jd_requirements missing from state")

    if "resume_text" not in state:
        raise KeyError("resume_text missing from state")

    if "matched" not in state or "missing" not in state:
        raise KeyError("gap results missing (matched / missing not found)")

    if "api_key" not in state:
        raise KeyError("api_key missing from state")

    # ---------------- EVALUATION ----------------
    evaluation = evaluate_constraints(
        jd_requirements=state["jd_requirements"],
        resume_text=state["resume_text"],
        api_key=state["api_key"]
    )

    return {
        **state,  # 🔥 preserve everything
        "final_evaluation": evaluation,
        "last_action": "EVALUATION_AGENT"
    }
