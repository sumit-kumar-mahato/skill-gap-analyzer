# agents/lg_nodes/resume_node.py

from agents.resume_agent import resume_agent


def resume_node(state: dict) -> dict:
    """
    LangGraph node: Extract resume skills & evidence.
    Preserves full state.
    """

    if "resume_text" not in state:
        raise KeyError("resume_text missing from state")

    if "api_key" not in state:
        raise KeyError("api_key missing from state")

    resume_data = resume_agent(
        state["resume_text"],
        api_key=state["api_key"]
    )

    return {
        **state,  # 🔥 preserve everything
        "resume_evidence": resume_data.get("evidence", []),
        "last_action": "RESUME_AGENT"
    }
