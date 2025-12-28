# agents/lg_nodes/jd_node.py

from agents.jd_agent import jd_agent


def jd_node(state: dict) -> dict:
    """
    LangGraph node: Extract job requirements from JD text.
    Preserves full state.
    """

    if "jd_text" not in state:
        raise KeyError("jd_text missing from state")

    if "api_key" not in state:
        raise KeyError("api_key missing from state")

    jd_data = jd_agent(
        state["jd_text"],
        api_key=state["api_key"]
    )

    return {
        **state,  # 🔥 preserve everything
        "jd_requirements": jd_data.get("requirements", []),
        "last_action": "JD_AGENT"
    }
