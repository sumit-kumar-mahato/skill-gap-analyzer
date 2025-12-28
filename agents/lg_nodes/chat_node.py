# agents/lg_nodes/chat_node.py

from agents.chat_agent import chat_agent


def chat_node(state: dict) -> dict:
    """
    LangGraph node: Recruiter-style Q&A.
    Reactive agent — runs ONLY when a user question exists.
    Preserves full LangGraph state.
    """

    # ---------------- STATE GUARDS ----------------
    if "chat_question" not in state or not state["chat_question"].strip():
        raise KeyError(
            "chat_question missing — CHAT_AGENT should only run when user asks a question"
        )

    if "api_key" not in state:
        raise KeyError("api_key missing from state")

    # Optional but recommended: ensure analysis exists
    if "final_evaluation" not in state:
        raise KeyError(
            "final_evaluation missing — CHAT_AGENT requires completed analysis"
        )

    # ---------------- CHAT RESPONSE ----------------
    answer = chat_agent(
        question=state["chat_question"],
        context=state,
        api_key=state["api_key"]
    )

    return {
        **state,  # 🔥 preserve entire graph state
        "chat_answer": answer,
        "last_action": "CHAT_AGENT"
    }
