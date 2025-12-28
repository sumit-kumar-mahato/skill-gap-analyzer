# agents/lg_nodes/orchestrator_router.py

from agents.orchestrator import decide_next_action


def orchestrator_router(state: dict) -> dict:
    """
    LangGraph ORCHESTRATOR node.

    Responsibilities:
    - Decide which agent should run next
    - Decide WHEN the system is done
    - Log decision + reasoning
    - Preserve full graph state
    """

    # ---------------- SAFETY CHECK ----------------
    if "api_key" not in state:
        raise KeyError("api_key missing from LangGraph state")

    # ---------------- ORCHESTRATOR DECISION ----------------
    decision = decide_next_action(
        state,
        api_key=state["api_key"]
    )

    chosen_action = decision["next_action"]
    reason = decision.get("reason", "N/A")

    # ---------------- TRACE LOG ----------------
    trace = state.get("orchestrator_trace", [])

    trace.append({
        "step": len(trace) + 1,
        "chosen_action": chosen_action,
        "reason_for_choice": reason,
        "state_snapshot": {
            "has_jd_requirements": bool(state.get("jd_requirements")),
            "has_resume_evidence": bool(state.get("resume_evidence")),
            "has_skill_rag": bool(state.get("inferred_skills")),
            "has_gap_analysis": bool(state.get("matched") or state.get("missing")),
            "has_evaluation": bool(state.get("final_evaluation")),
            "confidence": round(state.get("confidence", 0), 2)
        }
    })

    # ---------------- TERMINATION LOGIC ----------------
    is_done = chosen_action == "DONE"

    # ---------------- RETURN FULL STATE ----------------
    return {
        **state,                        # 🔥 preserve EVERYTHING
        "next_action": chosen_action,   # used by conditional routing
        "planner_reason": reason,
        "last_action": chosen_action,
        "orchestrator_trace": trace,
        "is_done": is_done               # 🔥 CRITICAL FIX
    }
