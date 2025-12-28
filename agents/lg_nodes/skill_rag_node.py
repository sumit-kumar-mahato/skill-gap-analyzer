# agents/lg_nodes/skill_rag_node.py

from rag.skill_rag import infer_parent_skills


def skill_rag_node(state: dict) -> dict:
    """
    LangGraph node: Infer high-level skills from resume evidence.
    Preserves full graph state.
    """

    if "resume_evidence" not in state:
        raise KeyError("resume_evidence missing from state")

    evidence = state.get("resume_evidence", [])

    inferred = infer_parent_skills(evidence)

    return {
        **state,  # 🔥 preserve everything
        "inferred_skills": inferred,
        "last_action": "SKILL_RAG"
    }
