# agents/langgraph_graph.py

from langgraph.graph import StateGraph, END

from agents.langgraph_state import SkillGapState

# ---------------- ROUTER ----------------
from agents.lg_nodes.orchestrator_router import orchestrator_router

# ---------------- AGENT NODES ----------------
from agents.lg_nodes.jd_node import jd_node
from agents.lg_nodes.resume_node import resume_node
from agents.lg_nodes.skill_rag_node import skill_rag_node
from agents.lg_nodes.gap_node import gap_node
from agents.lg_nodes.evaluation_node import evaluation_node
from agents.lg_nodes.recommendation_node import recommendation_node
from agents.lg_nodes.chat_node import chat_node


def build_skill_gap_graph():
    """
    Builds and returns the LangGraph-based
    Skill Gap Analyzer agentic graph.
    """

    graph = StateGraph(SkillGapState)

    # ==================================================
    # REGISTER NODES
    # ==================================================

    graph.add_node("ORCHESTRATOR", orchestrator_router)

    graph.add_node("JD_AGENT", jd_node)
    graph.add_node("RESUME_AGENT", resume_node)
    graph.add_node("SKILL_RAG", skill_rag_node)
    graph.add_node("GAP_AGENT", gap_node)
    graph.add_node("EVALUATION_AGENT", evaluation_node)
    graph.add_node("RECOMMENDATION_AGENT", recommendation_node)
    graph.add_node("CHAT_AGENT", chat_node)

    # ==================================================
    # ENTRY POINT
    # ==================================================

    graph.set_entry_point("ORCHESTRATOR")

    # ==================================================
    # CONDITIONAL ROUTING (THE MAGIC)
    # ==================================================

    graph.add_conditional_edges(
        "ORCHESTRATOR",
        orchestrator_router,
        {
            "JD_AGENT": "JD_AGENT",
            "RESUME_AGENT": "RESUME_AGENT",
            "SKILL_RAG": "SKILL_RAG",
            "GAP_AGENT": "GAP_AGENT",
            "EVALUATION_AGENT": "EVALUATION_AGENT",
            "RECOMMENDATION_AGENT": "RECOMMENDATION_AGENT",
            "CHAT_AGENT": "CHAT_AGENT",
            "HUMAN": END,   # pause handled in UI
            "DONE": END
        }
    )

    # ==================================================
    # LOOP BACK TO ORCHESTRATOR
    # ==================================================

    graph.add_edge("JD_AGENT", "ORCHESTRATOR")
    graph.add_edge("RESUME_AGENT", "ORCHESTRATOR")
    graph.add_edge("SKILL_RAG", "ORCHESTRATOR")
    graph.add_edge("GAP_AGENT", "ORCHESTRATOR")
    graph.add_edge("EVALUATION_AGENT", "ORCHESTRATOR")
    graph.add_edge("RECOMMENDATION_AGENT", "ORCHESTRATOR")
    graph.add_edge("CHAT_AGENT", "ORCHESTRATOR")

    # ==================================================
    # COMPILE GRAPH
    # ==================================================

    return graph.compile()
