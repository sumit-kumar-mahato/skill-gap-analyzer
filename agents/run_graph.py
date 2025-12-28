# agents/run_graph.py

from typing import Dict, Any
from langgraph.graph import StateGraph, END

from agents.lg_nodes.jd_node import jd_node
from agents.lg_nodes.resume_node import resume_node
from agents.lg_nodes.skill_rag_node import skill_rag_node
from agents.lg_nodes.gap_node import gap_node
from agents.lg_nodes.evaluation_node import evaluation_node
from agents.lg_nodes.recommendation_node import recommendation_node
from agents.lg_nodes.chat_node import chat_node
from agents.lg_nodes.orchestrator_router import orchestrator_router


def run_skill_gap_graph(
    resume_text: str,
    jd_text: str,
    api_key: str
) -> Dict[str, Any]:
    """
    Entry point for LangGraph-powered Skill Gap Analyzer
    """

    # ---------------- INITIAL GRAPH STATE ----------------
    initial_state: Dict[str, Any] = {
        "resume_text": resume_text,
        "jd_text": jd_text,
        "api_key": api_key,                 # MUST persist
        "orchestrator_trace": [],
        "next_action": None,
        "final_evaluation": {},
        "confidence": 0.0,
        "is_done": False
    }

    # ---------------- BUILD GRAPH ----------------
    graph = StateGraph(dict)

    # Nodes
    graph.add_node("ORCHESTRATOR", orchestrator_router)
    graph.add_node("JD_AGENT", jd_node)
    graph.add_node("RESUME_AGENT", resume_node)
    graph.add_node("SKILL_RAG", skill_rag_node)
    graph.add_node("GAP_AGENT", gap_node)
    graph.add_node("EVALUATION_AGENT", evaluation_node)
    graph.add_node("RECOMMENDATION_AGENT", recommendation_node)
    graph.add_node("CHAT_AGENT", chat_node)

    # Entry point
    graph.set_entry_point("ORCHESTRATOR")

    # ---------------- CONDITIONAL ROUTING ----------------
    graph.add_conditional_edges(
        "ORCHESTRATOR",
        lambda state: state["next_action"],
        {
            "JD_AGENT": "JD_AGENT",
            "RESUME_AGENT": "RESUME_AGENT",
            "SKILL_RAG": "SKILL_RAG",
            "GAP_AGENT": "GAP_AGENT",
            "EVALUATION_AGENT": "EVALUATION_AGENT",
            "RECOMMENDATION_AGENT": "RECOMMENDATION_AGENT",
            "CHAT_AGENT": "CHAT_AGENT",

            # 🔥 CRITICAL FIXES
            "HUMAN": END,   # pause → return control to UI
            "DONE": END     # terminate graph
        }
    )

    # Every agent returns control to orchestrator
    for node in [
        "JD_AGENT",
        "RESUME_AGENT",
        "SKILL_RAG",
        "GAP_AGENT",
        "EVALUATION_AGENT",
        "RECOMMENDATION_AGENT",
        "CHAT_AGENT"
    ]:
        graph.add_edge(node, "ORCHESTRATOR")

    compiled_graph = graph.compile()

    # ---------------- RUN GRAPH ----------------
    final_state = compiled_graph.invoke(initial_state)
    return final_state
