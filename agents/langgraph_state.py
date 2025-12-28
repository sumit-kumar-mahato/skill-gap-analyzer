# agents/langgraph_state.py

from typing import TypedDict, List, Dict, Any, Optional


class SkillGapState(TypedDict, total=False):
    """
    Shared state for LangGraph-based Skill Gap Analyzer.
    This is the single source of truth across all agents.
    """

    # ---------------- INPUT ----------------
    resume_text: str
    jd_text: str

    # ---------------- EXTRACTION ----------------
    jd_requirements: List[str]
    resume_evidence: List[str]
    inferred_skills: List[str]

    # ---------------- ANALYSIS ----------------
    matched: List[str]
    partially_met: List[Dict[str, str]]
    missing: List[Dict[str, str]]
    confidence: float

    # ---------------- EVALUATION ----------------
    final_evaluation: Dict[str, Any]

    # ---------------- RECOMMENDATION ----------------
    recommendations: List[Dict[str, Any]]

    # ---------------- CHAT ----------------
    chat_question: str
    chat_answer: str

    # ---------------- HUMAN IN LOOP ----------------
    human_question: str
    human_response: str

    # ---------------- ORCHESTRATION ----------------
    last_action: str
    planner_reason: str

    orchestrator_trace: List[Dict[str, Any]]
    done: bool
