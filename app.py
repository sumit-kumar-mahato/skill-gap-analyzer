# app.py

import streamlit as st
import streamlit.components.v1 as components

from utils.pdf_parser import extract_text_from_pdf
from agents.run_graph import run_skill_gap_graph
from agents.actions import execute_action
from utils.groq_client import groq_call
from utils.prompts import JD_SUMMARY_PROMPT, RESUME_SUMMARY_PROMPT

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Skill Gap Analyzer — Agentic AI (LangGraph)",
    page_icon="🧠",
    layout="wide"
)

# ------------------ HEADER ------------------
st.title("🧠 Skill Gap Analyzer (Agentic AI — LangGraph)")
st.caption(
    "LangGraph-powered agentic system with explainable orchestration "
    "and recruiter-focused human-in-the-loop decisions"
)

st.markdown("---")

API_KEY = st.secrets["GROQ_API_KEY"]

# ------------------ SESSION STATE ------------------
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None

# ------------------ INPUT SECTION ------------------
col1, col2 = st.columns([1, 2])

with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

with col2:
    jd_text = st.text_area("📋 Paste Job Description", height=220)

st.markdown("---")

# ------------------ MERMAID HELPER ------------------
def build_mermaid_from_trace(trace: list) -> str:
    if not trace:
        return ""

    lines = ["flowchart TD"]
    prev = "ORCHESTRATOR"

    for step in trace:
        action = step.get("chosen_action", "UNKNOWN")

        if action == "DONE":
            lines.append(f"{prev} --> DONE")
            break

        lines.append(f"{prev} --> {action}")
        lines.append(f"{action} --> ORCHESTRATOR")
        prev = action

    return "\n".join(lines)

def render_mermaid(mermaid_code: str):
    if not mermaid_code:
        st.info("Mermaid graph unavailable.")
        return

    html = f"""
    <div class="mermaid">
    {mermaid_code}
    </div>

    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
    </script>
    """

    components.html(html, height=500, scrolling=True)

# ================= AGENTIC EXECUTION =================
if st.button("🔍 Analyze Skill Gap", use_container_width=True):

    if not resume_file or not jd_text.strip():
        st.error("❗ Please upload a resume and paste a job description.")
        st.stop()

    with st.spinner("📖 Reading resume..."):
        resume_text = extract_text_from_pdf(resume_file)

    with st.spinner("🧠 Running LangGraph agentic reasoning..."):
        st.session_state.agent_state = run_skill_gap_graph(
            resume_text=resume_text,
            jd_text=jd_text,
            api_key=API_KEY
        )

# ================= RESULTS DISPLAY =================
if st.session_state.agent_state:

    result = st.session_state.agent_state
    evaluation = result.get("final_evaluation") or {}

    # Always define these
    met = list(evaluation.get("met", []))
    partial = list(evaluation.get("partially_met", []))
    missing = list(evaluation.get("missing", []))

    trace = result.get("orchestrator_trace", [])

    # ==================================================
    # 🧠 ORCHESTRATOR DECISION TIMELINE
    # ==================================================
    st.markdown("## 🧠 Orchestrator Decision Timeline")

    if not trace:
        st.info("No orchestration trace available.")
    else:
        for step in trace:
            with st.expander(
                f"Step {step.get('step')} → {step.get('chosen_action')}",
                expanded=False
            ):
                st.markdown("**Why this agent was chosen:**")
                st.markdown(step.get("reason_for_choice", "N/A"))

                st.markdown("**State snapshot:**")
                st.json(step.get("state_snapshot", {}))

        st.markdown("### 🔁 Execution Flow")
        st.code(" → ".join([t.get("chosen_action", "?") for t in trace]))

    # ==================================================
    # 🔀 MERMAID GRAPH VISUALIZATION (FIXED)
    # ==================================================
    st.markdown("---")
    st.markdown("## 🔀 Orchestrator Execution Graph")

    mermaid_code = build_mermaid_from_trace(trace)
    render_mermaid(mermaid_code)

    # ==================================================
    # 🤝 HUMAN-IN-THE-LOOP POLICY
    # ==================================================
    st.markdown("---")
    st.markdown("## 🤝 Recruiter Review Policy")

    hitl_choice = st.radio(
        "How should partially matched skills be treated?",
        [
            "🟡 Conservative – keep AI decision",
            "🟢 Trust resume – treat partial as met",
            "🔴 Strict – treat partial as missing"
        ],
        index=0
    )

    if hitl_choice.startswith("🟢"):
        for item in partial:
            met.append({
                "requirement": item["requirement"],
                "reason": "Promoted by recruiter policy"
            })
        partial = []

    elif hitl_choice.startswith("🔴"):
        for item in partial:
            missing.append({
                "requirement": item["requirement"],
                "reason": "Downgraded by recruiter policy"
            })
        partial = []

    # ---------------- OVERALL MATCH ----------------
    st.markdown("---")
    st.subheader("📊 Overall Match")

    total = len(met) + len(partial) + len(missing)
    match_percentage = int(
        ((len(met) + 0.5 * len(partial)) / total) * 100
    ) if total else 0

    st.metric("Requirement Match Percentage", f"{match_percentage}%")
    st.progress(match_percentage / 100 if total else 0)

    # ---------------- REQUIREMENT EVALUATION ----------------
    st.markdown("---")
    st.subheader("📌 Requirement Evaluation (Recruiter View)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ✅ Met")
        for i in met:
            st.markdown(f"- **{i['requirement']}**")
            st.caption(i["reason"])

    with col2:
        st.markdown("### 🟡 Partially Met")
        for i in partial:
            st.markdown(f"- **{i['requirement']}**")
            st.caption(i["reason"])

    with col3:
        st.markdown("### ❌ Missing")
        if not missing:
            st.success("🎉 No critical gaps found.")
        for i in missing:
            st.markdown(f"- **{i['requirement']}**")
            st.caption(i["reason"])

    # ---------------- CHATBOT ----------------
    st.markdown("---")
    st.subheader("💬 Query Chatbot")

    user_question = st.text_input(
        "Ask a question"
    )

    if st.button("Ask AI") and user_question:
        st.session_state.agent_state["chat_question"] = user_question
        st.session_state.agent_state = execute_action(
            "CHAT_AGENT",
            st.session_state.agent_state,
            API_KEY
        )

    if "chat_answer" in st.session_state.agent_state:
        st.info(st.session_state.agent_state["chat_answer"])

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption(
    "Built with Streamlit • LangGraph • LangChain • Groq LLM • "
    "Explainable Agentic AI • Human-in-the-Loop"
)

