import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TaskFlow Support Agent",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str}
    st.session_state.chat_history = []

if "last_trace" not in st.session_state:
    st.session_state.last_trace = []

if "last_meta" not in st.session_state:
    # intent, confidence, escalation, flags for the most recent turn
    st.session_state.last_meta = {}


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def call_agent(message: str) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/chat",
            json={"message": message, "thread_id": st.session_state.thread_id},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")
        return {}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_title, col_session = st.columns([3, 1])
with col_title:
    st.title("🤖 TaskFlow Support Agent")
    st.caption("Multi-agent support assistant — LangGraph + OpenAI + Pinecone")
with col_session:
    st.caption(f"**Session:** `{st.session_state.thread_id[:8]}...`")
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.last_trace = []
        st.session_state.last_meta = {}
        st.rerun()


# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------

chat_col, trace_col = st.columns([3, 2])


# ----------------------- Left: Chat -----------------------

with chat_col:
    st.subheader("💬 Chat")

    chat_container = st.container(height=520, border=False)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask about plans, integrations, billing...")
    if user_input:
        # Show the user's message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Agent thinking..."):
            result = call_agent(user_input)

        if result:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": result.get("final_response", "(no response)"),
                }
            )
            st.session_state.last_trace = result.get("trace", [])
            st.session_state.last_meta = {
                "intent": result.get("intent"),
                "confidence": result.get("confidence"),
                "escalation_required": result.get("escalation_required", False),
                "guardrail_flags": result.get("guardrail_flags", []),
            }

        st.rerun()


# ----------------------- Right: Trace panel -----------------------

with trace_col:
    st.subheader("🔍 Agent Trace")

    meta = st.session_state.last_meta

    # Summary badges
    if meta:
        badge_cols = st.columns(2)
        with badge_cols[0]:
            intent = meta.get("intent") or "—"
            conf = meta.get("confidence")
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else "—"
            st.metric("Intent", intent, help=f"Confidence: {conf_str}")

        with badge_cols[1]:
            esc = meta.get("escalation_required", False)
            st.metric("Escalated", "🚨 Yes" if esc else "✅ No")

        flags = meta.get("guardrail_flags", [])
        if flags:
            st.warning("**Guardrail flags:**\n- " + "\n- ".join(flags))
        else:
            st.success("No guardrail flags")

        st.divider()

    # Trace steps
    if not st.session_state.last_trace:
        st.info("Send a message to see the agent's reasoning trace here.")
    else:
        for i, step in enumerate(st.session_state.last_trace, 1):
            node = step.get("node", "unknown")
            status = step.get("status", "ok")
            details = step.get("details", {})

            # Icon + label by node type
            if node == "intent_classifier":
                icon = "🎯"
            elif node == "guardrail":
                icon = "🛡️"
            elif node.endswith("_subagent"):
                icon = "🧩"
            else:
                icon = "⚙️"

            # Status badge
            if status in ("ok", "deflected"):
                status_icon = "✅"
            elif status == "tool_call":
                status_icon = "🔧"
            elif status == "escalated":
                status_icon = "🚨"
            elif status == "flagged":
                status_icon = "⚠️"
            elif status == "max_iterations":
                status_icon = "⛔"
            else:
                status_icon = "•"

            with st.expander(
                f"{icon} **Step {i}: {node}** — {status_icon} `{status}`",
                expanded=(i == len(st.session_state.last_trace)),
            ):
                # Render details depending on node type
                if node == "intent_classifier":
                    st.markdown(f"**Intent:** `{details.get('intent')}`")
                    st.markdown(f"**Confidence:** {details.get('confidence')}")
                    st.markdown(f"**Reasoning:** {details.get('reasoning', '—')}")

                elif node.endswith("_subagent") and status == "tool_call":
                    st.markdown(f"**Tool:** `{details.get('tool')}`")
                    st.markdown("**Args:**")
                    st.json(details.get("args", {}))
                    st.markdown("**Result:**")
                    st.json(details.get("result", {}))

                elif node.endswith("_subagent") and status == "ok":
                    st.markdown(f"**Tool calls made:** {details.get('tool_calls_made', 0)}")
                    st.markdown(f"**Iterations:** {details.get('iterations', 0)}")

                elif node == "guardrail":
                    flags = details.get("flags", [])
                    if flags:
                        st.warning("Flags: " + ", ".join(flags))
                    else:
                        st.success("No flags")
                    st.caption(
                        f"Original: {details.get('original_length')} chars → "
                        f"Redacted: {details.get('redacted_length')} chars"
                    )

                else:
                    st.json(details)
