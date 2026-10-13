from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.state import SupportState

SYSTEM_PROMPT = """You are the Deflector subagent for TaskFlow support.

The user has asked something outside TaskFlow's scope. Your job is to
politely decline and redirect.

Rules:
- Do NOT answer the out-of-scope question, even partially.
- Do NOT pretend you don't understand — you do, you're just declining.
- Do NOT lecture about your limitations or mention "AI language model".
- In 2-3 sentences: acknowledge briefly, say you can only help with
  TaskFlow topics, and invite them to ask about TaskFlow.
- Mention 2-3 example TaskFlow topics to make the redirect concrete
  (e.g., "billing, integrations, or account settings").
- Never be robotic. Sound like a helpful human, not a script.
"""


def deflector_subagent(state: SupportState) -> dict:
    """LangGraph node: handle out_of_scope messages with a polite redirect."""

    last_user_msg = state["messages"][-1]
    llm = get_llm(temperature=0.3)
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=last_user_msg.content),
        ]
    )

    trace = state.get("trace", [])
    trace.append(
        {
            "node": "deflector_subagent",
            "status": "deflected",
            "details": {
                "resaon": "oot_of_scope intent",
            },
        }
    )

    return {
        "draft_response": response.content,
        "trace": trace,
    }
