from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Intent = Literal[
    "account_enquiry",
    "product_question",
    "complaint",
    "out_of_scope",
]


class SupportState(TypedDict, total=False):
    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # intent of the user query(Routing)
    intent: Intent
    confidence: float

    # Subagent outputs
    retrieved_context: list[dict]  # docs from Pinecone
    tool_results: list[dict]  # tool call outputs
    draft_response: str  # subagent draft before guardrails

    # Guardrails + final
    guardrail_flags: list[str]
    escalation_required: bool
    final_response: str

    # UI trace (every node appends a step)
    trace: list[dict]
