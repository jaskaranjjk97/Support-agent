from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import get_llm
from app.state import SupportState
from app.tools.escalation_tool import create_escalation_ticket

Severity = Literal[
    "low",
    "medium",
    "high",
]


class ComplaintAssessment(BaseModel):
    severity: Severity = Field(description="How serious the complaint is ?")
    escalate: bool = Field(description="Whether the Human should take over or not.")
    reason: str = Field(description="One sentence explanation of assessment.")


ASSESS_SYSTEM = """You are assessing a customer complaint for TaskFlow.

Rate severity and decide whether to escalate to a human:

- low: minor annoyance, user is patient, no urgent impact.
  Example: "the export is slow today"
- medium: real inconvenience or a repeated issue, user is frustrated
  but calm.
  Example: "I've asked about this twice and haven't heard back"
- high: strong emotion, requests for a human, mentions of refunds /
  cancellations / legal action, or critical bugs (data loss, outage).
  Example: "this is unacceptable, I want a refund NOW"

Set escalate=True for medium and high severity.
Return JSON with severity, escalate, and reason.
"""


RESPOND_SYSTEM = """You are the Complaint subagent for TaskFlow support.

Write a short, empathetic response to the user's complaint.

Rules:
- Acknowledge the specific issue (don't give a generic "we're sorry").
- Do NOT make excuses or blame the user.
- Do NOT promise timelines, refunds, or specific outcomes.
- The user message below contains a line starting with either:
    "TICKET CONTEXT: A ticket has been created with ID <id>."
    or
    "TICKET CONTEXT: No ticket was created."
- If the ticket context says a ticket was created, mention the exact ticket ID
  and say a human will follow up.
- If the ticket context says no ticket was created, do NOT mention any ticket
  or ticket ID. Simply acknowledge the issue and offer to help further.
- NEVER invent a ticket ID. Only ever use the exact ID from the ticket context.
- Keep it under 4 sentences.
"""


def complaint_subagent(state: SupportState) -> dict:
    """LangGraph node:assess the complaint severity and escalate optionally as required."""

    last_user_msg = state["messages"][-1]

    # Phase 1 : Assess the severity

    assessor = get_llm().with_structured_output(ComplaintAssessment)
    assessment: ComplaintAssessment = assessor.invoke(
        [
            SystemMessage(content=ASSESS_SYSTEM),
            HumanMessage(content=last_user_msg.content),
        ]
    )

    trace = state.get("trace", [])
    tool_results = []
    escalation_required = assessment.escalate
    ticket_info = None

    # Phase 2 : Create a ticket if escalating
    if assessment.escalate:
        ticket_info = create_escalation_ticket(
            user_message=last_user_msg.content, severity=assessment.severity
        )

        tool_results.append(
            {
                "tool": "create_escalation_ticket",
                "args": {"severity": assessment.severity},
                "result": ticket_info,
            }
        )
        trace.append(
            {
                "node": "complaint_subagent",
                "status": "escalated",
                "details": {
                    "severity": assessment.severity,
                    "reason": assessment.reason,
                    "ticket_id": ticket_info["ticket_id"],
                },
            }
        )

    else:
        trace.append(
            {
                "node": "complaint_subagent",
                "status": "not_escalated",
                "details": {"severity": assessment.severity, "reason": assessment.reason},
            }
        )

    # Phase 3 : Draft response
    if ticket_info:
        ticket_context = (
            f"TICKET CONTEXT: A ticket has been created with ID {ticket_info['ticket_id']}."
        )
    else:
        ticket_context = "TICKET CONTEXT: No ticket was created."

    responder = get_llm(temperature=0.3)
    response = responder.invoke(
        [
            SystemMessage(content=RESPOND_SYSTEM),
            HumanMessage(content=(f"User_complaint:{last_user_msg.content}{ticket_context}")),
        ]
    )

    return {
        "draft_response": response.content,
        "tool_results": tool_results,
        "escalation_required": escalation_required,
        "trace": trace,
    }
