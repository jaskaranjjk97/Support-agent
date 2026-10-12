from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from app.llm import get_llm
from app.state import SupportState, Intent


class IntentResult(BaseModel):
    intent: Intent = Field(description="The classified intent of the user's message")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0–1")
    reasoning: str = Field(description="One sentence explaining the classification")


SYSTEM_PROMPT = """You are an intent classifier for TaskFlow, a project management SaaS.

Classify the user's message into exactly one of these intents:

- account_enquiry: questions about the user's account, plan, billing, seats,
  renewal, or orders. Examples: "what plan am I on?", "why was I charged twice?"

- product_question: how-to questions about using TaskFlow features.
  Examples: "how do I set up Slack?", "how do I invite team members?"

- complaint: negative feedback, frustration, requests to speak to a human,
  or reports of serious bugs / outages.
  Examples: "this is unacceptable, I want a refund!", "your app keeps crashing"

- out_of_scope: anything not related to TaskFlow.
  Examples: "what's the weather?", "tell me a joke"

Return JSON with intent, confidence (0–1), and a one-sentence reasoning.
If the message is ambiguous, pick the most likely intent and lower the confidence.
"""


def classify_intent(state: SupportState) -> dict:
    """LangGraph node: classify the latest user message into an intent."""
    last_user_msg = state["messages"][-1]

    llm = get_llm().with_structured_output(IntentResult)

    result: IntentResult = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=last_user_msg.content),
    ])

    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "trace": state.get("trace", []) + [{
            "node": "intent_classifier",
            "status": "ok",
            "details": {
                "intent": result.intent,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            },
        }],
    }