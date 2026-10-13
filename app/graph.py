from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.nodes.account import account_subagent
from app.nodes.complaint import complaint_subagent
from app.nodes.deflector import deflector_subagent
from app.nodes.guardrail import guardrail_node
from app.nodes.intent import classify_intent
from app.nodes.product import product_subagent
from app.state import SupportState

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def route_by_intent(state: SupportState) -> str:
    """Return the name of the next node based on classified intent."""

    intent = state.get("intent", "ot_of_scope")
    return {
        "account_enquiry": "account_subagent",
        "product_question": "product_subagent",
        "complaint": "complaint_subagent",
        "out_of_scope": "deflector_subagent",
    }.get(intent, "deflector_subagent")  # default to deflector if unknown


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph():
    """Build and compile the support agent graph."""

    graph = StateGraph(SupportState)

    # 1. Register every node
    graph.add_node("intent_classifier", classify_intent)
    graph.add_node("account_subagent", account_subagent)
    graph.add_node("product_subagent", product_subagent)
    graph.add_node("complaint_subagent", complaint_subagent)
    graph.add_node("deflector_subagent", deflector_subagent)
    graph.add_node("guardrail", guardrail_node)

    # 2. Entry point
    graph.add_edge(START, "intent_classifier")

    # 3. Conditional routing from the classifier
    graph.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "account_subagent": "account_subagent",
            "product_subagent": "product_subagent",
            "complaint_subagent": "complaint_subagent",
            "deflector_subagent": "deflector_subagent",
        },
    )

    # 4. Every subagent flows to the guardrail
    for node in [
        "account_subagent",
        "product_subagent",
        "complaint_subagent",
        "deflector_subagent",
    ]:
        graph.add_edge(node, "guardrail")

    # 5. Guardrail is the terminal node
    graph.add_edge("guardrail", END)

    # 6. Compile with a checkpointer for conversation memory
    return graph.compile(checkpointer=MemorySaver())


# Singleton for import from FastAPI / Streamlit
compiled_graph = build_graph()
