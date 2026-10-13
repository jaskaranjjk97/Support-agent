import uuid

from langchain_core.messages import HumanMessage

from app.graph import compiled_graph

SAMPLES = [
    "What plan am I on? My user ID is U1001.",
    "How do I set up the Slack integration?",
    "This is UNACCEPTABLE, I want a refund and to speak to a human!",
    "What's the weather in Mumbai?",
]


def run(msg: str, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    result = compiled_graph.invoke(
        {"messages": [HumanMessage(content=msg)], "trace": []},
        config=config,
    )
    print(f"\n{'=' * 70}")
    print(f"> {msg}")
    print(f"  intent: {result.get('intent')} (confidence={result.get('confidence'):.2f})")
    print(f"  escalated: {result.get('escalation_required', False)}")
    print(f"  flags: {result.get('guardrail_flags', [])}")
    print(f"  final_response: {result.get('final_response')}")
    print(f"  trace: {[(t['node'], t.get('status')) for t in result.get('trace', [])]}")


def main():
    for msg in SAMPLES:
        run(msg, thread_id=str(uuid.uuid4()))


if __name__ == "__main__":
    main()
