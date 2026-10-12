from langchain_core.messages import HumanMessage

from app.nodes.product import product_subagent

SAMPLES = [
    "How do I set up the Slack integration?",
    "What plans does TaskFlow offer and how much do they cost?",
    "How do I invite team members to my workspace?",
    "What should I do if my tasks aren't syncing?",
    "Do you support integration with Figma?",
]


def main():
    for msg in SAMPLES:
        print(f"\n{'=' * 60}\n> {msg}")
        state = {"messages": [HumanMessage(content=msg)], "trace": []}
        result = product_subagent(state)
        print(f"  answer: {result['draft_response']}")
        print(
            f"  sources: {[d['source'] for d in result.get('retrieved_context', [])]}"
        )


if __name__ == "__main__":
    main()
