from langchain_core.messages import HumanMessage

from app.nodes.deflector import deflector_subagent

SAMPLES = [
    "What's the weather in Mumbai today?",
    "Can you recommend a good pizza place nearby?",
    "Tell me a joke.",
    "Who won the cricket world cup in 2011?",
    "Write me a poem about databases.",
]


def main():
    for msg in SAMPLES:
        print(f"\n{'=' * 60}\n> {msg}")
        state = {"messages": [HumanMessage(content=msg)], "trace": []}
        result = deflector_subagent(state)
        print(f"  response: {result['draft_response']}")


if __name__ == "__main__":
    main()
