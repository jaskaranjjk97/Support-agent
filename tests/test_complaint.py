from langchain_core.messages import HumanMessage

from app.nodes.complaint import complaint_subagent

SAMPLES = [
    "The export feature is a bit slow today, but not urgent.",
    "I've asked about this issue twice and still haven't heard back. Can someone look into it?",
    "This is UNACCEPTABLE. I've been charged twice and no one responds. "
    "I want a refund and I want to speak to a human NOW.",
    "Your app has been down for 2 hours and we're losing customers.",
]


def main():
    for msg in SAMPLES:
        print(f"\n{'=' * 60}\n> {msg}")
        state = {"messages": [HumanMessage(content=msg)], "trace": []}
        result = complaint_subagent(state)
        print(f"  escalation_required: {result['escalation_required']}")
        print(f"  response: {result['draft_response']}")
        for tr in result.get("tool_results", []):
            print(f"  ticket: {tr['result']}")


if __name__ == "__main__":
    main()
