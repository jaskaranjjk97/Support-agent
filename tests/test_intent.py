from langchain_core.messages import HumanMessage

from app.nodes.intent import classify_intent

SAMPLES =[
    "What plan am I on? My email is aarav@acmecorp.com",
    "How do I set up the Slack integration?",
    "This is ridiculous, I've been charged twice and no one responds! I want a human.",
    "What's the weather in Mumbai today?",
    "Why was my payment for order O5002 marked as failed?",
    "Can you recommend a good pizza place nearby?",
]

def main():
    for msg in SAMPLES:
        state = {"messages":[HumanMessage(content=msg)],"trace":[]}
        result = classify_intent(state)
        print(f"\n> {msg}")
        print(f"  intent     = {result['intent']}")
        print(f"  confidence = {result['confidence']}")
        print(f"  reasoning  = {result['trace'][-1]['details']['reasoning']}")


if __name__ == "__main__":
    main()