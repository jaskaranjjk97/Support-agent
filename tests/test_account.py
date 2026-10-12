from langchain_core.messages import HumanMessage

from app.nodes.account import account_subagent

SAMPLES = [
    "What plan am I on? My user ID is U1001.",
    "Why was order O5002 marked as failed?",
    "Please reset my password. My user ID is U1003.",
    "What's my account status? My email is priya@brightlabs.io",  # no user_id
    "Show me the details for user U9999.",  # unknown user
]


def main():
    for msg in SAMPLES:
        print(f"\n{'=' * 60}\n> {msg}")
        state = {"messages": [HumanMessage(content=msg)], "trace": []}
        result = account_subagent(state)
        print(f"  draft_response: {result['draft_response']}")
        print("  tool_results:")
        for tr in result.get("tool_results", []):
            print(f"    - {tr['tool']}({tr['args']}) → {tr['result']}")


if __name__ == "__main__":
    main()
