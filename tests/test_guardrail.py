from app.nodes.guardrail import grounding_score, guardrail_node


def run_case(name, state):
    print(f"\n{'=' * 60}\n> {name}")
    result = guardrail_node(state)
    print(f"  final_response: {result['final_response']}")
    print(f"  flags: {result['guardrail_flags']}")


def main():
    # 1. PII redaction — email and phone in the response
    run_case(
        "PII: email + phone in response",
        {
            "intent": "account_enquiry",
            "draft_response": (
                "Your account is active. I've sent a reset link to "
                "aarav@acmecorp.com and you can call us at +91 9876543210."
            ),
            "trace": [],
        },
    )

    # 2. PII redaction — credit card
    run_case(
        "PII: credit card in response",
        {
            "intent": "account_enquiry",
            "draft_response": "The card on file is 4111 1111 1111 1111.",
            "trace": [],
        },
    )

    # 3. Clean product answer — should NOT flag hallucination
    run_case(
        "Product: grounded response",
        {
            "intent": "product_question",
            "draft_response": (
                "To set up Slack, go to Settings > Integrations > Slack, "
                "click Connect, and authorize the TaskFlow app."
            ),
            "retrieved_context": [
                {
                    "source": "integrations.md",
                    "title": "Integrations",
                    "snippet": "Go to Settings > Integrations > Slack. Click Connect "
                    "and authorize the TaskFlow app. Choose a default "
                    "channel for notifications.",
                }
            ],
            "trace": [],
        },
    )

    # 4. Hallucinated product answer — should flag
    run_case(
        "Product: hallucinated response",
        {
            "intent": "product_question",
            "draft_response": (
                "TaskFlow integrates with Figma, Notion, Linear, and Jira "
                "out of the box. Just enable them from the Plugins tab."
            ),
            "retrieved_context": [
                {
                    "source": "integrations.md",
                    "title": "Integrations",
                    "snippet": "TaskFlow integrates with Slack, GitHub, Google "
                    "Calendar, and Zapier.",
                }
            ],
            "trace": [],
        },
    )

    # 5. Out-of-scope but properly deflected
    run_case(
        "Out-of-scope: properly deflected",
        {
            "intent": "out_of_scope",
            "draft_response": (
                "I can only help with TaskFlow topics like billing or integrations."
            ),
            "trace": [],
        },
    )

    # 6. Out-of-scope that failed to deflect (should flag)
    run_case(
        "Out-of-scope: failed to deflect",
        {
            "intent": "out_of_scope",
            "draft_response": "The weather in Mumbai is 32 degrees and humid.",
            "trace": [],
        },
    )

    # 7. Direct grounding score test
    print(f"\n{'=' * 60}\n> Grounding score sanity check")
    ctx = [
        {
            "source": "billing.md",
            "title": "Billing",
            "snippet": "Starter plan costs INR 799 per month.",
        }
    ]
    print(f"  grounded:  {grounding_score('The Starter plan is INR 799 per month.', ctx):.2f}")
    print(f"  off-topic: {grounding_score('TaskFlow supports Figma and Notion.', ctx):.2f}")


if __name__ == "__main__":
    main()
