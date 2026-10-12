from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.rag import retrieve
from app.state import SupportState

SYSTEM_PROMPT = """You are the Product subagent for TaskFlow support.

You answer how-to and feature questions about TaskFlow using ONLY the
retrieved help documentation provided below.

Rules:
- Base every statement on the retrieved context. Do not use outside knowledge.
- If the retrieved context does not answer the question, say:
  "I don't have that information in the help docs — I can escalate this
  to our team if you'd like."
- Cite the source document by name when relevant
  (e.g., "According to integrations.md, ...").
- Keep responses clear and step-by-step when the docs describe a procedure.
- Never invent features, pricing, or integrations that aren't in the docs.
"""


def product_subagent(state: SupportState) -> dict:
    """LangGraph node: Answer questions related to product query via RAG with help docs."""

    last_user_msg = state["messages"][-1]

    # 1. Retrieve relevant help docs
    docs = retrieve(last_user_msg.content, k=4)

    # 2. Format retrieved context for the prompt

    context_blocks = []

    for d in docs:
        source = d.metadata.get("source", "unknown")
        context_blocks.append(f"Source: {source}\n{d.page_content} ")

    context_text = "\n\n-----\n\n".join(context_blocks)

    # Generate a grounded answer
    llm = get_llm(temperature=0.2)

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Help documentation:\n\n{context_text}\n\n"
                    f"User question: {last_user_msg.content}"
                )
            ),
        ]
    )

    # 4. Track retrieved context for the guardrail + UI

    retrieved_context = [
        {
            "source": d.metadata.get("source", "unknown"),
            "title": d.metadata.get("title", ""),
            "snippet": d.page_content[:200] + "...",
        }
        for d in docs
    ]

    trace = state.get("trace", [])

    trace.append(
        {
            "node": "product_subagent",
            "status": "ok",
            "details": {
                "docs_retrieved": len(docs),
                "sources": [d["source"] for d in retrieved_context],
            },
        }
    )

    return {
        "draft_response": response.content,
        "retrieved_context": retrieved_context,
        "trace": trace,
    }
