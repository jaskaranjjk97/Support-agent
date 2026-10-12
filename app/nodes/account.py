from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from app.llm import get_llm
from app.state import SupportState
from app.tools.account_tool import (
    get_account_details,
    get_order_status,
    reset_password,
)
from app.tools.registry import ACCOUNT_TOOLS

TOOL_MAP = {
    "get_account_details": get_account_details,
    "get_order_status": get_order_status,
    "reset_password": reset_password,
}


SYSTEM_PROMPT = """You are the Account subagent for TaskFlow support.

You handle questions about a user's plan, billing, orders, seats, renewals,
and password resets.

Available tools and their required arguments:
- get_account_details(user_id): needs a user_id like "U1001"
- get_order_status(order_id): needs an order_id like "O5002"
- reset_password(user_id): needs a user_id like "U1001"

Rules:
- Match the tool to what the user actually provides. If they give an
  order ID, call get_order_status. If they give a user ID, call
  get_account_details or reset_password as appropriate.
- Never invent account, order, or user data. Only report what tools return.
- If a tool needs an argument the user hasn't provided, ask for that
  specific argument. Do not ask for a user ID if the tool doesn't need one.
- If a tool returns not_found, apologize and ask for the correct ID.
- Keep responses short, friendly, and factual.
"""

MAX_ITERATIONS = 5


def account_subagent(state: SupportState) -> dict:
    """LangGraph node: it handles account related enquiries via tool calling."""

    # Binding the langchain tool objects we created in registry with StructureTool.from_function
    llm = get_llm().bind_tools(ACCOUNT_TOOLS)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]

    tool_results = []
    trace = state.get("trace", [])

    for step in range(MAX_ITERATIONS):
        ai_msg: AIMessage = llm.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            # LLM decided its done - final text
            trace.append(
                {
                    "node": "account_subagent",
                    "status": "ok",
                    "details": {
                        "tool_calls_made": len(tool_results),
                        "iterations": step + 1,
                    },
                }
            )

            return {
                "draft_response": ai_msg.content,
                "tool_results": tool_results,
                "trace": trace,
            }

        # Execute each tool call
        for tc in ai_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            fn = TOOL_MAP.get(tool_name)

            if fn is None:
                result = {"error": f"unknown_tool:{tool_name}"}
            else:
                result = fn(**tool_args)

            tool_results.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                }
            )

            trace.append(
                {
                    "node": "account_subagent",
                    "status": "tool_call",
                    "details": {"tool": tool_name, "args": tool_args, "result": result},
                }
            )

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # Safety valve — max iterations reached
    trace.append(
        {
            "node": "account_subagent",
            "status": "max_iterations",
            "details": {"iterations": MAX_ITERATIONS},
        }
    )

    return {
        "draft_response": "I'm sorry, I couldn't complete that request. "
        "Could you rephrase or provide more details?",
        "tool_results": tool_results,
        "trace": trace,
    }
