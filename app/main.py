import uuid
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.graph import compiled_graph

app = FastAPI(
    title="TaskFlow support agent",
    description="multi-agent customer support agent build with LangGraph.",
    version="0.1.0",
)

# CORS — permissive for dev; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows specific domains
    allow_methods=["*"],  # Allows all standard HTTP methods(GET<POST,etc.)
    allow_headers=["*"],  # Allows all custom/standard HTTP headers
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Conversation ID; reuse across turns to keep context",
    )


class TraceEntry(BaseModel):
    node: str
    status: str
    details: dict = {}


class ChatResponse(BaseModel):
    thread_id: str
    intent: str | None = None
    confidence: float | None = None
    final_response: str
    escalation_required: bool = False
    guardrail_flags: list[str] = []
    trace: list[TraceEntry] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "service": "taskflow-support-agent",
        "docs": "/docs",
        "health": "/health",
        "chat": "POST /chat",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run one turn of conversation through the agent graph."""
    try:
        result = compiled_graph.invoke(
            {"messages": [HumanMessage(content=request.message)], "trace": []},
            config={"configurable": {"thread_id": request.thread_id}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agent_error: {e}")

    return ChatResponse(
        thread_id=request.thread_id,
        intent=result.get("intent"),
        confidence=result.get("confidence"),
        final_response=result.get("final_response", ""),
        escalation_required=result.get("escalation_required", False),
        guardrail_flags=result.get("guardrail_flags", []),
        trace=[
            TraceEntry(node=t["node"], status=t["status"], details=t.get("details", {}))
            for t in result.get("trace", [])
        ],
    )
