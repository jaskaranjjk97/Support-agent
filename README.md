# TaskFlow Support Agent

A multi-agent customer support assistant for a fictional SaaS product (TaskFlow),
built with LangGraph. The system classifies incoming user messages, routes them
to specialist subagents, runs every response through a guardrail layer, and
escalates high-severity complaints to a human.

## Status

- [x] Project skeleton + config
- [x] `SupportState` + LangGraph state design
- [x] Intent classification node (structured output)
- [ ] Account subagent (tool calling)
- [ ] Product subagent (RAG over help docs via Pinecone)
- [ ] Complaint subagent (human-in-the-loop)
- [ ] Deflector subagent (safe refusal)
- [ ] Guardrails (PII, hallucination, scope)
- [ ] FastAPI backend
- [ ] Streamlit UI with routing trace
- [ ] Docker + deployment

## Architecture



Full diagram: `docs/architecture.md` (coming soon).

## Stack

| Layer | Tech |
|-------|------|
| Orchestration | LangGraph |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | Pinecone (serverless) |
| Backend | FastAPI |
| UI | Streamlit |
| Deployment | Docker |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # then add your OpenAI + Pinecone keys
python -m tests.test_setup