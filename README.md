# TaskFlow Support Agent

A multi-agent customer support assistant built with LangGraph.
Routes user messages to specialist subagents (account, product, complaint,
deflector), runs every response through a guardrail layer, and escalates
high-severity complaints to a human.

## Architecture
[diagram placeholder - add later]

## Setup
1. `python -m venv .venv && source .venv/bin/activate or windows: .venv\Scripts\Activate.ps1`
2. `if the virtual environment doesn't activate use this:Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`
3. `pip install -r requirements.txt`
4. `cp .env.example .env` and add your OpenAI key
5. `python -m tests.test_setup`

## Run
- Backend: `uvicorn app.main:app --reload`
- UI: `streamlit run ui/streamlit_app.py`

## Stack
LangGraph, LangChain, OpenAI, ChromaDB, FastAPI, Streamlit, Docker