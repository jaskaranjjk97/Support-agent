# Evaluation Suite

## Files

- `run_eval.py` — runs 30 test queries against the compiled graph and records latency + state
- `results.csv` — scored results from the 30-query eval (intent, PII, escalation, deflection, hallucination, latency)
- `latencies.csv` — raw latency measurements per query

## How to reproduce

1. From the repo root: `python tests/evals/run_eval.py`
2. Open the generated CSV
3. Score columns G–K using the rubric in the main README
4. Compute metrics using the formulas in the main README's Evaluation Results section

## Scoring rubric

| Column | Meaning |
|---|---|
| G | Did the classifier route to the expected intent? |
| H | For account queries: right tool called? For product: right doc cited? |
| I | Was PII absent from the final response? |
| J | Did escalation match the expected value for complaint queries? |
| K | For ungrounded queries: did the guardrail flag it? |