import re

from app.state import SupportState

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Indian mobile: 10 digits starting with 6-9, optional +91 prefix (with or without separator)
PHONE_RE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")
# 13-19 digit card number with optional spaces or dashes
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
# Aadhaar: 4 digits, space, 4 digits, space, 4 digits
AADHAAR_RE = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")

PII_PATTERNS = [
    ("email", EMAIL_RE),
    ("phone", PHONE_RE),
    ("card", CARD_RE),
    ("aadhaar", AADHAAR_RE),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII with tuple of redacted text and types found.

    Returns:
       (redacted_text, list_of_found_types)
    """

    found: list[str] = []
    redacted = text
    for label, pattern in PII_PATTERNS:
        if pattern.search(redacted):
            found.append(label)
            redacted = pattern.sub(f"[{label.upper()}_REDACTED]", redacted)

    return redacted, found


# ---------------------------------------------------------------------------
# Hallucination check
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "should",
    "could",
    "may",
    "might",
    "must",
    "can",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "about",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "from",
    "up",
    "down",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "now",
    "and",
    "or",
    "but",
    "if",
    "while",
    "as",
    "until",
    "because",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "them",
    "their",
    "our",
    "your",
    "my",
    "me",
    "us",
    "also",
    "according",
    "source",
}

TOKEN_RE = re.compile(r"\w+")


def _content_tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 2}


def grounding_score(response: str, retrieved_context: list[dict]) -> float:
    """Fraction of content words in the response that appear in retrieved context.

    Returns a value in [0, 1]. 1.0 means every content word is grounded.
    """

    if not retrieved_context:  # Nothing to compare against
        return 1.0

    context_text = " ".join(
        f"{d.get('source', '')} {d.get('title', '')} {d.get('snippet', '')}"
        for d in retrieved_context
    )

    context_tokens = _content_tokens(context_text)
    response_tokens = _content_tokens(response)

    if not response_tokens:
        return 1.0

    return len(response_tokens & context_tokens) / len(response_tokens)


HALLUCINATION_THRESHOLD = 0.3


# -------------------------------------
# Guardrail Node
# -------------------------------------


def guardrail_node(state: SupportState) -> dict:
    """LangGraph Node: checks PII redaction,hallucinated response,verify scope."""

    draft = state.get("draft_response", "")
    intent = state.get("intent", "")
    retrieved = state.get("retrieved_context", [])

    flags: list[str] = []

    # 1. PII Redaction(this modifies the text)
    redacted, pii_found = redact_pii(draft)
    if pii_found:
        flags.append(f"pii_redacted: {','.join(pii_found)}")

    # 2. Hallucination check(product question only)
    if intent == "product_question" and retrieved:
        score = grounding_score(redacted, retrieved)
        if score < HALLUCINATION_THRESHOLD:
            flags.append(f"hallucination_suspect: {score:.2f}")

    # 3.Scope check (out_of_scope should mention TaskFlow)
    if intent == "out_of_scope" and "taskflow" not in redacted.lower():
        flags.append("scope_check_failed")

    trace = state.get("trace", []) + [
        {
            "node": "guardrail",
            "status": "flagged" if flags else "ok",
            "details": {
                "flags": flags,
                "original_length": len(draft),
                "redacted_length": len(redacted),
            },
        }
    ]

    return {
        "final_response": redacted,
        "guardrail_flags": flags,
        "trace": trace,
    }
