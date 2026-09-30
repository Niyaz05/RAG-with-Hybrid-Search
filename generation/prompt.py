"""
DAY 4 — PART A: Citation-enforced generation
==============================================
Pipeline: retrieved chunks -> prompt -> Gemma 4 -> answer -> VERIFY citations.

WHY VERIFY? Prompting the model to "cite your sources" reduces made-up
citations but cannot guarantee it. A fluent model can still write
[april_2026.pdf, slide 99] for a slide that was never retrieved. So we
parse every citation out of the answer and check it against the chunks we
actually gave the model. Prompting is the request; verification is the check.

DEPENDENCY INJECTION AGAIN: generate_answer() takes a chat_fn parameter
(like embed_fn on Day 2, score_fn on Day 3). Tests pass a fake chat_fn, so
no Ollama is needed to prove the logic.
"""

import re
from typing import Callable

GEN_MODEL = "gemma4"
REFUSAL_TEXT = "I don't have this in the course slides."

SYSTEM_PROMPT = f"""You are a study assistant for a university course on Advanced Computer Vision and Video Analytics. Answer the student's question using ONLY the course slide excerpts provided in the context.

Rules:
1. Use only information found in the excerpts. Do not use outside knowledge.
2. After every claim, cite its source in exactly this format: [source_file.pdf, slide N]. Copy the tag from the excerpt header exactly. Put one source per bracket.
3. If the excerpts do not contain the answer, reply with exactly: {REFUSAL_TEXT}
4. Be concise and explain like a tutor."""

# Matches e.g. "[april_2026.pdf, slide 7]" or "[jan_2026.pdf, slides 1-2]"
CITATION_PATTERN = re.compile(
    r"\[([\w\-. ]+\.pdf),\s*(slides?\s+\d+(?:\s*-\s*\d+)?)\]", re.IGNORECASE
)

ChatFn = Callable[[list[dict]], str]


# ---------------------------------------------------------------------------
# PURE HELPERS
# ---------------------------------------------------------------------------

def slide_numbers(slide_label: str) -> set[int]:
    """
    'slide 7' -> {7};  'slides 1-2' -> {1, 2}.
    Merged chunks from chunk.py carry ranges, so comparing slides means
    comparing SETS of numbers, not strings.
    """
    nums = [int(n) for n in re.findall(r"\d+", slide_label)]
    if len(nums) == 2 and "-" in slide_label:
        return set(range(nums[0], nums[1] + 1))
    return set(nums)


def format_context(chunks: list[dict]) -> str:
    """
    Render chunks as excerpts, each with a header tag the model can copy
    verbatim into its citations.
    """
    blocks = [f"[{c['source']}, {c['slide_label']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(blocks)


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    user_content = f"Course slide excerpts:\n\n{format_context(chunks)}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def extract_citations(answer: str) -> list[tuple[str, str]]:
    """Pull every (source, slide_label) citation out of the answer text."""
    return [(src.lower(), label.lower()) for src, label in CITATION_PATTERN.findall(answer)]


def verify_citations(answer: str, chunks: list[dict]) -> dict:
    """
    A citation is VALID if some retrieved chunk has the same source file AND
    an overlapping slide number. Overlap (not exact string match) means
    citing 'slide 2' is fine when the retrieved chunk was 'slides 1-2'.
    """
    valid, invalid = [], []
    for source, label in extract_citations(answer):
        cited = slide_numbers(label)
        supported = any(
            c["source"].lower() == source and cited & slide_numbers(c["slide_label"])
            for c in chunks
        )
        (valid if supported else invalid).append((source, label))
    return {"valid": valid, "invalid": invalid}


def is_refusal(answer: str) -> bool:
    return REFUSAL_TEXT.lower() in answer.lower()


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def generate_answer(query: str, chunks: list[dict], chat_fn: ChatFn) -> dict:
    """
    Returns {"answer", "refused", "citations_valid", "citations_invalid"}.

    If retrieval returned nothing, we refuse WITHOUT calling the model at
    all: it's cheaper, and it's impossible to hallucinate from no context.
    """
    if not chunks:
        return {"answer": REFUSAL_TEXT, "refused": True,
                "citations_valid": [], "citations_invalid": []}

    answer = chat_fn(build_messages(query, chunks))
    """
    generate_answer()
       │
       ↓
     chat_fn
     /    \
    /      \
Ollama     Fake
Gemma      test function
    """
    checked = verify_citations(answer, chunks)
    return {
        "answer": answer,
        "refused": is_refusal(answer),
        "citations_valid": checked["valid"],
        "citations_invalid": checked["invalid"],
    }


# ---------------------------------------------------------------------------
# THE REAL chat_fn — the only code here that talks to Ollama
# ---------------------------------------------------------------------------

def ollama_chat(messages: list[dict]) -> str:
    import ollama  # lazy import so tests don't need Ollama running
    response = ollama.chat(model=GEN_MODEL, messages=messages)
    return response["message"]["content"]