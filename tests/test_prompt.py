"""
TESTS for generation/prompt.py
================================
Run with:  pytest tests/test_prompt.py -v
All tests use a FAKE chat_fn, so Gemma/Ollama is never called.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "generation"))

# pyrefly: ignore [missing-import]
from prompt import (
    slide_numbers, format_context, build_messages, extract_citations,
    verify_citations, is_refusal, generate_answer, REFUSAL_TEXT,
)


def _chunks():
    return [
        {"chunk_id": "c1", "source": "april_2026.pdf", "slide_label": "slide 7",
         "text": "fx and fy differ because pixels may be rectangular."},
        {"chunk_id": "c2", "source": "jan_2026.pdf", "slide_label": "slides 1-2",
         "text": "Course intro and overview."},
    ]


# --- slide_numbers -----------------------------------------------------------

def test_slide_numbers_single_and_range():
    assert slide_numbers("slide 7") == {7}
    assert slide_numbers("slides 1-3") == {1, 2, 3}


# --- prompt building ---------------------------------------------------------

def test_format_context_includes_copyable_source_tags():
    ctx = format_context(_chunks())
    assert "[april_2026.pdf, slide 7]" in ctx
    assert "[jan_2026.pdf, slides 1-2]" in ctx


def test_build_messages_has_system_and_user_roles_with_question():
    msgs = build_messages("Why fx and fy?", _chunks())
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Why fx and fy?" in msgs[1]["content"]


# --- citation extraction / verification -------------------------------------

def test_extract_citations_finds_multiple_and_ignores_other_brackets():
    answer = "Pixels can be rectangular [april_2026.pdf, slide 7]. See also [1] and [jan_2026.pdf, slides 1-2]."
    assert extract_citations(answer) == [("april_2026.pdf", "slide 7"), ("jan_2026.pdf", "slides 1-2")]


def test_verify_accepts_real_citation():
    result = verify_citations("Yes [april_2026.pdf, slide 7].", _chunks())
    assert result["valid"] == [("april_2026.pdf", "slide 7")]
    assert result["invalid"] == []


def test_verify_flags_fabricated_slide_number():
    result = verify_citations("Claim [april_2026.pdf, slide 99].", _chunks())
    assert result["invalid"] == [("april_2026.pdf", "slide 99")]


def test_verify_flags_real_slide_from_wrong_pdf():
    result = verify_citations("Claim [feb_2026.pdf, slide 7].", _chunks())
    assert result["invalid"] == [("feb_2026.pdf", "slide 7")]


def test_verify_accepts_single_slide_inside_merged_range():
    result = verify_citations("Intro [jan_2026.pdf, slide 2].", _chunks())
    assert result["valid"] == [("jan_2026.pdf", "slide 2")]


# --- generate_answer ---------------------------------------------------------

def test_generate_answer_reports_valid_and_invalid_citations():
    fake = lambda msgs: "Rectangular pixels [april_2026.pdf, slide 7] and magic [april_2026.pdf, slide 99]."
    result = generate_answer("Why fx and fy?", _chunks(), chat_fn=fake)
    assert result["citations_valid"] == [("april_2026.pdf", "slide 7")]
    assert result["citations_invalid"] == [("april_2026.pdf", "slide 99")]
    assert result["refused"] is False


def test_no_chunks_refuses_without_calling_the_model():
    def must_not_be_called(msgs):
        raise AssertionError("chat_fn should not be called when there is no context")

    result = generate_answer("Capital of France?", [], chat_fn=must_not_be_called)
    assert result["refused"] is True
    assert result["answer"] == REFUSAL_TEXT


def test_model_refusal_is_detected():
    result = generate_answer("Off-topic?", _chunks(), chat_fn=lambda m: REFUSAL_TEXT)
    assert result["refused"] is True
    assert is_refusal(REFUSAL_TEXT)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])