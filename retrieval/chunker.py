from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import pymupdf as fitz

MIN_WORDS = 40
MAX_WORDS = 250

@dataclass
class Chunk:
    chunk_id: str
    source_pdf: str
    slide_label: str
    text: str
    word_count: int = field(init=False) #by keeping init as false, you keep it as a field of the class, but dont include
    #it as a parameter in the automatically generated __init__() function

    def __post_init__(self):
        self.word_count = len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_pdf": self.source_pdf,
            "slide_label": self.slide_label,
            "text": self.text,
            "word_count": self.word_count
        }

def _word_count(text: str | None) -> int:
    """Count words in text, return 0 if None"""
    if text is None:
        return 0
    return len(text.split())

def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text) #reduces consecutive newlines to 2 lines
    return text.strip()

def _page_blocks(page: "fitz.Page") -> list[str]:
    """
    Return this page's text blocks (title, bullet list, caption, etc.)
    as separate strings, in reading order (top-to-bottom, left-to-right).
    Using blocks instead of raw get_text() gives us real paragraph-like
    units to split on, since slide layouts are naturally block-structured.
    """
    raw_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type), helps return text blocks
    # keep only text blocks (type 0), sort by vertical then horizontal position, these are the coordinates that tell where block appears on the page
    text_blocks = [b for b in raw_blocks if b[6] == 0 and b[4].strip()] #0 is for text blocks, 1 is for images, we only want text blocks
    text_blocks.sort(key=lambda b: (round(b[1], 1), b[0])) #sort by vertical then horizontal position
    return [_clean(b[4]) for b in text_blocks if _clean(b[4])]
 
 
def _load_pages(pdf_path: Path) -> list[dict]:
    """Returns [{page_num, text, blocks}] for every non-empty page."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        blocks = _page_blocks(page)
        full_text = _clean("\n\n".join(blocks))
        if full_text:
            pages.append({"page_num": i, "text": full_text, "blocks": blocks})
    doc.close()
    return pages
 
 
def _split_dense(blocks: list[str], slide_label: str, max_words: int) -> list[tuple[str, str]]:
    """
    Greedily pack a slide's text blocks into sub-chunks that each stay
    under max_words. Returns [(sub_label, text), ...].
    If the slide doesn't need splitting, returns a single entry.
    """
    full_text = "\n\n".join(blocks)
    if _word_count(full_text) <= max_words:
        return [(slide_label, full_text)]
 
    sub_chunks: list[str] = []
    current: list[str] = []
    current_wc = 0
 
    for block in blocks:
        bwc = _word_count(block)
        if current and current_wc + bwc > max_words:
            sub_chunks.append("\n\n".join(current))
            current = [block]
            current_wc = bwc
        else:
            current.append(block)
            current_wc += bwc
    if current:
        sub_chunks.append("\n\n".join(current))
 
    # label sub-chunks 20a, 20b, 20c ...
    labeled = []
    for idx, sc in enumerate(sub_chunks):
        suffix = chr(ord("a") + idx)
        labeled.append((f"{slide_label}{suffix}", sc))
    return labeled
 
 
def chunk_pdf(pdf_path: str | Path, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> list[Chunk]:
    """
    Main entry point. Parses `pdf_path` and returns a list of Chunk objects
    ready for embedding/indexing.
    """
    pdf_path = Path(pdf_path)
    source_name = pdf_path.stem  # "march" from "march.pdf"
    pages = _load_pages(pdf_path)
 
    raw_chunks: list[tuple[str, str]] = []  # (slide_label, text) before dense-splitting
    buffer_blocks: list[str] = []
    buffer_start: int | None = None
    buffer_end: int | None = None
 
    for page in pages:
        wc = _word_count(page["text"])
 
        if wc < min_words:
            # thin slide: fold its blocks into the buffer, merge forward
            if buffer_start is None:
                buffer_start = page["page_num"]
            buffer_end = page["page_num"]
            buffer_blocks.extend(page["blocks"])
            continue
 
        # this page has enough content to anchor a chunk
        if buffer_blocks:
            label = (
                f"{buffer_start}-{page['page_num']}"
                if buffer_start != page["page_num"]
                else str(page["page_num"])
            )
            merged_blocks = buffer_blocks + page["blocks"]
            raw_chunks.append((label, "\n\n".join(merged_blocks)))
            buffer_blocks, buffer_start, buffer_end = [], None, None
        else:
            raw_chunks.append((str(page["page_num"]), page["text"]))
 
    # trailing thin pages with nothing dense after them: merge into the
    # previous chunk if one exists, otherwise emit them as their own chunk
    if buffer_blocks:
        label = f"{buffer_start}-{buffer_end}" if buffer_start != buffer_end else str(buffer_start)
        if raw_chunks:
            prev_label, prev_text = raw_chunks[-1]
            raw_chunks[-1] = (f"{prev_label}, {label}", prev_text + "\n\n" + "\n\n".join(buffer_blocks))
        else:
            raw_chunks.append((label, "\n\n".join(buffer_blocks)))
 
    # now split any chunk that ended up too dense, using its blocks again
    final: list[Chunk] = []
    counter = 0
    for label, text in raw_chunks:
        blocks = [b for b in text.split("\n\n") if b.strip()]
        for sub_label, sub_text in _split_dense(blocks, label, max_words):
            final.append(
                Chunk(
                    chunk_id=f"{source_name}_c{counter:03d}",
                    source_pdf=f"{source_name}.pdf",
                    slide_label=sub_label,
                    text=sub_text,
                )
            )
            counter += 1
 
    return final
 
 
if __name__ == "__main__":
    import sys
 
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/march.pdf"
    chunks = chunk_pdf(path)
    print(f"Parsed {path} -> {len(chunks)} chunks\n")
    for c in chunks:
        print(f"[{c.chunk_id}] slide {c.slide_label} ({c.word_count} words)")

