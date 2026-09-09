"""
DAY 1 - Step 2 : Chunking

Goal: turn parsed pages into "chunks"

THIN pages and DENSE pages
"""

import json
from pathlib import Path

IN_PATH = Path(__file__).parent.parent / "data" / "parsed_pages.json"
OUT_PATH = Path(__file__).parent.parent / "data" / "chunks.json"

MIN_CHARS = 120
MAX_CHARS = 1400

def split_dense_page(text: str, max_chars: int)-> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current = ""
    #iterate through paragraphs, keep adding to current as long as it stays under max_chars
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            pieces.append(current.strip())
            current = para
        elif len(para) > max_chars:
            if current:
                pieces.append(current.strip())
                current = ""
            for i in range(0, len(para), max_chars): #divide into blocks of max chars and add in pieces
                pieces.append(para[i : i + max_chars].strip())
        else:
            current = f"{current}\n\n{para}" if current else para

    if current:
        pieces.append(current.strip())

    return pieces if pieces else [text]

def build_chunks(pages: list[dict]) -> list[dict]:
    chunks = []
    pending_merge = None
    chunk_counter = 0

    def flush(text, source, week, slide_label):
        nonlocal chunk_counter #nonlocal is used when a inner function wants to modify a variable from the outer function

        for piece in split_dense_page(text, MAX_CHARS):
            chunk_counter +=1
            chunks.append(
                {
                    "chunk_id" : f"c{chunk_counter:04d}",
                    "source" : source,
                    "week": week,
                    "slide_label": slide_label,
                    "text" : piece
                }
            )
    for page in pages:
        text, source, week, slide = page["text"], page["source"], page["week"], page["slide"]

        if pending_merge is not None:
            if pending_merge["source"] == source: #if same pdf then merfe
                text = pending_merge["text"] +  "\n\n" + text
                slide_label = f"slides {pending_merge['slide']}-{slide}"
            else:
                flush(
                    pending_merge["text"],
                    pending_merge["source"],
                    pending_merge["week"],
                    f"slide {pending_merge['slide']}",
                )
                slide_label = f"slide {slide}"
            pending_merge = None
        else:
            slide_label = f"slide {slide}"

        if len(text) < MIN_CHARS: # if its below MIN_CHARS, then save it but not flush
            pending_merge = {"text": text, "source": source, "week": week, "slide": slide}
            continue

        flush(text, source, week, slide_label)

    if pending_merge is not None: #add the last page 
        flush(
            pending_merge["text"],
            pending_merge["source"],
            pending_merge["week"],
            f"slide {pending_merge['slide']}",
        )

    return chunks

def main():
    pages = json.loads(IN_PATH.read_text())
    chunks = build_chunks(pages)
    OUT_PATH.write_text(json.dumps(chunks, indent = 2, ensure_ascii= False))
    print(f"Built {len(chunks)} from {len(pages)} slides")
    print(f"Saved to {OUT_PATH}")

    lengths = [len(c["text"]) for c in chunks]
    print(f"Chunk length: min={min(lengths)}, max = {max(lengths)} avg = {sum(lengths)//len(lengths)}")

if __name__ =="__main__":
    main()
        


        