from retrieval.chunker import chunk_pdf


pdf_path = "data/raw/April_2026.pdf"

# Run the chunking function
chunks = chunk_pdf(pdf_path)

print(f"\nPDF: {pdf_path}")
print(f"Total chunks: {len(chunks)}")
print("=" * 80)

# Print every chunk
for chunk in chunks:
    print(f"\nChunk ID     : {chunk.chunk_id}")
    print(f"Source PDF   : {chunk.source_pdf}")
    print(f"Slide        : {chunk.slide_label}")
    print(f"Word count   : {chunk.word_count}")
    print("-" * 80)
    print(chunk.text)
    print("=" * 80)