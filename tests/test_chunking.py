from documind.chunking import chunk_text

def test_chunking_overlap_and_content():
    text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_text(text, chunk_words=500, overlap_words=80)
    assert len(chunks) >= 3
    assert "word0" in chunks[0]
    assert "word1199" in chunks[-1]
