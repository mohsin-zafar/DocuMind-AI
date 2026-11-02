import re

def chunk_text(text, chunk_words=550, overlap_words=90):
    words = re.findall(r"\S+", text)
    if not words: return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words): break
        start = max(end - overlap_words, start + 1)
    return chunks
