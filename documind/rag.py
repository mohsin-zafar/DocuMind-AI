import json
from pathlib import Path
import numpy as np
from .chunking import chunk_text

class RAGEngine:
    def __init__(self, model_name, data_dir):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / "faiss.index"
        self.meta_path = self.data_dir / "chunks.json"
        self._embedder = None
        self._index = None
        self._metadata = []
        self._load_index()

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.model_name, device="cpu")
            except Exception as e:
                raise RuntimeError(f"Could not load embedding model '{self.model_name}'. Confirm sentence-transformers is installed and the model is downloaded. Detail: {e}")
        return self._embedder

    @property
    def embedding_dimension(self):
        return self.embedder.get_sentence_embedding_dimension()

    def _load_index(self):
        if self.index_path.exists() and self.meta_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(self.index_path))
                self._metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                self._index, self._metadata = None, []

    def _save(self):
        import faiss
        tmp_index = self.data_dir / "faiss.index.tmp"
        tmp_meta = self.data_dir / "chunks.json.tmp"
        faiss.write_index(self._index, str(tmp_index))
        tmp_meta.write_text(json.dumps(self._metadata, ensure_ascii=False), encoding="utf-8")
        tmp_index.replace(self.index_path)
        tmp_meta.replace(self.meta_path)

    def add_documents(self, documents):
        import faiss
        texts, metadata = [], []
        for doc in documents:
            pieces = chunk_text(doc["text"])
            for i, piece in enumerate(pieces):
                texts.append(piece)
                metadata.append({"source": doc["source"], "location": f"Chunk {i+1}", "text": piece})
        if not texts: return 0
        vectors = self.embedder.encode(texts, batch_size=16, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype="float32")
        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        if self._index.d != vectors.shape[1]:
            raise RuntimeError("Existing FAISS index dimension does not match the embedding model. Back up and remove data/faiss.index and data/chunks.json, then re-index.")
        self._index.add(vectors)
        self._metadata.extend(metadata)
        self._save()
        return len(texts)

    def stats(self):
        return {"chunks": len(self._metadata), "documents": len(set(x["source"] for x in self._metadata))}

    def retrieve(self, question, top_k=5):
        if self._index is None or not self._metadata: return []
        vector = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, ids = self._index.search(vector, min(top_k, len(self._metadata)))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                item = dict(self._metadata[int(idx)])
                item["score"] = float(score)
                results.append(item)
        return results

    def answer(self, question, ollama):
        sources = self.retrieve(question)
        if not sources:
            return "I could not find relevant indexed content. Upload and index documents first, then ask again.", []
        context = "\n\n".join(f"[Source {i+1}: {s['source']} / {s['location']}]\n{s['text']}" for i,s in enumerate(sources))
        system = ("You are a careful document question-answering assistant. Answer only from the supplied context. "
                  "If the context does not contain the answer, say so clearly. Never invent facts, citations, dates, or numbers. "
                  "Cite source names and locations inline. Distinguish explicit facts from reasonable inferences.")
        prompt = f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nGive a clear, structured answer with source citations."
        answer = ollama.generate(prompt, system=system, temperature=0.1)
        return answer, sources
