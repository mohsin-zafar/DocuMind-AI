from io import BytesIO, StringIO
from pathlib import Path
import csv

def _decode_text(data):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try: return data.decode(enc)
        except UnicodeDecodeError: pass
    return data.decode("utf-8", errors="replace")

def extract_uploaded_files(uploaded_files):
    results = []
    for f in uploaded_files:
        name = getattr(f, "name", "uploaded_document")
        ext = Path(name).suffix.lower()
        raw = f.getvalue()
        text = ""
        locations = []
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(BytesIO(raw))
                for i, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text += f"\n\n[Page {i}]\n{page_text}"
            except Exception as e: raise RuntimeError(f"Could not read PDF '{name}': {e}")
        elif ext == ".docx":
            try:
                from docx import Document
                doc = Document(BytesIO(raw))
                text = "\n".join(p.text for p in doc.paragraphs)
                for ti, table in enumerate(doc.tables, 1):
                    text += f"\n\n[Table {ti}]\n" + "\n".join(" | ".join(c.text for c in row.cells) for row in table.rows)
            except Exception as e: raise RuntimeError(f"Could not read DOCX '{name}': {e}")
        elif ext in (".txt", ".md"):
            text = _decode_text(raw)
        elif ext == ".csv":
            text = _decode_text(raw)
        elif ext == ".xlsx":
            try:
                import pandas as pd
                book = pd.ExcelFile(BytesIO(raw))
                parts = []
                for sheet in book.sheet_names:
                    df = pd.read_excel(book, sheet_name=sheet).fillna("")
                    parts.append(f"[Sheet: {sheet}]\n{df.to_csv(index=False)}")
                text = "\n\n".join(parts)
            except Exception as e: raise RuntimeError(f"Could not read XLSX '{name}': {e}")
        else:
            continue
        if text.strip():
            results.append({"source": name, "text": text.strip()})
        elif ext == ".pdf":
            results.append({"source": name, "text": "", "warning": "No extractable text. This may be a scanned PDF; OCR is not included in this build."})
    return [d for d in results if d["text"].strip()]
