import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()
from documind.config import settings
from documind.ollama_client import OllamaService
from documind.documents import extract_uploaded_files
from documind.rag import RAGEngine
from documind.compare import compare_documents
from documind.ats import evaluate_resumes

st.set_page_config(page_title="DOCUMIND AI", page_icon="📚", layout="wide")
st.title("📚 DOCUMIND AI")
st.caption("Private, local-first document intelligence powered by your existing Ollama installation.")

@st.cache_resource(show_spinner=False)
def get_ollama():
    return OllamaService(
        host=settings.ollama_host,
        model=(
            settings.groq_model
            if settings.ai_provider == "groq"
            else settings.ollama_model
        ),
        timeout=settings.request_timeout,
        provider=settings.ai_provider,
        api_key=settings.groq_api_key,
    )

@st.cache_resource(show_spinner=False)
def get_rag_engine():
    return RAGEngine(settings.embedding_model, settings.data_dir)

ollama = get_ollama()
with st.sidebar:
    st.header("Connection")
    st.caption(f"Ollama: `{settings.ollama_host}`")
    st.caption(f"Model: `{settings.ollama_model}`")
    if st.button("Test AI connection", use_container_width=True):
        ok, detail = ollama.health_check()
        if ok: st.success(detail)
        else: st.error(detail)
    st.divider()
    page = st.radio("Workspace", ["Ask documents (RAG)", "Compare documents", "ATS resume screening", "System diagnostics"])
    st.caption("Your files and vector index are stored locally.")

def show_connection_warning():
    ok, detail = ollama.health_check()
    if not ok:
        st.error(detail)
        st.info("On Windows, open PowerShell and run `ollama serve` (or start the Ollama desktop app). Keep it running, then click **Test AI connection**.")
    return ok

if page == "System diagnostics":
    st.subheader("System diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Ollama service**")
        ok, detail = ollama.health_check()
        (st.success if ok else st.error)(detail)
        st.write(f"Configured URL: `{settings.ollama_host}`")
        st.write(f"Configured model: `{settings.ollama_model}`")
    with c2:
        st.write("**Embedding model**")
        st.write(settings.embedding_model)
        try:
            engine = get_rag_engine()
            st.success(f"Embedding model loaded ({engine.embedding_dimension} dimensions).")
        except Exception as e:
            st.error(f"Could not load embedding model: {e}")
    st.write("**Installed Ollama models**")
    try:
        models = ollama.list_models()
        st.write(models or "No models returned.")
    except Exception as e:
        st.error(str(e))
    st.markdown("**Windows quick checks**")
    st.code("ollama list\nollama serve\ncurl http://localhost:11434/api/tags", language="powershell")

elif page == "Ask documents (RAG)":
    st.subheader("Ask questions across your documents")
    st.write("Upload files, index them, and ask questions. Answers include source references and are instructed not to invent facts.")
    uploads = st.file_uploader("Upload documents", type=["pdf","docx","txt","md","csv","xlsx"], accept_multiple_files=True)
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Index uploaded files", type="primary", disabled=not uploads):
            with st.spinner("Extracting text and creating local embeddings..."):
                try:
                    docs = extract_uploaded_files(uploads)
                    if not docs: st.warning("No readable text was found in the uploaded files.")
                    else:
                        engine = get_rag_engine()
                        count = engine.add_documents(docs)
                        st.session_state["rag_engine_ready"] = True
                        st.success(f"Indexed {count} text chunks from {len(docs)} document(s).")
                except Exception as e: st.exception(e)
    with col2:
        st.caption("Supported: PDF, DOCX, TXT, Markdown, CSV, XLSX. Scanned/image-only PDFs need OCR and are reported as empty.")
    try:
        engine = get_rag_engine()
        stats = engine.stats()
        st.caption(f"Local knowledge base: {stats['chunks']} chunks · {stats['documents']} documents")
    except Exception as e:
        st.warning(f"Embedding setup: {e}")
    question = st.text_area("Your question", placeholder="What are the main requirements across these documents?")
    if st.button("Generate answer", type="primary", disabled=not question.strip()):
        if show_connection_warning():
            try:
                engine = get_rag_engine()
                answer, sources = engine.answer(question, ollama)
                st.markdown("### Answer")
                st.markdown(answer)
                st.markdown("### Sources")
                if sources:
                    for s in sources:
                        with st.expander(f"{s['source']} · {s.get('location','') or 'location not available'} · score {s['score']:.3f}"):
                            st.write(s["text"])
                else: st.info("No matching sources were retrieved.")
            except Exception as e: st.exception(e)

elif page == "Compare documents":
    st.subheader("General-purpose document comparison")
    uploads = st.file_uploader("Upload 2 or more documents to compare", type=["pdf","docx","txt","md","csv","xlsx"], accept_multiple_files=True, key="compare_uploads")
    focus = st.text_input("Optional comparison focus", placeholder="e.g. pricing, obligations, skills, deadlines, differences")
    if st.button("Compare documents", type="primary", disabled=not uploads or len(uploads)<2):
        if show_connection_warning():
            try:
                docs = extract_uploaded_files(uploads)
                if len(docs)<2: st.warning("At least two readable documents are required.")
                else:
                    with st.spinner("Comparing documents..."):
                        result = compare_documents(docs, focus, ollama)
                    st.markdown(result)
            except Exception as e: st.exception(e)

elif page == "ATS resume screening":
    st.subheader("ATS-style resume analysis")
    jd_file = st.file_uploader("Upload job description", type=["pdf","docx","txt","md"], key="jd")
    resumes = st.file_uploader("Upload one or more resumes", type=["pdf","docx","txt"], accept_multiple_files=True, key="resumes")
    if st.button("Evaluate candidates", type="primary", disabled=not jd_file or not resumes):
        if show_connection_warning():
            try:
                jd_docs = extract_uploaded_files([jd_file])
                resume_docs = extract_uploaded_files(resumes)
                if not jd_docs or not resume_docs: st.error("Could not extract text from the job description or resumes.")
                else:
                    with st.spinner("Evaluating resumes against job requirements..."):
                        report = evaluate_resumes(jd_docs[0]["text"], resume_docs, ollama)
                    st.markdown(report)
                    st.caption("Decision-support only. Verify evidence manually; do not use this as an automatic hiring decision.")
            except Exception as e: st.exception(e)
