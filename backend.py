"""
backend.py — MirAI Student Policy Advisor RAG Backend
FastAPI + LangChain + ChromaDB + Google Gemini
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "mirai_handbook"

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set — API calls will fail.")

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MirAI Policy Advisor API",
    description="RAG-powered student policy advisor backed by the Mirai SoT handbook.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not configured.")
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def _get_vectorstore() -> Chroma:
    embeddings = _get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def _get_llm() -> ChatGoogleGenerativeAI:
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not configured.")
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the MirAI Student Policy Advisor. \
Answer ONLY from the handbook passages provided below.

Rules (non-negotiable):
1. If the retrieved context does not contain the answer, say:
   "I could not find that information in the handbook. Please consult the administration."
2. Never invent policies, deadlines, email addresses, fines, or procedures.
3. Never invent monetary fines for any disciplinary matter.
4. Always cite the page number(s) when available.
5. Keep answers factual and concise.

Retrieved context:
{context}
"""

HUMAN_PROMPT = "Question: {question}"

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


def _format_docs(docs) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page", "?")
        source = d.metadata.get("source", "handbook")
        parts.append(f"[Page {page} | {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "chroma_dir": CHROMA_DIR,
        "api_key_set": bool(GOOGLE_API_KEY),
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Ingest a PDF handbook into ChromaDB."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set.")

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        if not pages:
            raise HTTPException(status_code=422, detail="PDF appears to be empty or unreadable.")

        # Tag source with the original filename
        for doc in pages:
            doc.metadata["source"] = file.filename

        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        logger.info("Split PDF into %d chunks.", len(chunks))

        # Embed + persist
        embeddings = _get_embeddings()

        # Remove existing collection to avoid duplicates on re-ingestion
        existing = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
        try:
            existing.delete_collection()
            logger.info("Deleted existing collection for re-ingestion.")
        except Exception:
            pass

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
        )
        count = vectorstore._collection.count()
        logger.info("Persisted %d vectors.", count)

        return {
            "status": "success",
            "filename": file.filename,
            "pages_loaded": len(pages),
            "chunks_stored": count,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed.")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(exc)}")
    finally:
        os.unlink(tmp_path)


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(request: ChatRequest):
    """Answer a student policy question using the ingested handbook."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set.")

    # Check vectorstore has data
    try:
        vectorstore = _get_vectorstore()
        count = vectorstore._collection.count()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store unavailable. Please ingest the handbook first. ({exc})",
        )

    if count == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed. Please ingest the handbook via POST /ingest first.",
        )

    try:
        llm = _get_llm()

        # MultiQueryRetriever to expand the query
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
        )

        # Retrieve docs
        docs = retriever.invoke(question)

        if not docs:
            return {
                "answer": (
                    "I could not find relevant information in the handbook for your question. "
                    "Please consult the administration directly."
                ),
                "grounded": False,
                "sources": [],
                "chunk_count": 0,
            }

        # Build LCEL chain
        chain = (
            {
                "context": lambda _: _format_docs(docs),
                "question": RunnablePassthrough(),
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

        # Collect source metadata
        sources = []
        seen = set()
        for d in docs:
            page = d.metadata.get("page", "unknown")
            src = d.metadata.get("source", "handbook")
            key = (src, page)
            if key not in seen:
                seen.add(key)
                sources.append({"source": src, "page": page})

        return {
            "answer": answer,
            "grounded": True,
            "sources": sources,
            "chunk_count": len(docs),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat failed.")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(exc)}")
