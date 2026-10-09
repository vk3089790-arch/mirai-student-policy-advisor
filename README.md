# MirAI Autonomous Student Policy Advisor

RAG-powered student policy advisor grounded in the **Mirai School of Technology Policy Handbook 2026**.

Built with: FastAPI · LangChain LCEL · ChromaDB · Google Gemini · Streamlit

---

## Quick Start

### 1. Create a Python environment

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Google API key

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your real key
```

Get a key at: https://aistudio.google.com/app/apikey

### 4. Ingest the handbook

**Via Streamlit UI (recommended):**
- Upload `Mirai_SoT_Policy_Handbook_2026.pdf` using the sidebar uploader.

**Via curl:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@Mirai_SoT_Policy_Handbook_2026.pdf"
```

### 5. Run the backend (Terminal 1)

```bash
uvicorn backend:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 6. Run the frontend (Terminal 2)

```bash
streamlit run frontend.py
```

Opens at: http://localhost:8501

### 7. Run the evaluation

After ingesting the handbook:

```bash
python evaluate.py
```

Results are written to `rag_eval_scores.csv`.

---

## Architecture

```
Student → Streamlit (frontend.py)
            ↓ HTTP
         FastAPI (backend.py)
            ↓
         MultiQueryRetriever
            ↓
         ChromaDB (local, persistent)
            ↓
         LCEL Chain → Gemini (answer)
```

**Anti-hallucination guardrails:**
- Answers only from retrieved handbook chunks.
- Temperature = 0.0.
- Explicit system prompt refusing to invent policies, fines, or procedures.
- Returns a polite refusal when context is insufficient.

---

## Four Certification Queries

| # | Question | Expected |
|---|----------|----------|
| T1 | 72% attendance → how many marks? | 4 marks |
| T2 | Ratnam campus medical leave — who to email, how many days? | Email Yashaswini Ma'am within 7 days |
| T3 | New Cybersecurity society — ask Management directly? | 40% batch support; submit to Faculty Coordinator first |
| T4 | Fine for smoking on campus? | No monetary fine invented; Disciplinary Committee action cited |

---

## Citations and Refusals

- Every answer includes source **page numbers** from the handbook.
- If the handbook does not contain an answer, the system says:
  > "I could not find that information in the handbook. Please consult the administration."
- The system **never** invents policies, email addresses, fines, or deadlines.

---

## Configurable Models

Set these in `.env` to override defaults:

| Variable | Default |
|----------|---------|
| `CHAT_MODEL` | `gemini-1.5-flash` |
| `EMBED_MODEL` | `models/text-embedding-004` |
| `CHROMA_DIR` | `./chroma_db` |
| `BACKEND_URL` | `http://localhost:8000` |

---

## Known Limitations & Troubleshooting

| Problem | Fix |
|---------|-----|
| `GOOGLE_API_KEY not set` | Copy `.env.example` → `.env` and add your key |
| `503 No documents indexed` | Ingest the handbook via `/ingest` first |
| Backend not reachable | Start `uvicorn backend:app --reload --port 8000` |
| Wrong model error | Set `CHAT_MODEL` in `.env` to an available Gemini model |
| ChromaDB conflicts | Delete `chroma_db/` and re-ingest |
| numpy version warning | Safe to ignore — does not affect RAG functionality |
| `429 free tier daily quota` | Switch `CHAT_MODEL` in `.env` to a different model (e.g. `gemini-2.5-flash`); each model has its own 20 req/day free quota. MultiQueryRetriever uses ~4 calls per question — budget accordingly. |

---

## Project Files

```
miraihandbook/
├── backend.py          # FastAPI RAG backend
├── frontend.py         # Streamlit chat UI
├── evaluate.py         # LLM-as-a-judge evaluation
├── requirements.txt    # Python dependencies
├── rag_eval_scores.csv # Evaluation output (populated by evaluate.py)
├── .env.example        # API key template
├── .gitignore
└── README.md
```

> **Note:** The handbook PDF and `chroma_db/` are excluded from Git (see `.gitignore`).
