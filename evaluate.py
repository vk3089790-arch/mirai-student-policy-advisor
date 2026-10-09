"""
evaluate.py — LLM-as-a-Judge evaluation for MirAI Policy Advisor
Runs the 4 mandatory certification queries directly through the RAG pipeline.
Writes real results to rag_eval_scores.csv.
"""

import os
import csv
import json
import time
import datetime
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
CHAT_MODEL     = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
# Use a lighter model for judging to conserve daily quota
JUDGE_MODEL    = os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite")
CHROMA_DIR     = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "mirai_handbook"
OUTPUT_CSV     = "rag_eval_scores.csv"

# ── Four certification queries ─────────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "T1",
        "question": "I have 72% attendance. How many attendance marks will I get?",
        "expected": "4 marks",
    },
    {
        "id": "T2",
        "question": (
            "I study at the Ratnam campus. I got sick and need medical leave. "
            "Who do I email and how many days do I have to submit my documents?"
        ),
        "expected": "Email Yashaswini Ma'am within exactly 7 days of the illness or treatment.",
    },
    {
        "id": "T3",
        "question": (
            "We want to start a new Cybersecurity society under the Tech Club. "
            "Do we ask Management directly?"
        ),
        "expected": (
            "40% batch support required; submit proposal to Faculty Coordinator first, not Management."
        ),
    },
    {
        "id": "T4",
        "question": "How much is the fine for smoking a cigarette on campus?",
        "expected": (
            "Tobacco is prohibited and leads to Disciplinary Committee action. "
            "No specific monetary fine should be invented."
        ),
    },
]

JUDGE_PROMPT = """You are an impartial LLM judge evaluating a RAG-based student policy advisor.

Question: {question}

Expected outcome: {expected}

Generated answer:
{answer}

Evaluate the answer on:
1. Factual correctness vs expected outcome
2. Completeness
3. Faithfulness (no hallucination — answer must not invent policies or fines)

For T4 (smoking fine): if the answer correctly REFUSES to invent a monetary fine, that is a PASS and deserves score 5.

Respond ONLY with valid JSON (no markdown fences):
{{"score": <1-5>, "reasoning": "<one concise sentence>"}}"""

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

def build_chain_components():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL, temperature=0.0, google_api_key=GOOGLE_API_KEY
    )
    # Use basic retriever (k=6) — avoids 3 extra LLM calls from MultiQueryRetriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}"),
    ])

    def format_docs(docs):
        parts = []
        for d in docs:
            page = d.metadata.get("page", "?")
            src  = d.metadata.get("source", "handbook")
            parts.append(f"[Page {page} | {src}]\n{d.page_content}")
        return "\n\n---\n\n".join(parts)

    return retriever, llm, rag_prompt, format_docs, StrOutputParser(), RunnablePassthrough


def run_rag(question, retriever, llm, rag_prompt, format_docs, StrOutputParser, RunnablePassthrough):
    docs = retriever.invoke(question)
    if not docs:
        return "No relevant context found.", [], 0

    chain = (
        {"context": lambda _: format_docs(docs), "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)

    sources = []
    seen = set()
    for d in docs:
        k = (d.metadata.get("source", "handbook"), d.metadata.get("page", "?"))
        if k not in seen:
            seen.add(k)
            sources.append(k)

    context_snippet = "\n\n".join(
        f"[p.{d.metadata.get('page','?')}] {d.page_content[:300]}" for d in docs[:3]
    )
    return answer, sources, context_snippet


def judge(question, expected, answer):
    if not GOOGLE_API_KEY:
        return 0, "No API key — cannot run judge."
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        judge_llm = ChatGoogleGenerativeAI(
            model=JUDGE_MODEL, temperature=0.0, google_api_key=GOOGLE_API_KEY
        )
        prompt = JUDGE_PROMPT.format(
            question=question, expected=expected, answer=answer
        )
        result = judge_llm.invoke([HumanMessage(content=prompt)])
        raw = result.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        return int(parsed.get("score", 0)), parsed.get("reasoning", "")
    except Exception as e:
        return 0, f"Judge error: {e}"


def main():
    print(f"\n{'='*60}")
    print(f"MirAI Evaluation  —  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Chat model : {CHAT_MODEL}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Embed model: {EMBED_MODEL}")
    print(f"Output     : {OUTPUT_CSV}")
    print(f"{'='*60}\n")

    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY not set. Copy .env.example → .env and add your key.")
        return

    # Build RAG components once
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    try:
        retriever, llm, rag_prompt, format_docs, sop, rp = build_chain_components()
        count = retriever.vectorstore._collection.count()
        print(f"ChromaDB chunks available: {count}")
        if count == 0:
            print("ERROR: No documents indexed. Ingest the handbook first:")
            print("  python3 -c \"import backend; ...\" OR POST /ingest via the API")
            return
    except Exception as e:
        print(f"ERROR initialising RAG components: {e}")
        return

    fieldnames = [
        "id", "question", "expected", "answer",
        "retrieved_context", "score", "reasoning",
        "source_pages", "grounded", "timestamp",
    ]
    rows = []
    SLEEP_S = 65  # seconds between tests (basic retriever = 1 LLM call → 2 calls with judge, well under 5 RPM)

    for i, tc in enumerate(TEST_CASES):
        print(f"── {tc['id']}: {tc['question'][:70]}…")
        timestamp = datetime.datetime.now().isoformat()
        try:
            answer, sources, context_snippet = run_rag(
                tc["question"], retriever, llm, rag_prompt, format_docs, StrOutputParser, RunnablePassthrough
            )
            grounded = "I could not find" not in answer and "consult the administration" not in answer.lower()
            source_pages = ", ".join(f"p.{s[1]}" for s in sources)
            print(f"   Answer: {answer[:150]}…")

            print(f"   Running judge…")
            time.sleep(15)  # brief pause before judge call
            score, reasoning = judge(tc["question"], tc["expected"], answer)
            print(f"   Score: {score}/5 | {reasoning}\n")

        except Exception as e:
            answer  = f"ERROR: {e}"
            context_snippet = ""
            source_pages    = ""
            grounded        = False
            score           = 0
            reasoning       = str(e)
            print(f"   ERROR: {e}\n")

        rows.append({
            "id": tc["id"],
            "question": tc["question"],
            "expected": tc["expected"],
            "answer": answer,
            "retrieved_context": context_snippet,
            "score": score,
            "reasoning": reasoning,
            "source_pages": source_pages,
            "grounded": grounded,
            "timestamp": timestamp,
        })

        if i < len(TEST_CASES) - 1:
            print(f"   ⏳ Sleeping {SLEEP_S}s before next test (rate-limit guard)…\n")
            time.sleep(SLEEP_S)

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    scored = [r for r in rows if isinstance(r["score"], int) and r["score"] > 0]
    avg = sum(r["score"] for r in scored) / len(scored) if scored else 0
    print(f"{'='*60}")
    print(f"✅ Evaluation complete → {OUTPUT_CSV}")
    print(f"   Tests run: {len(rows)} | Scored: {len(scored)} | Average: {avg:.1f}/5")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
