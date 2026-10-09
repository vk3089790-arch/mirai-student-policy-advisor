import sys
import os
import time
sys.path.insert(0, '/Users/vinaykumar/miraihandbook')
os.chdir('/Users/vinaykumar/miraihandbook')

from dotenv import load_dotenv
load_dotenv('/Users/vinaykumar/miraihandbook/.env')

import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
EMBED_MODEL    = os.environ.get("EMBED_MODEL", "models/gemini-embedding-001")
CHAT_MODEL     = os.environ.get("CHAT_MODEL", "gemini-3.8-flash")
CHROMA_DIR     = "./chroma_db"
COLLECTION_NAME = "mirai_handbook"

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

def format_docs(docs):
    parts = []
    for d in docs:
        page = d.metadata.get("page", "?")
        src  = d.metadata.get("source", "handbook")
        parts.append(f"[Page {page} | {src}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def ask(question, retriever, llm):
    docs = retriever.invoke(question)
    if not docs:
        return "No relevant context found.", [], 0
    chain = (
        {"context": lambda _: format_docs(docs), "question": RunnablePassthrough()}
        | rag_prompt | llm | StrOutputParser()
    )
    answer = chain.invoke(question)
    sources = []
    seen = set()
    for d in docs:
        k = (d.metadata.get("source"), d.metadata.get("page"))
        if k not in seen:
            seen.add(k)
            sources.append(k)
    return answer, sources, len(docs)

def main():
    print(f"Models: embed={EMBED_MODEL}  chat={CHAT_MODEL}\n")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY)
    vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=CHROMA_DIR)
    count = vectorstore._collection.count()
    print(f"ChromaDB chunks available: {count}")
    if count == 0:
        print("ERROR: No documents indexed. Run ingestion first.")
        return

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.0, google_api_key=GOOGLE_API_KEY)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

    tests = [
        {
            "id": "T1",
            "q": "I have 72% attendance. How many attendance marks will I get?",
            "expected": "4 marks"
        },
        {
            "id": "T2",
            "q": "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?",
            "expected": "Email Yashaswini Ma'am within 7 days"
        },
        {
            "id": "T3",
            "q": "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?",
            "expected": "40% batch support; submit to Faculty Coordinator first, not Management"
        },
        {
            "id": "T4",
            "q": "How much is the fine for smoking a cigarette on campus?",
            "expected": "No monetary fine invented; Disciplinary Committee action cited"
        },
    ]

    # Which tests to run (pass IDs as args, or all if no args)
    run_ids = sys.argv[1:] if len(sys.argv) > 1 else [t["id"] for t in tests]
    selected = [t for t in tests if t["id"] in run_ids]

    SLEEP_BETWEEN = 75  # seconds — free tier = 5 RPM; MultiQueryRetriever burns ~4 calls/query

    for i, t in enumerate(selected):
        print(f"\n{'='*60}")
        print(f"[{t['id']}] {t['q']}")
        print(f"Expected: {t['expected']}")
        print("-"*60)
        try:
            answer, sources, n = ask(t['q'], retriever, llm)
            print(f"Answer ({n} chunks):\n{answer}")
            print(f"Sources: {sources}")
        except Exception as e:
            print(f"ERROR: {e}")
        if i < len(selected) - 1:
            print(f"\n⏳ Sleeping {SLEEP_BETWEEN}s to respect free-tier rate limit...")
            time.sleep(SLEEP_BETWEEN)

if __name__ == "__main__":
    main()
