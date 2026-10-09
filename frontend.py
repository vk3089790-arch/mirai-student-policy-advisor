"""
frontend.py — MirAI Student Policy Advisor Streamlit UI
"""

import streamlit as st
import httpx

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="MirAI Student Policy Advisor",
    page_icon="📘",
    layout="centered",
)

# ── Default backend URL ────────────────────────────────────────────────────────
DEFAULT_BACKEND_URL = "http://localhost:8000"

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("📘 MirAI Student Policy Advisor")
st.caption(
    "Ask any question about the Mirai School of Technology student policies. "
    "All answers are grounded in the official handbook."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    # Store backend URL in session state so it persists across reruns
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = DEFAULT_BACKEND_URL
    backend_url = st.text_input("Backend URL", value=st.session_state.backend_url)
    st.session_state.backend_url = backend_url

    st.markdown("---")
    st.header("📄 Ingest Handbook")
    uploaded_file = st.file_uploader("Upload PDF handbook", type=["pdf"])

    if st.button("📥 Ingest PDF", disabled=uploaded_file is None):
        with st.spinner("Ingesting handbook…"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                resp = httpx.post(f"{backend_url}/ingest", files=files, timeout=120.0)
                resp.raise_for_status()
                data = resp.json()
                st.success(
                    f"✅ Ingested **{data.get('filename', '')}**\n\n"
                    f"Pages: {data.get('pages_loaded')} | Chunks: {data.get('chunks_stored')}"
                )
            except httpx.ConnectError:
                st.error("❌ Cannot connect to backend. Is it running on port 8000?")
            except httpx.HTTPStatusError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    pass
                st.error(f"❌ Ingestion failed: {detail or str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

    st.markdown("---")
    if st.button("🔍 Check Backend Health"):
        try:
            resp = httpx.get(f"{backend_url}/health", timeout=5.0)
            data = resp.json()
            st.json(data)
        except Exception as e:
            st.error(f"Backend unavailable: {e}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat History ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.write(f"• Page {s.get('page', '?')} — {s.get('source', 'handbook')}")

# ── Chat Input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a policy question…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Searching handbook…"):
            try:
                resp = httpx.post(
                    f"{backend_url}/chat",
                    json={"question": prompt},
                    timeout=90.0,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                grounded = data.get("grounded", False)

                st.markdown(answer)

                if not grounded:
                    st.warning("⚠️ Answer may not be grounded in the handbook.")

                if sources:
                    with st.expander("📎 Sources"):
                        for s in sources:
                            st.write(f"• Page {s.get('page', '?')} — {s.get('source', 'handbook')}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except httpx.ConnectError:
                err = (
                    "❌ Cannot connect to the backend. Please start it with:\n\n"
                    "```\nuvicorn backend:app --reload --port 8000\n```"
                )
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except httpx.HTTPStatusError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    pass
                err = f"❌ Backend error: {detail or str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"❌ Unexpected error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
