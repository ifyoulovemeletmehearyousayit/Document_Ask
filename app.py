from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 600

st.set_page_config(page_title="DocAsk", page_icon="⬛", layout="wide")
st.markdown("""
    <style>
        #MainMenu, footer, header { visibility: hidden; }
        .stButton > button {
            border-radius: 4px;
            border: 1px solid #333;
            transition: 0.3s;
        }
        .stButton > button:hover { border-color: #fff; color: #fff; }
    </style>
""", unsafe_allow_html=True)


# ─── helpers ───────────────────────────────────────────────────────────────────

def fetch_documents() -> list[str]:
    try:
        res = requests.get(f"{API_URL}/documents", timeout=5)
        res.raise_for_status()
        return res.json().get("documents", [])
    except Exception:
        return []


# ─── session state init ────────────────────────────────────────────────────────

if "docs" not in st.session_state:
    st.session_state.docs = fetch_documents()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ─── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⬛ Workspace")
    st.caption("Manage your knowledge base")
    st.divider()

    with st.expander("➕ Upload New Document", expanded=False):
        uploaded_file = st.file_uploader(
            "Select PDF or Image",
            type=["pdf", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if uploaded_file and st.button("Upload & Process", use_container_width=True):
            with st.spinner("Processing..."):
                ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
                endpoint = "pdf" if ext == "pdf" else "image"
                try:
                    res = requests.post(
                        f"{API_URL}/upload/{endpoint}",
                        files={"file": (uploaded_file.name, uploaded_file, "application/octet-stream")},
                        timeout=REQUEST_TIMEOUT,
                    )
                    if res.status_code == 200:
                        st.success("Uploaded!")
                        st.session_state.docs = fetch_documents()
                    else:
                        st.error("Upload failed.")
                except Exception as exc:
                    st.error(f"Error: {exc}")

    st.divider()

    col1, col2 = st.columns([4, 1])
    col1.markdown("**Database**")
    if col2.button("🔄", help="Refresh"):
        st.session_state.docs = fetch_documents()

    docs = st.session_state.docs
    if not docs:
        st.caption("No documents found.")
    else:
        for doc in docs:
            c1, c2 = st.columns([5, 1])
            c1.caption(f"📄 {doc}")
            if c2.button("✕", key=f"del_{doc}", help="Delete"):
                requests.delete(f"{API_URL}/documents/{doc}", timeout=10)
                st.session_state.docs = fetch_documents()
                st.rerun()

    st.divider()

    st.markdown("**Search Target**")
    source_options = ["All documents"] + docs
    selected_source = st.selectbox("Query from:", options=source_options, label_visibility="collapsed")
    source_filter = None if selected_source == "All documents" else selected_source


# ─── main chat ─────────────────────────────────────────────────────────────────

st.title("Document Ask")
st.caption("Ask questions based on your uploaded context.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📎 **Sources:** {', '.join(msg['sources'])}")

if prompt := st.chat_input("Type your question here... (e.g. สรุปเนื้อหาให้หน่อย)"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            payload = {"question": prompt}
            if source_filter:
                payload["source_filter"] = source_filter

            try:
                res = requests.post(f"{API_URL}/query", json=payload, timeout=REQUEST_TIMEOUT)
                if res.status_code == 200:
                    data = res.json()
                    answer, sources = data["answer"], data.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        st.caption(f"📎 **Sources:** {', '.join(sources)}")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    st.error("Error generating answer.")
            except Exception as exc:
                st.error(f"Connection failed: {exc}")