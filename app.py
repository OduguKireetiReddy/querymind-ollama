import streamlit as st
import requests
from rag_engine import RAGEngine

st.set_page_config(page_title="QueryMind — Local AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] { background: #0a0a0a; color: #e0e0e0; }
    .main .block-container { padding-top: 1.5rem; max-width: 1000px; }
    section[data-testid="stSidebar"] { background: #0e0e0e; border-right: 1px solid #1c1c1c; }
    section[data-testid="stSidebar"] * { color: #c0c0c0 !important; }
    section[data-testid="stSidebar"] .stButton > button {
        background: #141414; border: 1px solid #2a2a2a; color: #c0c0c0 !important;
        width: 100%; border-radius: 6px; padding: 0.45rem; font-size: 0.85rem; transition: all 0.2s;
    }
    section[data-testid="stSidebar"] .stButton > button:hover { background: #1e1e1e; border-color: #555; }
    [data-testid="stFileUploader"] { border: 1px dashed #2a2a2a; border-radius: 8px; padding: 0.5rem; background: #0e0e0e; }
    .qm-logo { font-size: 1.6rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; margin-bottom: 0; }
    .qm-logo span { color: #666; }
    .qm-tagline { font-size: 0.78rem; color: #444; margin: 0 0 1.2rem; letter-spacing: 0.04em; text-transform: uppercase; }
    .badge { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 4px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 6px; }
    .badge-doc    { background: #0d1f0d; color: #4caf50; border: 1px solid #1a3a1a; }
    .badge-ollama { background: #150d20; color: #a78bfa; border: 1px solid #2e1a4a; }
    .badge-web    { background: #0d1525; color: #4a9eff; border: 1px solid #1a2e4a; }
    .badge-wiki   { background: #1a1500; color: #c9a227; border: 1px solid #3a3000; }
    .status-pill { display: inline-block; padding: 2px 9px; border-radius: 3px; font-size: 0.7rem;
        font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.4rem; }
    .pill-ready   { background: #0d1f0d; color: #4caf50; border: 1px solid #1a3a1a; }
    .pill-waiting { background: #1a1500; color: #c9a227; border: 1px solid #3a3000; }
    .pill-online  { background: #0d1525; color: #4a9eff; border: 1px solid #1a2e4a; }
    .pill-offline { background: #200d0d; color: #f87171; border: 1px solid #4a1a1a; }
    .metric-card { background: #111; border: 1px solid #1e1e1e; border-radius: 6px; padding: 0.7rem 0.9rem; margin-bottom: 0.4rem; }
    .metric-label { font-size: 0.65rem; color: #444; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-value { font-size: 1.1rem; font-weight: 600; color: #e0e0e0; }
    .rel-wrap { margin: 2px 0 8px; }
    .rel-label { font-size: 0.68rem; color: #444; margin-bottom: 3px; }
    .rel-bg { background: #1a1a1a; border-radius: 3px; height: 3px; }
    .rel-fill { height: 3px; border-radius: 3px; background: #4caf50; }
    .qm-divider { border: none; border-top: 1px solid #1a1a1a; margin: 0.8rem 0; }
    [data-testid="stChatMessage"] { background: #0e0e0e !important; border: 1px solid #1a1a1a; border-radius: 8px; margin-bottom: 0.5rem; }
    .section-label { font-size: 0.65rem; color: #333; text-transform: uppercase; letter-spacing: 0.08em; margin: 0.8rem 0 0.3rem; }
    .model-info { background: #111; border: 1px solid #2a2a2a; border-radius: 6px; padding: 8px 12px; font-size: 11px; color: #555; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────────
if "rag"       not in st.session_state: st.session_state.rag       = RAGEngine()
if "messages"  not in st.session_state: st.session_state.messages  = []
if "doc_ready" not in st.session_state: st.session_state.doc_ready = False
if "doc_stats" not in st.session_state: st.session_state.doc_stats = {}

# ── Check Ollama status ─────────────────────────────────────────────────────────
def ollama_online():
    try:
        r = requests.get("http://localhost:11434", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

online = ollama_online()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="qm-logo">Query<span>Mind</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="qm-tagline">Local AI · phi3:mini · GTX 1650</div>', unsafe_allow_html=True)

    # Ollama status
    if online:
        st.markdown('<span class="status-pill pill-online">● Ollama online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill pill-offline">✕ Ollama offline</span>', unsafe_allow_html=True)
        st.error("Start Ollama: run `ollama serve` in a terminal")

    st.markdown('<div class="model-info">Model: phi3:mini · Running on GTX 1650 GPU · No API key needed</div>', unsafe_allow_html=True)

    st.markdown('<hr class="qm-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload", type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and online:
        if st.button("⚡  Process Documents", use_container_width=True):
            with st.spinner("Indexing..."):
                stats = st.session_state.rag.ingest(uploaded_files)
                st.session_state.doc_ready = True
                st.session_state.doc_stats = stats
            st.success(f"Indexed {stats['chunks']} chunks from {stats['files']} file(s)")

    st.markdown('<hr class="qm-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Status</div>', unsafe_allow_html=True)

    if st.session_state.doc_ready:
        st.markdown('<span class="status-pill pill-ready">● Docs ready</span>', unsafe_allow_html=True)
        stats = st.session_state.doc_stats
        st.markdown(f'<div class="metric-card"><div class="metric-label">Chunks indexed</div><div class="metric-value">{stats.get("chunks",0)}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-label">Files loaded</div><div class="metric-value">{stats.get("files",0)}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill pill-waiting">○ No docs — AI mode</span>', unsafe_allow_html=True)

    st.markdown('<hr class="qm-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
    top_k       = st.slider("Chunks to retrieve", 1, 8, 3)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    threshold   = st.slider("Doc relevance threshold", 0.1, 0.7, 0.40, 0.05)

    st.markdown('<hr class="qm-divider">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("↺ Reset", use_container_width=True):
            st.session_state.messages  = []
            st.session_state.doc_ready = False
            st.session_state.doc_stats = {}
            st.session_state.rag       = RAGEngine()
            st.rerun()

# ── Helpers ─────────────────────────────────────────────────────────────────────
def show_badge(source_type, relevance_score=None):
    badges = {
        "document": ("📄", "badge-doc",    "Document"),
        "ollama":   ("🧠", "badge-ollama", "Ollama phi3:mini"),
        "web":      ("🌐", "badge-web",    "DuckDuckGo"),
        "wikipedia":("📖", "badge-wiki",   "Wikipedia"),
    }
    if source_type in badges:
        icon, cls, label = badges[source_type]
        st.markdown(f'<span class="badge {cls}">{icon} {label}</span>', unsafe_allow_html=True)
    if source_type == "document" and relevance_score:
        pct = min(relevance_score * 100, 100)
        st.markdown(
            f'<div class="rel-wrap"><div class="rel-label">Relevance: {relevance_score:.2f}</div>'
            f'<div class="rel-bg"><div class="rel-fill" style="width:{pct:.0f}%"></div></div></div>',
            unsafe_allow_html=True
        )

def show_sources(sources, source_type):
    if not sources:
        return
    icons = {"document": "📄", "web": "🌐", "wikipedia": "📖"}
    icon  = icons.get(source_type, "🔗")
    with st.expander(f"{icon} {len(sources)} source(s) used"):
        for i, src in enumerate(sources, 1):
            if source_type == "document":
                st.markdown(f"**Chunk {i}** — `{src['source']}` · page {src.get('page','?')} · score `{src.get('score',0):.3f}`")
            else:
                title = src.get("title", f"Source {i}")
                url   = src.get("source", "#")
                st.markdown(f"**[{i}] [{title}]({url})**")
            st.caption(src["text"][:280] + ("..." if len(src["text"]) > 280 else ""))

# ── Main area ───────────────────────────────────────────────────────────────────
st.markdown('<div class="qm-logo" style="font-size:2rem">Query<span>Mind</span></div>', unsafe_allow_html=True)
st.markdown('<div class="qm-tagline">Fully local · phi3:mini on GTX 1650 · No API key · 4-layer search</div>', unsafe_allow_html=True)

st.markdown(
    '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:1rem">'
    '<span class="badge badge-doc">📄 Documents</span>'
    '<span class="badge badge-ollama">🧠 Ollama Local</span>'
    '<span class="badge badge-web">🌐 DuckDuckGo</span>'
    '<span class="badge badge-wiki">📖 Wikipedia</span>'
    '</div>',
    unsafe_allow_html=True
)
st.markdown('<hr class="qm-divider">', unsafe_allow_html=True)

if not st.session_state.messages:
    if not online:
        st.error("Ollama is not running. Open a terminal and run: `ollama serve`")
    else:
        st.markdown('<div style="color:#333;font-size:0.85rem">Ready! Ask anything — running fully on your local GPU.</div>', unsafe_allow_html=True)

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            show_badge(msg.get("source_type"), msg.get("relevance_score"))
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources", []), msg.get("source_type", ""))

# Chat input
if prompt := st.chat_input("Ask anything...", disabled=not online):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking locally..."):
            import rag_engine
            rag_engine.RELEVANCE_THRESHOLD = threshold
            result = st.session_state.rag.query(prompt, top_k=top_k, temperature=temperature)

        show_badge(result["source_type"], result.get("relevance_score"))
        st.markdown(result["answer"])
        show_sources(result.get("sources", []), result["source_type"])

    st.session_state.messages.append({
        "role":            "assistant",
        "content":         result["answer"],
        "sources":         result.get("sources", []),
        "source_type":     result.get("source_type", "ollama"),
        "relevance_score": result.get("relevance_score", 0)
    })
