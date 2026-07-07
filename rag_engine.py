"""
rag_engine.py — Agentic RAG with Ollama (fully local)
Layer 1: FAISS document search
Layer 2: Ollama phi3:mini AI knowledge (local GPU)
Layer 3: DuckDuckGo web search (live data)
Layer 4: Wikipedia API (facts fallback)
"""

import time
import tempfile
import urllib.parse
import urllib.request
import json
import requests
from pathlib import Path

import faiss
import numpy as np
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi3:mini"
RELEVANCE_THRESHOLD = 0.40

# Greetings — handle instantly
GREETINGS = [
    "hi", "hello", "hey", "hii", "helo", "sup", "yo",
    "good morning", "good evening", "good afternoon", "good night",
    "hi there", "hello there", "hey there", "greetings"
]

# Live data keywords — skip Ollama, go straight to web
LIVE_KEYWORDS = [
    "latest", "today", "news", "current", "live", "now", "price",
    "winner", "score", "2024", "2025", "2026", "breaking", "update",
    "recently", "this week", "this month", "yesterday", "just happened"
]


class RAGEngine:
    def __init__(self):
        self.chunks = []
        self.index  = None
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    # ── Ollama chat call ────────────────────────────────────────────────────────
    def _ollama_chat(self, prompt: str, temperature: float = 0.3) -> str:
        payload = {
            "model":    OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   False,
            "options":  {"temperature": temperature}
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama is not running. Start it with: ollama serve")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedder.encode(texts, show_progress_bar=False)
        return np.array(vectors, dtype=np.float32)

    def _embed_query(self, query: str) -> np.ndarray:
        return np.array(self.embedder.encode([query]), dtype=np.float32)

    # ── Document ingestion ──────────────────────────────────────────────────────
    def ingest(self, uploaded_files) -> dict:
        all_docs = []
        for file in uploaded_files:
            suffix = Path(file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            try:
                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                    docs   = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = file.name
                else:
                    loader = TextLoader(tmp_path, encoding="utf-8")
                    docs   = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = file.name
                        doc.metadata["page"]   = 0
                all_docs.extend(docs)
            finally:
                import os; os.unlink(tmp_path)

        if not all_docs:
            return {"chunks": 0, "files": 0}

        split_docs = self.splitter.split_documents(all_docs)
        self.chunks = [
            {
                "text":   doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page":   doc.metadata.get("page", 0)
            }
            for doc in split_docs if doc.page_content.strip()
        ]

        texts   = [c["text"] for c in self.chunks]
        vectors = self._embed_texts(texts)
        faiss.normalize_L2(vectors)
        dim         = vectors.shape[1]
        self.index  = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

        return {"chunks": len(self.chunks), "files": len(uploaded_files)}

    # ── Document retrieval ──────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if self.index is None or not self.chunks:
            return []
        query_vec = self._embed_query(query)
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            retrieved.append(chunk)
        return retrieved

    # ── Layer 2: Ollama local knowledge ────────────────────────────────────────
    def _ollama_knowledge(self, question: str, temperature: float) -> dict:
        prompt = f"""Answer the following question from your training knowledge.

IMPORTANT RULES:
- If you know the answer confidently, answer directly and clearly.
- If the question is about very recent events after your training cutoff,
  start your response with: "NEEDS_WEB_SEARCH:" followed by a brief reason.
- If you genuinely don't know, start with: "NEEDS_WEB_SEARCH:" followed by reason.
- For coding, math, concepts, definitions — always answer directly.
- Be concise and accurate.

QUESTION: {question}

ANSWER:"""
        answer    = self._ollama_chat(prompt, temperature)
        needs_web = answer.startswith("NEEDS_WEB_SEARCH:")
        if needs_web:
            answer = answer.replace("NEEDS_WEB_SEARCH:", "").strip()
        return {"answer": answer, "needs_web": needs_web}

    # ── Layer 3: DuckDuckGo ─────────────────────────────────────────────────────
    def _duckduckgo_search(self, query: str, max_results: int = 4) -> list[dict]:
        results = []
        for wait in [1.5, 4, 4]:
            try:
                time.sleep(wait)
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results, safesearch="off"):
                        results.append({
                            "text":   r.get("body", ""),
                            "source": r.get("href", ""),
                            "title":  r.get("title", ""),
                            "page":   "web"
                        })
                if results:
                    return results
            except Exception:
                continue
        return []

    # ── Layer 4: Wikipedia ──────────────────────────────────────────────────────
    def _wikipedia_search(self, query: str) -> list[dict]:
        results = []
        try:
            search_url = (
                "https://en.wikipedia.org/w/api.php?action=query"
                "&list=search&format=json&srlimit=2&srsearch="
                + urllib.parse.quote(query)
            )
            req = urllib.request.Request(search_url, headers={"User-Agent": "QueryMind/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())

            hits = data.get("query", {}).get("search", [])
            if not hits:
                return []

            for hit in hits[:2]:
                title = hit["title"]
                extract_url = (
                    "https://en.wikipedia.org/w/api.php?action=query"
                    "&prop=extracts&exintro=true&explaintext=true"
                    "&format=json&redirects=1&titles="
                    + urllib.parse.quote(title)
                )
                req2 = urllib.request.Request(extract_url, headers={"User-Agent": "QueryMind/1.0"})
                with urllib.request.urlopen(req2, timeout=8) as resp2:
                    data2 = json.loads(resp2.read().decode())

                pages = data2.get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = page.get("extract", "").strip()
                    if extract and len(extract) > 100:
                        results.append({
                            "text":   extract[:600] + ("..." if len(extract) > 600 else ""),
                            "source": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                            "title":  f"Wikipedia — {title}",
                            "page":   "wikipedia"
                        })
        except Exception:
            pass
        return results

    # ── Generate answer from web/wiki context ───────────────────────────────────
    def _generate_from_context(self, question: str, sources: list[dict], temperature: float) -> str:
        context_parts = []
        for i, r in enumerate(sources, 1):
            context_parts.append(f"[Source {i} — {r.get('title','')}]\n{r['text']}")
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant. Answer the question using the search results below.

SEARCH RESULTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Give a DIRECT answer immediately.
- For sports/events: state the winner, score, key details directly.
- For prices: state the exact value found.
- Mention sources at the end.
- Be concise and accurate.

ANSWER:"""
        return self._ollama_chat(prompt, temperature)

    # ── Main query pipeline ─────────────────────────────────────────────────────
    def query(self, question: str, top_k: int = 3, temperature: float = 0.3) -> dict:
        q_lower = question.lower().strip()

        # ── Greeting handler ──────────────────────────────────────────────────
        if q_lower in GREETINGS or q_lower.rstrip("!?") in GREETINGS:
            return {
                "answer":          "Hi! I'm QueryMind 🧠 — running fully local on your GPU via Ollama! Ask me anything.",
                "sources":         [],
                "source_type":     "ollama",
                "relevance_score": 0,
                "used_web":        False
            }

        is_live = any(kw in q_lower for kw in LIVE_KEYWORDS)

        # ── Layer 1: Document search ──────────────────────────────────────────
        doc_sources = self.retrieve(question, top_k=top_k)
        best_score  = doc_sources[0]["score"] if doc_sources else 0.0

        if doc_sources and best_score >= RELEVANCE_THRESHOLD:
            context_parts = []
            for i, src in enumerate(doc_sources, 1):
                context_parts.append(
                    f"[Source {i} — {src['source']}, page {src.get('page','?')}]\n{src['text']}"
                )
            context = "\n\n---\n\n".join(context_parts)
            prompt = f"""You are a helpful assistant answering from document excerpts.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based only on the context above.
- Cite [Source 1], [Source 2] etc when referencing.
- Be concise and accurate.

ANSWER:"""
            answer = self._ollama_chat(prompt, temperature)
            return {
                "answer":          answer,
                "sources":         doc_sources,
                "source_type":     "document",
                "relevance_score": round(best_score, 3),
                "used_web":        False
            }

        # ── Layer 2: Ollama local knowledge (skip for live queries) ──────────
        if not is_live:
            ollama_result = self._ollama_knowledge(question, temperature)
            if not ollama_result["needs_web"]:
                return {
                    "answer":          ollama_result["answer"],
                    "sources":         [],
                    "source_type":     "ollama",
                    "relevance_score": round(best_score, 3),
                    "used_web":        False
                }

        # ── Layer 3: DuckDuckGo ───────────────────────────────────────────────
        ddg_results = self._duckduckgo_search(question)
        if ddg_results:
            answer = self._generate_from_context(question, ddg_results, temperature)
            return {
                "answer":          answer,
                "sources":         ddg_results,
                "source_type":     "web",
                "relevance_score": round(best_score, 3),
                "used_web":        True
            }

        # ── Layer 4: Wikipedia ────────────────────────────────────────────────
        wiki_results = self._wikipedia_search(question)
        if wiki_results:
            answer = self._generate_from_context(question, wiki_results, temperature)
            return {
                "answer":          answer,
                "sources":         wiki_results,
                "source_type":     "wikipedia",
                "relevance_score": round(best_score, 3),
                "used_web":        True
            }

        # ── Final fallback ────────────────────────────────────────────────────
        ollama_result = self._ollama_knowledge(question, temperature)
        return {
            "answer":          ollama_result["answer"],
            "sources":         [],
            "source_type":     "ollama",
            "relevance_score": round(best_score, 3),
            "used_web":        False
        }
