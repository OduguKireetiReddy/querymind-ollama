# QueryMind — Local Agentic RAG Chatbot

A fully local, agentic RAG (Retrieval-Augmented Generation) chatbot that intelligently 
routes each query through multiple retrieval layers — running entirely offline with 
no API keys and no cloud costs.

## Problem
Most RAG chatbots rely on a single knowledge source and cloud LLM APIs, which means 
ongoing costs, no offline access, and no fallback when the retrieval source doesn't 
have the answer. QueryMind solves this with a 4-layer agentic pipeline that runs 
fully on local hardware.

## Architecture
1. **Document Search (FAISS)** — user-uploaded PDFs/text files are chunked, embedded 
   (`sentence-transformers/all-MiniLM-L6-v2`), and indexed in FAISS for fast retrieval.
2. **Local LLM Knowledge (Ollama · phi3:mini)** — for general questions, the model 
   answers directly from its own knowledge, running locally on a GTX 1650 (4GB VRAM).
3. **Live Web Search (DuckDuckGo)** — triggered automatically for time-sensitive 
   queries (prices, news, "latest", "today", etc.) or when the LLM doesn't know the answer.
4. **Wikipedia Fallback** — final fallback layer for factual queries when web search 
   comes up empty.

The engine decides which layer to use per-query — checking document relevance scores 
first, then live-query keywords, before falling back through web search and Wikipedia.

## Tech Stack
- **UI**: Streamlit (custom dark theme, real-time Ollama status indicator)
- **LLM**: Ollama running `phi3:mini`, fully local — no API key required
- **Vector Search**: FAISS + Sentence Transformers
- **Document Loading**: LangChain (`PyPDFLoader`, `TextLoader`, `RecursiveCharacterTextSplitter`)
- **Web Search**: DuckDuckGo Search API
- **Facts Fallback**: Wikipedia API

## Key Features
- Adjustable retrieval settings (chunks to retrieve, temperature, relevance threshold)
- Source attribution and relevance scoring shown in the UI for every answer
- Automatic detection of "live" queries (news, prices, current events) to skip stale 
  LLM knowledge and go straight to the web
- Fully offline — no API keys, no per-token costs

## Key Outcomes / Learnings
- Built and optimized a functioning agentic RAG pipeline to run on a resource-constrained 
  local GPU (4GB VRAM), balancing model size, latency, and answer quality.
- Learned how to design a routing/decision layer that picks the right retrieval source 
  automatically instead of hardcoding a single pipeline.
- Compared this local setup against a cloud-based version (Groq API + LLaMA 3.1 8B) to 
  understand trade-offs between speed, cost, and offline capability.

## Setup

```bash
# 1. Install Ollama and pull the model
ollama pull phi3:mini
ollama serve

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```
