# QueryMind (Ollama Edition)

An agentic Retrieval-Augmented Generation (RAG) chatbot that answers questions by 
pulling context from multiple sources instead of relying on a single static knowledge base — 
running fully offline on local hardware.

## Problem
Most RAG chatbots depend on a single retrieval source and require cloud API calls, 
which means no offline access and ongoing API costs. QueryMind solves this by combining 
multiple retrieval layers and running entirely locally.

## Approach
- **Multi-layer retrieval**: combines FAISS vector search (local knowledge base), 
  live web search, and Wikipedia lookup to answer a wider range of queries accurately.
- **Local LLM inference**: uses Ollama running the `phi3:mini` model, optimized to run 
  on a GTX 1650 (4GB VRAM) — no cloud API dependency.
- **Agentic pipeline**: query → multi-source retrieval → context ranking/assembly → 
  local LLM generation.

## Tech Stack
- Python
- Ollama (phi3:mini)
- FAISS (vector search)
- LangChain (agent orchestration, if used)
- Web search + Wikipedia APIs

## Key Outcomes / Learnings
- Got a fully functional agentic RAG pipeline running end-to-end on constrained 
  local hardware (4GB GPU).
- Learned the trade-offs between model size, latency, and retrieval quality when 
  optimizing for offline/resource-constrained inference.
- Compared against a cloud-based version (Groq API + LLaMA 3.1 8B) to understand 
  speed vs. cost vs. control trade-offs.

## Related
See the companion cloud version: [querymind-groq](https://github.com/OduguKireetiReddy/querymind-groq)
