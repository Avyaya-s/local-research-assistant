# Project Notes — Local Research Assistant

Technical decisions, lessons learned, and open questions captured during development.

---

## Design decisions

### Why three separate files (ingest.py, tools.py, agent.py)

Single responsibility principle applied to an agent project:

- `ingest.py` — runs offline, once, as a preparation step. Never imported by agent code.
- `tools.py` — pure capabilities. No agent logic, no LangGraph, no loop. Just functions.
- `agent.py` — pure orchestration. No tool implementation details, just wiring.

This separation means: adding a new tool = touch only `tools.py`. Changing agent behavior = touch only `agent.py`. Changing chunking strategy = touch only `ingest.py`. Each file has one reason to change.

### Why nomic-embed-text for embeddings

- Runs locally via Ollama — no API key, no data sent externally
- 768-dimension vectors — good balance of quality and speed
- Specifically designed for retrieval tasks — outperforms many larger models on RAG benchmarks
- Free alternative: `mxbai-embed-large` (also via Ollama, 1024 dimensions, slightly better quality)

### Why chunk_size=500 with chunk_overlap=50

500 characters (~100 words) is large enough to capture a complete thought but small enough that retrieval returns focused, relevant passages rather than entire sections. Too large → retrieved chunks contain too much irrelevant context. Too small → chunks lose surrounding context and become cryptic.

50-character overlap (~10 words) ensures sentences spanning chunk boundaries appear in both neighboring chunks, so retrieval never misses a fact because it happened to fall on a boundary.

These are starting values — tune based on your document types. Technical docs with dense information → smaller chunks. Narrative text → larger chunks.

### Why delete vectorstore before rebuilding

Chroma doesn't track which chunks came from which source files. If you update `basics.md` and just run `ingest.py` again without deleting the vectorstore, the old chunks from `basics.md` remain alongside the new ones — the model sees duplicate, potentially contradictory content. Always delete and rebuild for a clean state.

### Why `search_docs` is listed first in the tools list

```python
tools = [search_docs, web_search, fetch_page, save_summary]
```

Tool order subtly influences model behavior — models tend to consider tools listed earlier first. Combined with the system prompt's "search_docs FIRST" instruction, listing it first reinforces the local-before-web routing preference.

---

## Failure modes observed during development

### Thai language switching
**Symptom:** qwen2.5:14b occasionally reasoned internally in Thai or other languages on complex multi-part queries.
**Cause:** qwen2.5 is a multilingual model trained on many languages. On complex queries it sometimes switches to a non-English language for internal reasoning steps.
**Fix applied:** Added "Always respond in English regardless of the language used internally" to system prompt.
**Status:** Reduced but not eliminated — local models aren't 100% reliable on language enforcement.

### Placeholder content in dependent chaining
**Symptom:** When asked to "save the contents of test.txt to backup.txt," qwen wrote placeholder strings ("test file content", "Contents from test.txt") instead of the actual retrieved content.
**Cause:** Model didn't correctly pass the `read_file` observation as the `content` argument to `write_file`. Substituted a description of the content instead of the content itself.
**Fix applied:** More explicit prompting: "write exactly what you read, word for word, unchanged."
**Root cause:** 7B-14B models struggle with verbatim content passing between dependent tool calls. This is a reasoning limitation, not a framework limitation.

### Environmental state interference
**Symptom:** When `backup.txt` already existed from a previous run, qwen used its existing content instead of the freshly read content from `test.txt`.
**Cause:** Model inferred the file's existing content from prior context and used it as a shortcut instead of the actual `read_file` observation.
**Lesson:** Pre-existing state in the environment can silently influence model behavior. Agents don't operate in a vacuum — their outputs depend on what already exists on disk/in the database.

### Web routing instead of local routing
**Symptom:** First run with `concepts.md` indexed, asking about ReAct pattern went to web instead of local docs.
**Cause:** Vectorstore had too few chunks (only 1 from a small test file) — nomic couldn't find a close enough match, so the agent defaulted to web.
**Fix applied:** Added more substantial documents (`basics.md`, `concepts_v2.md`) — 60 chunks total gave nomic enough material to find relevant matches.
**Lesson:** RAG quality depends heavily on the richness of the indexed collection. A sparse vectorstore produces poor retrieval regardless of how good the embedding model is.

---

## Knowledge retrieval techniques — comparison

Six main techniques for giving a model access to external knowledge:

| Technique | This project | Best for | Limitation |
|---|---|---|---|
| RAG | ✅ search_docs | Private doc collections | Only as good as indexed docs |
| Live tool retrieval | ✅ web_search, fetch_page | Current information | External dependency, noisy |
| Short-term memory | ✅ LangGraph state | Session continuity | Lost when session ends |
| Fine-tuning | ❌ | Style/behavior baking | Expensive, stale, bad for facts |
| Context stuffing | ❌ | Small single documents | Context window limits |
| Long-term memory | ❌ planned | Cross-session continuity | Requires persistent storage |
| Knowledge graphs | ❌ | Structured relational data | Complex to build/maintain |

### RAG vs fine-tuning (important distinction)

RAG and fine-tuning are often confused as alternatives. They solve different problems:

- **Fine-tuning** teaches the model *how to behave* — style, tone, domain-specific language patterns. Good for "respond like a medical professional" or "always format responses as JSON."
- **RAG** gives the model *knowledge to retrieve* — specific facts, documents, data. Good for "answer questions about our company's internal docs."

Fine-tuning is almost always the wrong choice for knowledge retrieval — models hallucinate specific facts even after fine-tuning. RAG is more reliable because the source text is always present in context, not reconstructed from weights.

---

## Model compatibility findings

| Model | LangGraph compatible | Reasoning quality | VRAM | Notes |
|---|---|---|---|---|
| phi4 (14B) | ❌ | High | ~9GB | Best local reasoning, no native tools API in Ollama |
| qwen2.5:14b | ✅ | Good | ~9GB | Best balance for this project |
| qwen2.5:7b | ✅ | Medium | ~5GB | Struggles with dependent chaining |
| llama3.2 (3B) | ✅ | Limited | ~3GB | Too weak for multi-step tasks |

**The local model squeeze:** The best local reasoning model (phi4) isn't LangGraph-compatible. The compatible models have weaker reasoning. This is a current constraint of the local model ecosystem — not a permanent limitation as models improve.

**Production solution:** API models (GPT-4o, Claude Sonnet, Gemini) have both strong reasoning AND native tool support. LangGraph works perfectly with them. For production deployments, API models are the right choice.

---

## Open questions / future improvements

- **Long-term memory** — add LangGraph checkpointing so the agent remembers previous research sessions. Currently all context is lost when you quit.
- **Better document formats** — add support for Word docs (.docx), web pages (auto-fetch and index URLs), and audio transcripts.
- **Chunk size tuning** — experiment with larger chunks (800-1000 chars) for narrative text vs smaller (200-300 chars) for technical reference material.
- **Hybrid search** — combine semantic search (current) with keyword search (BM25) for better retrieval on specific technical terms that embeddings sometimes miss.
- **Source deduplication** — if multiple chunks from the same document are retrieved, consolidate them before injecting into context.
- **Recency filtering** — add date metadata to chunks during ingestion so the agent can filter by recency ("only search documents from the last 6 months").
- **Multi-modal** — add image understanding for PDFs with diagrams or screenshots.
- **Evaluation** — build a small test set of question-answer pairs to measure retrieval accuracy as the document collection grows.
