# Local Research Assistant

A local AI research assistant that searches your own document collection and the web to answer questions. Built with LangGraph, LangChain, Chroma, and local models via Ollama — fully private, no API keys required.

## What it does

- Searches your indexed documents semantically (RAG) before going to the web
- Falls back to live web search when the answer isn't in your local collection
- Fetches full page content when search snippets aren't enough
- Saves research summaries to markdown files
- Cites sources — tells you whether the answer came from your docs or the web
- Runs entirely locally — your documents never leave your machine

## Architecture overview

```
┌─────────────────────────────────────────────────────┐
│              LangGraph ReAct Agent                   │
│                                                     │
│  ┌──────────────┐         ┌──────────────────────┐  │
│  │  call_model  │ ──────► │   execute_tools      │  │
│  │  (qwen2.5)   │ ◄────── │                      │  │
│  └──────────────┘         │  search_docs (RAG)   │  │
│         │                 │  web_search           │  │
│         ▼                 │  fetch_page           │  │
│        END                │  save_summary         │  │
│                           └──────────────────────┘  │
└─────────────────────────────────────────────────────┘
         ▲                          ▲
         │                          │
   Ollama (local)            Chroma vectorstore
   qwen2.5:14b               (your indexed docs)
   nomic-embed-text
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Windows/Mac/Linux
- 10GB+ VRAM recommended (for qwen2.5:14b)

## Setup

**1. Clone and create virtual environment:**
```
git clone https://github.com/<your-username>/local-research-assistant.git
cd local-research-assistant
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Mac/Linux:
source .venv/bin/activate
```

**2. Install dependencies:**
```
pip install -r requirements.txt
```

**3. Pull required Ollama models:**
```
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

**4. Add your documents:**

Drop `.pdf`, `.txt`, or `.md` files into the `docs/` folder.

**5. Index your documents:**
```
python ingest.py
```

**6. Run the assistant:**
```
python agent.py
```

## Usage

```
=== Local Research Assistant ===
Model: qwen2.5:14b
Type 'quit' to exit

You: What is the ReAct pattern in agentic AI?
You: What is the current price of Bitcoin?
You: What are the best frameworks for building AI agents?
You: Summarize what you know about LangGraph and save it to langgraph_summary.md
```

Type `quit`, `exit`, or `q` to stop.

---

## How it works — technical deep dive

### The three files

**`ingest.py`** — runs once (or whenever you update your docs). Reads documents, splits them into chunks, embeds each chunk using `nomic-embed-text`, and stores everything in a local Chroma vector database. The agent never calls this — it's a preparation step.

**`tools.py`** — defines the four tools the agent can use. Each function is decorated with `@tool`, which makes LangChain auto-generate its JSON schema from the function name, type hints, and docstring. No manual tool description needed.

**`agent.py`** — builds the LangGraph state machine and runs the conversation loop. Imports tools from `tools.py`, creates the agent with a system prompt, and handles the while-True input loop.

---

### The RAG pipeline (how `search_docs` works)

RAG stands for Retrieval-Augmented Generation. It is a technique — not a library — for giving a model access to external knowledge at query time.

**Ingestion (runs once via `ingest.py`):**

```
your documents
      ↓
chunking (RecursiveCharacterTextSplitter, 500 chars, 50 overlap)
      ↓
embedding (nomic-embed-text converts each chunk to a 768-dimension vector)
      ↓
storage (Chroma stores both the text and its vector locally)
```

**Retrieval (runs every query via `search_docs`):**

```
your query
      ↓
embedding (nomic-embed-text converts query to a vector)
      ↓
similarity search (Chroma finds the 3 closest chunk vectors)
      ↓
retrieved chunks injected into model context
      ↓
model generates answer grounded in your documents
```

**Why semantic search, not keyword search:**

Keyword search matches exact words. Semantic search matches meaning. When you ask "what is the ReAct loop?" and your document says "think → act → observe pattern," semantic search finds it because the meanings are similar — even though the words don't match exactly. This is what the embedding vectors enable.

**Chunking and overlap:**

Documents are split into 500-character chunks with 50-character overlap. The overlap ensures that sentences spanning chunk boundaries aren't missed — both neighboring chunks contain part of that sentence, so retrieval finds it regardless of which chunk boundary it falls near.

---

### The agent loop (how LangGraph runs the agent)

The agent is a LangGraph state machine — a graph of nodes and edges that replaces a hand-written `for` loop:

```
[START]
   ↓
[call_model] ←─────────────────────┐
   ↓                               │
tool calls present?                │
   yes ↓           no ↓           │
[execute_tools]    [END]           │
   └───────────────────────────────┘
```

**State** — LangGraph maintains a `messages` list automatically across every node execution. This is the agent's working memory for the current session — the same growing message history you'd manage manually in a hand-rolled agent loop, automated by the framework.

**`call_model` node** — sends the current message history to qwen2.5:14b via `ChatOllama`. The model decides which tool to call (or gives a final answer if it has enough information).

**Conditional edge** — if the model returned tool calls, route to `execute_tools`. If it returned a final answer (no tool calls), route to END.

**`execute_tools` node** — executes whichever tool the model requested, appends the result as a `ToolMessage` to state, then loops back to `call_model`.

This loop continues until the model gives a final answer — identical in logic to a hand-written ReAct loop, expressed as a proper state machine.

---

### Tool routing (how the agent decides which tool to use)

The system prompt provides routing guidance, and each tool's docstring reinforces it:

- `search_docs` docstring says: "prefer this over web_search when the answer is likely in your local knowledge base" → model checks local docs first
- `web_search` docstring says: "use this for recent events, current prices, latest news" → model uses web for time-sensitive queries
- `fetch_page` docstring says: "use when search snippets are not enough" → model fetches full pages only when necessary
- `save_summary` docstring says: "use when the user asks to save or store results" → model only saves when explicitly asked

This is prompt-driven routing — the model reads the docstrings and reasons about which tool fits the task. No hardcoded `if/else` routing in the agent code.

---

### Knowledge retrieval techniques used

This project combines three of the six main techniques for giving a model external knowledge:

| Technique | Where used | How |
|---|---|---|
| RAG | `search_docs` | Pre-indexed docs → semantic retrieval → context injection |
| Live tool retrieval | `web_search`, `fetch_page` | Real-time web search and page fetching |
| Short-term memory | LangGraph state | Growing message history within each session |

The other three techniques (fine-tuning, context stuffing, knowledge graphs, long-term memory) are not used here but represent natural extension points for future versions.

---

### What disappeared compared to a hand-rolled agent

If you've built agents by hand before, here's what the framework replaced:

| Hand-rolled | Framework replacement |
|---|---|
| `SYSTEM_PROMPT` tool descriptions in English | `@tool` docstrings — auto-generated schemas |
| `parse_action()` — custom JSON parser | Gone — native tool calling, structured output |
| `for step in range(MAX_ITERATIONS)` | LangGraph state machine — nodes and edges |
| `messages.append(...)` — manual history | LangGraph state — automatic history management |
| Markdown fence stripping | Gone — schema enforced at API level |
| JSON nudge retry logic | Gone — framework handles malformed output |

The tool functions themselves (`read_file` → `search_docs`, `web_search`, etc.) are unchanged in concept — plain Python functions that never raise exceptions, always return strings. That discipline carries forward regardless of framework.

---

## Updating your document collection

When you add, remove, or update files in `docs/`:

```
# 1. Delete the existing vectorstore
Remove-Item -Recurse -Force vectorstore   # PowerShell
rm -rf vectorstore                        # Mac/Linux

# 2. Rebuild
python ingest.py
```

Always delete and rebuild rather than appending — Chroma doesn't automatically remove chunks from deleted files, so stale content accumulates if you only append.

## Supported document formats

- `.pdf` — via PyPDF
- `.txt` — plain text
- `.md` — markdown files

## Model configuration

Default models (in `agent.py` and `tools.py`):

```python
MODEL = "qwen2.5:14b"        # agent reasoning model
EMBEDDING_MODEL = "nomic-embed-text"  # embedding model
```

To use a different reasoning model, change `MODEL` in `agent.py`. Any Ollama model that supports native tool calling will work. Models tested:

| Model | Tool support | Reasoning quality | VRAM |
|---|---|---|---|
| qwen2.5:14b | ✅ | Good | ~9GB |
| qwen2.5:7b | ✅ | Medium | ~5GB |
| llama3.2 | ✅ | Limited | ~3GB |
| phi4 | ❌ | High | ~9GB |

Note: phi4 has excellent reasoning but doesn't support Ollama's native tools API — incompatible with LangGraph. This is a known local model tradeoff: models with the best reasoning (phi4) often lack native tool support, while tool-compatible models have weaker reasoning. API-based models (GPT-4o, Claude) have both.

## Known limitations

- **Language switching** — qwen2.5:14b occasionally reasons internally in non-English languages on complex queries. The system prompt includes an English-only instruction but this isn't 100% reliable with local models.
- **Dependent tool chaining** — qwen sometimes writes placeholder content instead of actual retrieved content when chaining read→write operations. Use explicit prompts ("write exactly what you read, word for word") for reliable copying tasks.
- **Stale vectorstore** — if you update a document in `docs/` without rebuilding the vectorstore, the agent searches old chunks. Always rebuild after changes.
- **Context window limits** — very long document collections or many tool calls in one session will eventually hit qwen's context window limit. Not an issue for typical use.

## Project structure

```
local-research-assistant/
├── agent.py          # LangGraph agent — conversation loop
├── tools.py          # Tool functions — search_docs, web_search, fetch_page, save_summary
├── ingest.py         # Document indexing — chunking, embedding, Chroma storage
├── docs/             # Drop your documents here (.pdf, .txt, .md)
├── outputs/          # Saved research summaries land here
├── vectorstore/      # Chroma database (auto-created by ingest.py, gitignored)
├── requirements.txt
└── README.md
```

## Background — what this project builds on

This project was built as part of a structured learning progression through agentic AI:

- **Stage 01-04**: Hand-rolled ReAct agents (single tool → multi-tool → web research) — understanding the mechanism by building it manually
- **Stage 05**: Framework rewrite using LangGraph — understanding what frameworks replace and why
- **Stage 06 (this project)**: First real standalone application — combining RAG, web retrieval, and LangGraph orchestration into a usable tool

The hand-rolled stages taught the exact failure modes (markdown-fence parsing, hallucinated tool-skipping, multi-step planning failures, date laundering, output type drift) that frameworks and better models are built to solve. This project uses those lessons directly.

Full learning progression: [learning-agentic-ai](https://https://github.com/avyaya-s/agentic-ai-learning)
