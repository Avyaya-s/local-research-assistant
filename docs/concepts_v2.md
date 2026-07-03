# concepts.md — Running Notes on Agentic AI Fundamentals

This file is a living log of core concepts as they're learned, in the order they came up. See `basics.md` for the structured beginner explainer; this file is closer to a lab notebook.

---

## Core mental model

- A model is a stateless text-in/text-out function. It has no memory of past calls unless the full history is resent every time.
- A "chat" is just: keep a growing list of messages, resend the whole list every call, append the new reply to the list.
- An "agent" adds exactly one new idea on top of chat: the model's text output can *be* a request to perform an action, which your own code detects and executes, feeding the result back in as a new message.
- This loop — think, act, observe, think again — is called **ReAct** (Reasoning + Acting), from the 2022 paper of the same name.
- The model is the decision-maker. Your code is the actuator. The model never directly touches the filesystem, network, or anything external — it only ever proposes via text.

## The four-layer landscape

1. **Model** — phi4, GPT-4, Claude, etc. Just text in/out.
2. **Agent loop** — ReAct, hand-built or framework-provided. This is what "agent" technically means.
3. **Frameworks** — LangChain, LlamaIndex, CrewAI. Pre-package layer 2's plumbing (parsing, retries, tool registries) — same ideas, less hand-written code.
4. **Patterns/products** — RAG (search tool instead of one-file tool), multi-agent systems (multiple loops talking to each other), MCP (standardized tool-description protocol instead of a hand-rolled system prompt), finished products (Claude Code, Cursor, etc).

## Lessons from 02_tool_calling

**Tool functions must never raise exceptions.** Always return a string, even on failure. A crash kills the loop; a returned error string becomes information the model can reason about.

**The system prompt is the entire contract between you and the model in a hand-rolled agent.** There's no special "tool calling API" at this stage — just plain-English instructions describing an exact text format, which the model may or may not follow perfectly.

**Local/small models do not reliably follow formatting instructions.** Phi4 wrapped JSON in markdown fences on some runs despite being told not to. Parsing must be defensive rather than relying on a "perfect" prompt.

**Models can skip tool calls and hallucinate grounded-sounding answers.** Fixed by adding an explicit rule: "you MUST call the tool first, never guess." After the fix, phi4 reliably called the tool and grounded its answer in the real result.

**The messages list is the only memory the agent has.** Every loop iteration appends the model's raw output and the tool's observation to `messages`, then resends the entire list.

**The tool call and the reasoning over its result are two separate model invocations.** Step 1: model decides to call a tool. Step 2: model sees the original question AND the tool's result together, and reasons over both.

**Growing message history will eventually hit context window limits.** Real frameworks handle this via summarization, truncation, or external memory stores.

## Lessons from 03_multi_tool_agent

**Adding tools doesn't change the loop's shape, only its inventory.** The hard part of agents isn't the loop architecture — it's getting the model to use that loop reliably.

**Side-effect tools (write_file) are a different risk category than read-only tools.** Always verify write-tool results on disk directly rather than trusting the model's claimed success message.

**Output type drift: models can satisfy the letter of a format while violating its intent.** Phi4 returned a JSON list instead of a string for the final answer. Fixed defensively in code — prompts describe intent but don't enforce it, so code must coerce types explicitly.

**Models can try to plan multiple steps ahead instead of acting one step at a time.** Phi4 emitted placeholder text copied from the system prompt example plus multi-step JSON blocks in one response. Fixed with two specific rules: exactly one JSON object per turn, no placeholder text.

**Different-looking failures need different-looking fixes, even when they both "fail to parse."** Diagnosing *why* a failure happens matters more than just noticing *that* it happened.

## Lessons from 03b_native_tools (comparison stage)

**The native tools API fixes all format failures instantly.** Markdown-fence wrapping, placeholder text, multi-step JSON blocks, output type drift — all gone without a single explicit prompt rule.

**But the native API cannot fix reasoning failures.** Llama3.2 (3B) collapsed dependent tool calls into one simultaneous call and later skipped the second tool call entirely, hallucinating a success message.

**Format reliability and reasoning reliability are entirely separate problems with separate solutions.**
- Format problems → solved by structured output APIs or frameworks.
- Reasoning problems → solved by using a larger or better-trained model.

**The `break` fix for dependent tool chains.** The native API allows models to batch multiple tool calls — efficient for independent calls, broken for dependent chains. Adding `break` after the first tool call forces one-at-a-time execution.

**Not all models support Ollama's native tools API.** Phi4 doesn't — required switching to llama3.2 (3B) for the comparison, which introduced model-size reasoning failures as a confounding variable.

## Lessons from 04_web_research_agent

**External tools introduce failure modes outside your control.** Network timeouts, site downtime, anti-scraping blocks, JS-rendered pages — local tools only fail due to things you control; web tools fail for reasons entirely outside your control.

**Web content must be cleaned before feeding to a model.** Raw HTML is enormous and full of noise. `fetch_page` strips non-content tags, filters short lines, and caps output — a deliberate lossy compression trading completeness for context-window fit.

**Date laundering: models can take real but stale facts and relabel them as current.** Fix: inject today's date into the system prompt as an f-string so the model can reason about recency.

**Never diagnose hallucination from truncated terminal output alone.** `observation[:500]` truncates display only — the full observation is always sent to the model. Production approach: log full observations to a debug file, truncate terminal print only.

**Search engines can misinterpret ambiguous abbreviations.** "Open source LLMs" returned Learning Management System results instead of Large Language Model results. The model answered from wrong results without flagging the mismatch.

**Open-ended queries are more likely to trigger mixed narrative + JSON failures.** The "EXACTLY ONE JSON object" rule reduced but didn't eliminate this on complex queries.

## The LangChain → LangGraph architectural shift (important landscape context)

This came up during `05_framework_rewrite` setup and is worth understanding clearly as a landscape concept, not just a version detail.

**The old way (LangChain pre-1.0):**
LangChain had a built-in `AgentExecutor` — a rigid, hardcoded Python loop that managed the agent's think → act → observe cycle. It worked for simple cases but struggled with complex agent logic, branching, parallel execution, and edge cases. This is architecturally very close to what was hand-built in `02`-`04`: a `for` loop, a parser, a message list, retry logic.

**The modern way (LangChain 1.x):**
The LangChain team deprecated `AgentExecutor` entirely. The new agent execution engine is **LangGraph** — a proper state machine framework where the agent loop is expressed as a graph of nodes and edges rather than a hardcoded `for` loop. LangChain 1.x now depends on LangGraph as its execution layer; when you build agents with LangChain today, they compile into LangGraph structures behind the scenes.

**Why this matters for your learning arc:**

Your hand-rolled stages map directly onto this evolution:

```
Your 02/03/04 hand-rolled loop  →  Old LangChain AgentExecutor  →  LangGraph
(for loop + parser + messages)      (framework-wrapped version        (proper state
                                     of the same thing)                machine)
```

You didn't miss LangChain — you built its predecessor by hand, then arrived at its current form. This is a stronger foundation than starting with LangGraph directly, because you understand *what problem* the state machine is solving (the rigidity and fragility of a hand-rolled loop) rather than just using it as a black box.

**LangGraph key concepts (different from the hand-rolled loop):**

- **State** — instead of a growing `messages` list you manage manually, LangGraph maintains typed state that persists across nodes automatically.
- **Nodes** — discrete steps in the agent's execution (call model, execute tool, decide next step) expressed as functions in a graph rather than sequential lines in a `for` loop.
- **Edges** — conditional routing between nodes (e.g. "if the model returned a tool call, go to the tool execution node; if it returned a final answer, go to the end node").
- **Persistence** — LangGraph can checkpoint state between steps, enabling pause/resume, human-in-the-loop interruptions, and time-travel debugging — none of which are possible with a simple `for` loop.

**Practical implication:** `from langgraph.prebuilt import create_react_agent` is the correct modern import for building ReAct agents, not `from langchain.agents import create_react_agent` (deprecated). The `prebuilt` module provides pre-assembled graphs for common patterns like ReAct, so you don't have to define nodes and edges manually for standard use cases.

## The frameworks picture (zoomed out)

Frameworks like LangChain/LangGraph are battle-hardened versions of the exact files written in `02`-`04`, with years of production failures baked into defensive code:

| Failure encountered | Framework solution |
|---|---|
| Markdown-fence JSON wrapping | Output parsers with multiple format handlers |
| Hallucinated tool-skipping | Structured tool-calling enforced at API level |
| Multi-step planning in one response | Framework-managed execution loop |
| Output type drift | Typed output schemas with automatic coercion |
| Context window overflow from web content | Built-in text splitters and token counters |
| Stale data / date blindness | Configurable retrieval filters and metadata |
| Wrong topic from search | Query rewriting modules, retrieval evaluation |

Frameworks don't eliminate these failures — they reduce frequency and handle them more gracefully. Understanding the hand-rolled layer means you can identify which layer a failure comes from: prompt, parsing, tool execution, or model reasoning.

## A reflection on scalability

The hand-patching-every-failure approach doesn't scale — and it's not supposed to. What was built in `02`-`04` is the deliberately manual version, specifically so failure modes become visible and understood. The real scalable answers: native/structured tool-calling APIs for format reliability, larger models for reasoning reliability, frameworks for production plumbing.

## Vocabulary log

- **Token** — the unit of text a model actually processes (~a word or word-piece).
- **Context window** — max amount of text a model can see in one call; everything must fit inside it.
- **Inference** — running the model to generate output.
- **Observation** — the result of a tool call, fed back into context as a new message.
- **Hallucination** — confidently stated but unverified/false output.
- **Stateless** — no memory between calls except what's explicitly resent.
- **Structured output** — constraining a model's output to a fixed schema at the inference/API level.
- **Type drift** — when a model's output is syntactically valid but an unexpected data type.
- **Date laundering** — retrieving a real fact from a real source but stripping or misrepresenting its timestamp to make stale information sound current.
- **Topic disambiguation failure** — when a search returns results about a different topic than intended and the model answers from the wrong results without flagging the mismatch.
- **AgentExecutor** — LangChain's deprecated rigid hardcoded loop for running agents. Replaced by LangGraph.
- **LangGraph** — a state machine framework where agent logic is expressed as a graph of nodes and edges. Now the execution engine underneath LangChain 1.x.
- **State machine** — a system that moves between defined states based on conditions, more flexible than a hardcoded loop because branching, persistence, and interruption are first-class concepts.
- **Node (LangGraph)** — a discrete step in the agent graph (e.g. call model, execute tool).
- **Edge (LangGraph)** — conditional routing between nodes based on the current state.

## Open questions / things to verify later

- How many of the bugs found in `02`-`04` disappear in `05_framework_rewrite` with LangGraph?
- Debug log vs truncation: implement verbose flag (`--verbose`) for production agent pattern.
- Recency filtering in DuckDuckGo — worth adding to `web_search` to prevent date laundering at the tool level.
- How does topic disambiguation failure change with better query reformulation prompting?
- LangGraph deeper dive: nodes, edges, and custom state — worth a dedicated `06` exploration before bigger projects.
