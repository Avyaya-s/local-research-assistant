# Agentic AI — Basics

This document explains agentic AI from zero assumptions. Read this before anything else in the repo.

## 1. What a language model actually is

A model like phi4 (or GPT-4, or Claude) is a function: text goes in, text comes out. That's it.

```
input text  ->  [ MODEL ]  ->  output text
```

It does not "remember" previous conversations on its own. It does not have access to your files, the internet, or a calculator. Every single time you call it, it only knows what's inside the text you send it *right now*. If you want it to "remember" something from five minutes ago, you have to literally paste that old text back into the new input. That's the whole trick behind every "memory" feature you'll ever see in agent frameworks — it's not real memory, it's re-sending history.

## 2. What "chat" actually is

A chat interface (ChatGPT, Claude.ai, your `01_chatbot_basics` script) is just this, wrapped in a loop:

1. You type a message.
2. Your message gets added to a growing list of all messages so far (the "conversation history").
3. The *entire* list is sent to the model.
4. The model generates a reply based on the entire list.
5. The reply gets added to the list too.
6. Repeat.

So "chat" is just: keep a list, keep appending to it, keep resending the whole thing. There's no special chat capability inside the model itself — it's just trained to expect text formatted as a back-and-forth conversation.

## 3. What a "tool" is

A tool is a normal function in your own code — read a file, search the web, run a calculation, send an email. The model **cannot run these itself**. It has no hands. It can only produce text.

So how does a model "use" a tool? Purely through an agreement made in plain English (or JSON):

- You tell the model, in the system prompt: "If you want to read a file, respond with this exact format: `{"action": "read_file", "path": "..."}`"
- The model, instead of answering normally, outputs that exact text.
- **Your Python code** notices that pattern, and *your code* actually opens the file — not the model.
- Your code takes the result and sends it back to the model as a new message.
- The model now has the file's contents in its context, and can use them to write a real answer.

The model is the *decision-maker* (deciding which tool to use and when), but your code is the *actuator* (actually doing it). This separation is the single most important idea in this entire repo.

## 4. What "agentic" means

A plain chatbot only ever does step 2-6 of section 2 above: text in, text out, repeat.

An **agent** adds one more capability: at any point, instead of replying with an answer, the model can reply with a request to use a tool. Your code executes that request, feeds the result back in, and the model continues reasoning — possibly requesting more tools — until it decides it has enough information to give a final answer.

This pattern — think, act, observe, think again — is called **ReAct** (Reasoning + Acting), from a 2022 paper of the same name. It is the foundational pattern underneath nearly every "AI agent" product or framework you'll encounter.

```
[ Think ] -> [ Act (call a tool) ] -> [ Observe (get the result) ] -> [ Think again ] -> ... -> [ Final answer ]
```

That loop, hand-written in plain Python with no libraries, is exactly what you're building in `02_tool_calling`.

## 5. The four layers of the agentic AI landscape

It helps to picture everything in this space as four stacked layers:

**Layer 1 — The model.**
Phi4, GPT-4, Claude, Llama, etc. Just a text-in/text-out function. No memory, no tools, no agency by itself.

**Layer 2 — The agent loop.**
The ReAct pattern: model proposes an action in structured text, your code executes it, result gets fed back, repeat until done. This is what "agent" technically means. You are hand-building this layer first, on purpose, so you understand it before anything hides it from you.

**Layer 3 — Frameworks.**
LangChain, LlamaIndex, CrewAI, etc. These don't introduce new ideas — they pre-package layer 2 so you don't have to write your own JSON parser, retry logic, or tool registry every time. Useful for production speed, but they hide the mechanics unless you already understand what's underneath (which is exactly why you're doing this hand-rolled phase first).

**Layer 4 — Patterns & finished products built on top.**
- **RAG** (Retrieval-Augmented Generation): give the model a "search my documents" tool instead of a "read this one file" tool — same loop, smarter tool.
- **Multi-agent systems**: multiple agent loops, each with its own role, talking to each other.
- **MCP** (Model Context Protocol): a standardized way to describe tools to a model, so every developer doesn't have to invent their own system-prompt format like we just did by hand.
- **Finished products**: Claude Code, Cursor, AutoGPT — polished UIs wrapped around a layer 2/3 loop with many tools (file edit, terminal, git, browser, etc).

## 6. Key vocabulary

| Term | Meaning |
|---|---|
| **Prompt** | The text you send into the model. |
| **System prompt** | A special instruction block sent before the user's message, used to set rules/behavior — e.g. "you may call these tools in this format." |
| **Context window** | The maximum amount of text (measured in tokens) a model can "see" at once. Everything — system prompt, full history, tool results — has to fit inside this. |
| **Token** | A chunk of text (roughly a word or word-piece) — the unit models actually process. |
| **Inference** | The act of running the model to generate output. |
| **Tool / function calling** | The mechanism by which a model requests an external action be performed on its behalf. |
| **Observation** | The result of a tool call, fed back into the model's context. |
| **Agent loop / ReAct** | The think → act → observe → think cycle. |
| **Hallucination** | The model stating something false (e.g. claiming a file doesn't exist) confidently, without actually checking — a key risk when tool use isn't enforced. |
| **Stateless** | The model has no memory of past calls unless you re-send the history yourself. |
| **Framework** | A library (LangChain, etc.) that pre-builds the agent loop, tool registry, and other plumbing. |

## 7. The progression this repo follows

| Folder | What it adds |
|---|---|
| `01_chatbot_basics` | Plain text in/out loop. No tools. Establishes the baseline: a model with nothing but conversation history. |
| `02_tool_calling` | Adds ONE tool (`read_file`) and the full hand-written ReAct loop: parse model output, execute action, feed back observation. |
| `03_multi_tool_agent` | Multiple tools — the model now has to *choose* which tool fits the task, not just use the one it's given. |
| `04_web_research_agent` | A tool that reaches outside the local machine (web search/fetch) — same loop, different I/O surface. |
| `05_framework_rewrite` | Rebuild the same agent using a real framework (e.g. LangChain), to see exactly what the framework does for you that you were doing by hand. |

Read this file first, then `01_chatbot_basics.md`, then `02_tool_calling.md`, in order.
