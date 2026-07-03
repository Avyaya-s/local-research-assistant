from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from datetime import date
from tools import web_search, fetch_page, search_docs, save_summary

# ── Config ───────────────────────────────────────────────────────────────────

MODEL = "qwen2.5:14b"

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a research assistant with access to two knowledge sources:
1. A local document collection (your indexed notes, papers, and documents)
2. The web (for current information not in your local collection)

Today's date is {date.today()}.

Guidelines:
- Always search_docs FIRST before going to the web — prefer local knowledge when available.
- Use web_search when the question is about current events, recent data, or topics not covered locally.
- Use fetch_page only when search snippets are insufficient and you need full page content.
- Use save_summary when the user asks to save, export, or store results.
- Always cite your sources — mention whether information came from local docs or the web.
- If local docs and web conflict, mention both and note the discrepancy.
- Keep answers concise and grounded in what you actually retrieved.
"""

# ── Agent ─────────────────────────────────────────────────────────────────────

llm = ChatOllama(model=MODEL)
tools = [search_docs, web_search, fetch_page, save_summary]
agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

# ── Run ───────────────────────────────────────────────────────────────────────

def run_agent(user_task: str):
    print(f"\nResearching: {user_task}\n")
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_task}]
    })

    # print the reasoning steps
    for msg in result["messages"]:
        role = msg.type if hasattr(msg, "type") else msg["role"]
        content = msg.content if hasattr(msg, "content") else msg["content"]
        if role == "tool" and content:
            print(f"[retrieved]: {content[:300]}...")
        elif role == "ai" and content:
            print(f"\n[agent]: {content}")

if __name__ == "__main__":
    print("=== Local Research Assistant ===")
    print(f"Model: {MODEL}")
    print(f"Type 'quit' to exit\n")
    while True:
        task = input("You: ").strip()
        if task.lower() in ("quit", "exit", "q"):
            break
        if not task:
            continue
        run_agent(task)