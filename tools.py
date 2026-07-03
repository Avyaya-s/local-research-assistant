from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import os
from datetime import date

# ── Config ──────────────────────────────────────────────────────────────────

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "nomic-embed-text"
OUTPUTS_DIR = "outputs"

# ── Shared vectorstore connection ───────────────────────────────────────────

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

def get_vectorstore():
    if not os.path.exists(VECTORSTORE_DIR):
        return None
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
    )

# ── Tools ───────────────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Searches the web for current information. Use this for recent events,
    current prices, latest news, or anything not likely to be in local documents."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        output = []
        for i, r in enumerate(results):
            output.append(f"[{i+1}] {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n")
        return "\n".join(output)
    except Exception as e:
        return f"ERROR: web search failed: {e}"

@tool
def fetch_page(url: str) -> str:
    """Fetches and returns the cleaned text content of a webpage.
    Use this when web_search snippets are not enough and you need full page content."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.splitlines() if len(l.strip()) > 40]
        cleaned = "\n".join(lines[:100])
        return cleaned if cleaned else "Could not extract useful content from this page."
    except Exception as e:
        return f"ERROR: could not fetch page: {e}"

@tool
def search_docs(query: str) -> str:
    """Searches your local document collection for relevant information.
    Use this for questions about documents you have indexed — notes, papers, reports.
    Prefer this over web_search when the answer is likely in your local knowledge base."""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return "No local documents indexed yet. Run ingest.py first to index your documents."
    try:
        results = vectorstore.similarity_search(query, k=3)
        if not results:
            return "No relevant documents found in local collection."
        output = []
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "unknown")
            output.append(f"[{i+1}] Source: {source}\n{doc.page_content}\n")
        return "\n".join(output)
    except Exception as e:
        return f"ERROR: document search failed: {e}"

@tool
def save_summary(filename: str, content: str) -> str:
    """Saves research output or a summary to a markdown file in the outputs folder.
    Use this when the user asks to save, export, or store research results."""
    try:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        if not filename.endswith(".md"):
            filename = filename + ".md"
        filepath = os.path.join(OUTPUTS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Research Summary\n*Generated: {date.today()}*\n\n")
            f.write(content)
        return f"Summary saved to '{filepath}'."
    except Exception as e:
        return f"ERROR: could not save summary: {e}"