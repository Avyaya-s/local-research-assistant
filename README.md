# Local Research Assistant

A local AI research assistant that searches the web and your own document collection to answer questions. Built with LangGraph, LangChain, Chroma, and local models via Ollama.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Pull required models: `ollama pull nomic-embed-text` and `ollama pull qwen2.5:14b`
3. Drop documents into `docs/`
4. Run `python ingest.py` to index your documents
5. Run `python agent.py` to start the assistant