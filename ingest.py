from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os

# ── Config ───────────────────────────────────────────────────────────────────

DOCS_DIR = "docs"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "nomic-embed-text"

# ── Setup ────────────────────────────────────────────────────────────────────

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# ── Loaders ──────────────────────────────────────────────────────────────────

def load_documents(docs_dir: str):
    documents = []
    files = os.listdir(docs_dir)
    if not files:
        print("No files found in docs/ folder.")
        return documents

    for filename in files:
        filepath = os.path.join(docs_dir, filename)
        print(f"Loading: {filename}")
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".txt") or filename.endswith(".md"):
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                print(f"  Skipping unsupported format: {filename}")
                continue
            docs = loader.load()
            documents.extend(docs)
            print(f"  Loaded {len(docs)} page(s)")
        except Exception as e:
            print(f"  ERROR loading {filename}: {e}")

    return documents

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Ingestion started ===\n")

    docs = load_documents(DOCS_DIR)
    if not docs:
        print("No documents loaded. Add .pdf, .txt, or .md files to docs/ and rerun.")
        exit()

    print(f"\nTotal pages loaded: {len(docs)}")

    chunks = splitter.split_documents(docs)
    print(f"Total chunks after splitting: {len(chunks)}")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

    print(f"\nEmbedding and storing in Chroma using '{EMBEDDING_MODEL}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )

    print(f"\n=== Ingestion complete ===")
    print(f"Vectorstore saved to: {VECTORSTORE_DIR}/")
    print(f"Total chunks indexed: {vectorstore._collection.count()}")
    print(f"\nYou can now run agent.py to query your documents.")