from langchain_community.document_loaders import PyPDFLoader, TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os

# ── Config ─────────────────────────────────────────────────────────────────

DOCS_DIR = "docs"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "nomic-embed-text"

# ── Embedding model ─────────────────────────────────────────────────────────

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# ── Text splitter ───────────────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# ── Load and chunk documents ────────────────────────────────────────────────

def load_documents(docs_dir: str):
    documents = []
    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        print(f"Loading: {filename}")
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".txt"):
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

# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Ingestion started ===\n")

    # load
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("No documents found. Add .pdf or .txt files to the docs/ folder.")
        exit()
    print(f"\nTotal pages/documents loaded: {len(docs)}")

    # chunk
    chunks = splitter.split_documents(docs)
    print(f"Total chunks after splitting: {len(chunks)}")

    # embed and store
    print(f"\nEmbedding chunks using '{EMBEDDING_MODEL}' and storing in Chroma...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
    print(f"\n=== Ingestion complete ===")
    print(f"Vectorstore saved to: {VECTORSTORE_DIR}/")
    print(f"Total chunks indexed: {vectorstore._collection.count()}")