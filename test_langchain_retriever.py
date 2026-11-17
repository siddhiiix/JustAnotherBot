
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Existing project configuration
PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "justanotherbot_documents"

# Connect to your existing embedding model
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)

# Connect to your EXISTING ChromaDB collection
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_DB_DIR),
    embedding_function=embeddings,
)

# Create a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

# Test retrieval
question = "What are the important events in Chapter 3?"

print("\nSearching existing ChromaDB collection...\n")

try:
    documents = retriever.invoke(question)

    print(f"Retrieved {len(documents)} documents.\n")

    for index, document in enumerate(documents, start=1):
        print(f"--- Result {index} ---")
        print("Content:", document.page_content[:500])
        print("Metadata:", document.metadata)
        print()

except Exception as error:
    print("Retrieval failed:", error)
