
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent

embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)

vector_store = Chroma(
    collection_name="justanotherbot_documents",
    persist_directory=str(PROJECT_ROOT / "chroma_db"),
    embedding_function=embeddings,
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 12,
        "lambda_mult": 0.5,
    },
)

question = "What are the important events in Chapter 3?"

documents = retriever.invoke(question)

print(f"\nMMR retrieved {len(documents)} documents.\n")

for index, document in enumerate(documents, start=1):
    print(f"--- Result {index} ---")
    print("Content:", document.page_content[:500])
    print("Metadata:", document.metadata)
    print()
