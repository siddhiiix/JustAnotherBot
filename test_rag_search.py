import asyncio

from ai.rag.document_loader import DOCUMENTS_DIR
from ai.rag.ingest import ingest_all_documents
from ai.rag.embeddings import embed_query
from ai.rag.vector_store import (
    search_chunks,
    get_collection_count,
)


async def main():
    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Add a sample document only if no PDF/TXT documents exist.
    supported_documents = [
        path
        for path in DOCUMENTS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".pdf", ".txt"}
    ]

    if not supported_documents:
        sample_file = (
            DOCUMENTS_DIR / "sample_rag_notes.txt"
        )

        sample_file.write_text(
            """
            Retrieval-Augmented Generation (RAG)

            RAG retrieves relevant information from external
            documents before a language model generates an answer.

            Embeddings represent text as numerical vectors.
            Similar meanings can produce nearby vectors in
            the embedding space.

            ChromaDB is a vector database that stores embeddings
            and supports similarity search.

            Text chunking divides long documents into smaller
            sections. Overlapping chunks help preserve context.

            Ollama can run language models and embedding models
            locally on a computer.
            """,
            encoding="utf-8",
        )

        print("Created sample_rag_notes.txt")

    print("\nIndexing documents...")

    stats = await ingest_all_documents()

    print("\nIngestion results:")
    print(f"Documents indexed: {stats['documents']}")
    print(f"Chunks indexed: {stats['chunks']}")
    print(f"Failed documents: {stats['failed']}")
    print(f"Total chunks in database: {get_collection_count()}")

    if get_collection_count() == 0:
        print("\nNo chunks are available to search.")
        return

    query = "How does RAG use embeddings to find relevant information?"

    print(f"\nSearch query: {query}")

    query_embedding = await embed_query(query)

    results = search_chunks(
        query_embedding=query_embedding,
        n_results=3,
    )

    if not results:
        print("\nNo matching chunks found.")
        return

    print("\nRetrieved results:")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {result['source']}")
        print(f"Chunk: {result['chunk_index']}")
        print(f"Cosine distance: {result['distance']:.4f}")
        print(result["text"][:500])


if __name__ == "__main__":
    asyncio.run(main())