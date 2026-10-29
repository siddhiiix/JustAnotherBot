import asyncio

from ai.rag.document_loader import DOCUMENTS_DIR
from ai.rag.ingest import ingest_all_documents
from ai.rag.rag_pipeline import ask_documents


async def main():
    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Ensure we have a sample document to test against.
    sample_file = DOCUMENTS_DIR / "sample_rag_notes.txt"

    if not sample_file.exists():
        sample_file.write_text(
            """
            Retrieval-Augmented Generation (RAG)

            RAG retrieves relevant information from external
            documents before a language model generates an answer.

            Embeddings represent text as numerical vectors.
            A vector database stores these vectors and retrieves
            text that is semantically similar to a question.

            ChromaDB is a vector database that supports similarity
            search over stored embeddings.

            Chunking divides long documents into smaller sections.
            Overlapping chunks help preserve context.

            Ollama can run language models and embedding models locally.
            """,
            encoding="utf-8",
        )

    print("Indexing documents...")
    stats = await ingest_all_documents()

    print(
        f"Indexed {stats['documents']} documents "
        f"and {stats['chunks']} chunks."
    )

    questions = [
        "What is Retrieval-Augmented Generation?",
        "What is the purpose of embeddings in RAG?",
        "What is the capital of Japan?",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")

        result = await ask_documents(question)

        print(f"\nANSWER:\n{result['answer']}")

        print("\nSOURCES:")

        if result["sources"]:
            for source in result["sources"]:
                print(
                    f"[{source['citation']}] "
                    f"{source['source']}"
                )
        else:
            print("No sources returned.")

        print(
            f"\nRetrieved chunks: "
            f"{result['retrieved_chunks']}"
        )


if __name__ == "__main__":
    asyncio.run(main())