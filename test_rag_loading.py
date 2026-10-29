from pathlib import Path

from ai.rag.document_loader import (
    DOCUMENTS_DIR,
    load_all_documents,
)
from ai.rag.text_splitter import split_text


def main():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create a sample document if the folder is empty.
    sample_file = DOCUMENTS_DIR / "sample_rag_notes.txt"

    if not any(
        path.is_file()
        and path.suffix.lower() in {".pdf", ".txt"}
        for path in DOCUMENTS_DIR.iterdir()
    ):
        sample_file.write_text(
            """
            Retrieval-Augmented Generation (RAG)

            RAG is a technique that retrieves relevant information
            from external documents before generating an answer.

            Embeddings represent text as numerical vectors.
            A vector database stores these vectors and helps retrieve
            text that is semantically similar to a user question.

            Chunking divides large documents into smaller sections.
            Overlapping chunks help preserve context between sections.
            """,
            encoding="utf-8",
        )

    documents = load_all_documents()

    if not documents:
        print("No documents were loaded.")
        return

    for document in documents:
        chunks = split_text(document["text"])

        print("\n" + "=" * 50)
        print(f"Document: {document['source']}")
        print(f"Characters: {len(document['text'])}")
        print(f"Chunks created: {len(chunks)}")

        for index, chunk in enumerate(chunks, start=1):
            print(f"\n--- Chunk {index} ---")
            print(chunk[:300])


if __name__ == "__main__":
    main()