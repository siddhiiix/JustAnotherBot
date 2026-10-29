
import logging

from ai.rag.document_loader import load_all_documents
from ai.rag.text_splitter import split_text
from ai.rag.embeddings import embed_texts
from ai.rag.vector_store import delete_source, add_chunks

logger = logging.getLogger("JustAnotherBot.RAG")


async def ingest_all_documents() -> dict:
    """Load, chunk, embed, and index all supported documents."""

    documents = load_all_documents()

    if not documents:
        logger.warning("No supported documents found.")
        return {
            "documents": 0,
            "chunks": 0,
            "failed": 0,
        }

    total_chunks = 0
    successful_documents = 0
    failed_documents = 0

    for document in documents:
        source = document["source"]
        text = document["text"]

        try:
            chunks = split_text(text)

            if not chunks:
                logger.warning(
                    "No text chunks produced for %s.",
                    source,
                )
                continue

            # Generate embeddings before replacing the old index.
            embeddings = await embed_texts(chunks)

            if len(embeddings) != len(chunks):
                raise ValueError(
                    f"Embedding count mismatch for {source}: "
                    f"{len(chunks)} chunks, {len(embeddings)} embeddings."
                )

            # Replace this document's existing indexed chunks.
            delete_source(source)

            indexed_count = add_chunks(
                chunks=chunks,
                embeddings=embeddings,
                source=source,
            )

            total_chunks += indexed_count
            successful_documents += 1

            logger.info(
                "Successfully indexed %s: %s chunks.",
                source,
                indexed_count,
            )

        except Exception:
            failed_documents += 1

            logger.exception(
                "Failed to index document: %s",
                source,
            )

    result = {
        "documents": successful_documents,
        "chunks": total_chunks,
        "failed": failed_documents,
    }

    logger.info(
        "RAG ingestion complete: %s documents, %s chunks, %s failures.",
        result["documents"],
        result["chunks"],
        result["failed"],
    )

    return result
