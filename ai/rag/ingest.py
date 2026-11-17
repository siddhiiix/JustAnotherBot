"""Document ingestion with metadata verification."""

import logging

from ai.rag.document_loader import load_all_documents
from ai.rag.text_splitter import split_text
from ai.rag.embeddings import embed_texts
from ai.rag.vector_store import (
    delete_source,
    add_chunks,
    get_chapter_distribution,
)

logger = logging.getLogger("JustAnotherBot.RAG")


async def ingest_all_documents() -> dict:
    """Load, chunk, embed, index, and verify documents."""

    documents = load_all_documents()

    if not documents:
        logger.warning(
            "No supported documents found in the documents folder."
        )

        return {
            "documents": 0,
            "chunks": 0,
            "failed": 0,
            "chapter_metadata": {},
        }

    total_chunks = 0
    successful_documents = 0
    failed_documents = 0
    chapter_metadata = {}

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
                failed_documents += 1
                continue

            logger.info(
                "Preparing %s: %s chunks.",
                source,
                len(chunks),
            )

            embeddings = await embed_texts(chunks)

            if len(embeddings) != len(chunks):
                raise ValueError(
                    f"Embedding count mismatch for {source}: "
                    f"{len(chunks)} chunks, "
                    f"{len(embeddings)} embeddings."
                )

            # Keep the previous index until chunking and embedding
            # have succeeded.
            delete_source(source)

            indexed_count = add_chunks(
                chunks=chunks,
                embeddings=embeddings,
                source=source,
            )

            distribution = get_chapter_distribution(source)
            chapter_metadata[source] = distribution

            logger.info(
                "INGEST CHECK | %s | indexed=%s | "
                "chapter metadata=%s",
                source,
                indexed_count,
                distribution,
            )

            if not distribution or all(
                "number=-1" in key for key in distribution
            ):
                logger.warning(
                    "Chapter numbers were not detected for %s. "
                    "Chapter-specific filtering may fail.",
                    source,
                )

            total_chunks += indexed_count
            successful_documents += 1

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
        "chapter_metadata": chapter_metadata,
    }

    logger.info(
        "RAG ingestion complete: documents=%s chunks=%s failures=%s",
        result["documents"],
        result["chunks"],
        result["failed"],
    )

    return result