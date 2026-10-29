import logging
import ollama

logger = logging.getLogger("JustAnotherBot.RAG")

EMBEDDING_MODEL = "embeddinggemma"
BATCH_SIZE = 32

ollama_client = ollama.AsyncClient()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert multiple text strings into embedding vectors."""

    if not texts:
        return []

    embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        response = await ollama_client.embed(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        batch_embeddings = response.embeddings

        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                "Ollama returned an unexpected number of embeddings."
            )

        embeddings.extend(batch_embeddings)

        logger.info(
            "Generated embeddings for %s/%s text chunks.",
            len(embeddings),
            len(texts),
        )

    return embeddings


async def embed_query(text: str) -> list[float]:
    """Generate an embedding for a single search query."""

    if not text.strip():
        raise ValueError("Search query cannot be empty.")

    embeddings = await embed_texts([text])

    return embeddings[0]