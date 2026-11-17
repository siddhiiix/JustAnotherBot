
import asyncio

from ai.rag.rag_pipeline import (
    retrieve_relevant_chunks,
    embed_query,
    has_meaningful_content,
    TOP_K,
    RETRIEVAL_CANDIDATES,
    MMR_LAMBDA,
)

from ai.rag.vector_store import collection
from ai.rag.rag_pipeline import build_search_queries


QUESTION = "Who is Elsa and what is her relationship with K.?"


async def similarity_search():
    """Run ordinary similarity search with the same filters."""

    queries = build_search_queries(QUESTION)
    merged = {}

    for query in queries:
        embedding = await embed_query(query)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=RETRIEVAL_CANDIDATES,
            where={"chapter_number": 3},
            include=["documents", "metadatas", "distances"],
        )

        for i, document in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            result = {
                "text": document,
                "chunk_index": metadata.get("chunk_index"),
                "page": metadata.get("page"),
                "distance": distance,
            }

            if not _has_meaningful_content(document):
                continue

            key = result["chunk_index"]

            if key not in merged or distance < merged[key]["distance"]:
                merged[key] = result

    return sorted(
        merged.values(),
        key=lambda item: item["distance"]
    )[:TOP_K]


def print_results(title, results):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    print(f"Retrieved: {len(results)} chunks\n")

    for rank, result in enumerate(results, start=1):
        print("-" * 65)
        print(f"RANK: {rank}")
        print(f"Chunk: {result['chunk_index']}")
        print(f"Page: {result['page']}")
        print(f"Distance: {result['distance']:.4f}")
        print("\nCONTENT:")
        print(result["text"][:400])
        print()


async def main():
    print(f"\nQUESTION: {QUESTION}")

    similarity_results = await similarity_search()
    mmr_results = await retrieve_relevant_chunks(QUESTION)

    print_results("SIMILARITY SEARCH", similarity_results)
    print_results("MMR SEARCH", mmr_results)

    similarity_chunks = {
        result["chunk_index"] for result in similarity_results
    }

    mmr_chunks = {
        result["chunk_index"] for result in mmr_results
    }

    print("\n" + "=" * 75)
    print("COMPARISON")
    print("=" * 75)

    print("Similarity chunks:", similarity_chunks)
    print("MMR chunks:", mmr_chunks)
    print("Common chunks:", similarity_chunks & mmr_chunks)
    print("Similarity-only chunks:", similarity_chunks - mmr_chunks)
    print("MMR-only chunks:", mmr_chunks - similarity_chunks)


if __name__ == "__main__":
    asyncio.run(main())
