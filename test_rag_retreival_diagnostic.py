
import asyncio

from ai.rag.embeddings import embed_query
from ai.rag.vector_store import search_chunks


async def main():
    question = "What are the important events in Chapter 3?"

    embedding = await embed_query(question)

    results = search_chunks(
        query_embedding=embedding,
        n_results=30,
        chapter_number=3,
    )

    print(f"\nRetrieved {len(results)} candidate chunks.")

    for rank, result in enumerate(results, start=1):
        content = result.get("text", "")

        print("\n" + "=" * 65)
        print(f"RANK: {rank}")
        print(f"Distance: {result.get('distance')}")
        print(f"Chunk: {result.get('chunk_index')}")
        print(f"Page: {result.get('page')}")
        print(f"Length: {len(content)} characters")
        print("\nCONTENT:")
        print(content[:900])


if __name__ == "__main__":
    asyncio.run(main())
