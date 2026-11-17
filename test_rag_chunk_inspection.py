
from ai.rag.rag_pipeline import retrieve_relevant_chunks

async def main():
    results = await retrieve_relevant_chunks(
        "What are the important events in Chapter 3?"
    )

    for i, result in enumerate(results, start=1):
        print(f"\n{'=' * 60}")
        print(f"RESULT {i}")
        print(f"Source: {result.get('source')}")
        print(f"Chapter: {result.get('chapter')}")
        print(f"Section: {result.get('section')}")
        print(f"Page: {result.get('page')}")
        print(f"Chunk index: {result.get('chunk_index')}")
        print(f"\nCONTENT:\n{result.get('text')}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
