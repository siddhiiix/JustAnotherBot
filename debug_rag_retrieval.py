"""Inspect RAG retrieval without generating an Ollama answer."""
import asyncio

from ai.rag.rag_pipeline import (
    retrieve_relevant_chunks,
    should_use_coverage,
    extract_chapter_number,
)

QUESTIONS = [
    "Who is Elsa and what is her relationship with K.?",
    "What are the important events in chapter 3?",
]

async def main():
    for question in QUESTIONS:
        chapter = extract_chapter_number(question)
        strategy = "coverage" if should_use_coverage(question) else "similarity"

        print("\n" + "=" * 90)
        print(f"QUESTION: {question}")
        print(f"SELECTED STRATEGY: {strategy}")
        print(f"CHAPTER FILTER: {chapter}")
        print("=" * 90)

        results = await retrieve_relevant_chunks(question, method="auto")
        print(f"Retrieved chunks: {len(results)}")

        for index, item in enumerate(results, start=1):
            print("\n" + "-" * 90)
            print(
                f"RESULT {index} | chunk={item.get('chunk_index')} | "
                f"page={item.get('page')} | chapter={item.get('chapter')} | "
                f"chapter_number={item.get('chapter_number')} | "
                f"distance={item.get('distance')}"
            )
            print(f"SOURCE: {item.get('source')}")
            print("\nTEXT:\n")
            print(item.get("text", ""))

    print("\nRetrieval inspection complete. No LLM answer was generated.")

if __name__ == "__main__":
    asyncio.run(main())
