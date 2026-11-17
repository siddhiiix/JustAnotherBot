
import asyncio

from ai.rag.rag_pipeline import ask_documents


async def main():
    question = "What are the important events in chapter 3?"

    print("\nQuestion:", question)
    print("\nGenerating answer...\n")

    result = await ask_documents(question)

    print("=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)

    print(result.get("answer", "No answer generated."))

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in result.get("sources", []):
        print(
            f"[{source.get('citation')}] "
            f"{source.get('source')}"
        )

    print("\nRetrieved chunks:", result.get("retrieved_chunks"))
    print("Distances:", result.get("distances"))


if __name__ == "__main__":
    asyncio.run(main())
