
import time
import statistics

from langchain_ollama import OllamaEmbeddings

from ai.rag.vector_store import search_chunks


# Use the same embedding model as the RAG pipeline.
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)

queries = [
    "What are the important events in Chapter 3?",
    "Explain the main characters.",
    "What happened at the end of the story?",
    "Summarise the document.",
    "What are the major themes?",
]

embedding_times = []
retrieval_times = []

print("=" * 60)
print("JUSTANOTHERBOT - RAG RETRIEVAL BENCHMARK")
print("=" * 60)

for index, query in enumerate(queries, start=1):

    try:
        # Measure embedding generation time.
        start = time.perf_counter()

        query_embedding = embeddings.embed_query(query)

        embedding_time = time.perf_counter() - start

        # Measure ChromaDB retrieval time.
        start = time.perf_counter()

        results = search_chunks(
            query_embedding=query_embedding,
            n_results=5,
        )

        retrieval_time = time.perf_counter() - start

        embedding_times.append(embedding_time)
        retrieval_times.append(retrieval_time)

        print(f"\nQuery {index}: {query}")
        print(f"Embedding time: {embedding_time:.4f} seconds")
        print(f"Retrieval time: {retrieval_time:.4f} seconds")
        print(f"Chunks retrieved: {len(results)}")

        for rank, result in enumerate(results, start=1):
            print(
                f"  {rank}. Source: {result['source']} | "
                f"Chapter: {result['chapter']} | "
                f"Page: {result['page']} | "
                f"Distance: {result['distance']:.4f}"
            )

    except Exception as exc:
        print(f"\nQuery failed: {query}")
        print(f"Error: {exc}")


print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)

if retrieval_times:
    print(f"Successful queries: {len(retrieval_times)}/{len(queries)}")
    print(
        f"Average embedding time: "
        f"{statistics.mean(embedding_times):.4f} seconds"
    )
    print(
        f"Average retrieval time: "
        f"{statistics.mean(retrieval_times):.4f} seconds"
    )
    print(
        f"Average total query time: "
        f"{statistics.mean([
            e + r for e, r in zip(embedding_times, retrieval_times)
        ]):.4f} seconds"
    )
else:
    print("No queries completed successfully.")
