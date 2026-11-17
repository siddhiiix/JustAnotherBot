
from ai.rag.vector_store import collection

SOURCE = "Frans_Kafka_the_trial.pdf"

data = collection.get(
    where={
        "$and": [
            {"source": SOURCE},
            {"chapter_number": 3},
        ]
    },
    include=["documents", "metadatas"],
)

records = []

for document, metadata in zip(
    data["documents"],
    data["metadatas"],
):
    records.append({
        "chunk": metadata.get("chunk_index", -1),
        "page": metadata.get("page", -1),
        "section": metadata.get("section", "Unknown"),
        "text": document or "",
    })

records.sort(key=lambda item: item["chunk"])

print(f"\nTotal Chapter 3 chunks: {len(records)}")

lengths = [len(item["text"]) for item in records]

if lengths:
    print(f"Shortest chunk: {min(lengths)} characters")
    print(f"Longest chunk: {max(lengths)} characters")
    print(f"Average length: {sum(lengths) // len(lengths)} characters")

print("\n========== LONGEST CHUNKS ==========")

for item in sorted(
    records,
    key=lambda x: len(x["text"]),
    reverse=True,
)[:5]:
    print(f"\nChunk: {item['chunk']}")
    print(f"Page: {item['page']}")
    print(f"Section: {item['section']}")
    print(f"Length: {len(item['text'])}")
    print(f"\n{item['text'][:1500]}")
    print("-" * 60)
