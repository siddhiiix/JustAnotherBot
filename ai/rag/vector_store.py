
import logging
import re
from pathlib import Path

import chromadb

logger = logging.getLogger("JustAnotherBot.RAG")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "justanotherbot_documents"

CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

CHAPTER_PATTERN = re.compile(
    r"\b(?:chapter|chap\.?)\s+"
    r"(one|two|three|four|five|six|seven|eight|nine|ten|"
    r"[IVXLCDM]+|\d+)\b",
    re.IGNORECASE,
)

PAGE_PATTERN = re.compile(
    r"\[Page:\s*(\d+)\]|\[Page\s+(\d+)\]",
    re.IGNORECASE,
)

SECTION_PATTERN = re.compile(
    r"\[Section:\s*(.+?)\]",
    re.IGNORECASE,
)

LEGACY_CHAPTER_PATTERN = re.compile(
    r"^\[\s*(Chapter\s+.+?)\s*\]\s*\n",
    re.IGNORECASE,
)

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

ROMAN_VALUES = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
}


def extract_chapter(text: str) -> str:
    """Extract a chapter heading from a chunk, if available."""
    legacy_match = LEGACY_CHAPTER_PATTERN.match(text.strip())

    if legacy_match:
        return legacy_match.group(1).strip()

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("[Section:"):
            heading = line[len("[Section:"):].rstrip("]").strip()

            if CHAPTER_PATTERN.search(heading):
                return heading

        if line.startswith("## "):
            heading = line[3:].strip()

            if CHAPTER_PATTERN.search(heading):
                return heading

    return "Unknown"


def extract_chapter_number(chapter: str) -> int:
    """Extract a numeric chapter identifier, or return -1."""
    match = CHAPTER_PATTERN.search(chapter)

    if not match:
        return -1

    value = match.group(1).lower()

    if value.isdigit():
        return int(value)

    if value in NUMBER_WORDS:
        return NUMBER_WORDS[value]

    return ROMAN_VALUES.get(value, -1)


def extract_page(text: str) -> int:
    """Extract the page number embedded in a chunk, if available."""
    match = PAGE_PATTERN.search(text)

    if not match:
        return -1

    return int(match.group(1) or match.group(2))


def extract_section(text: str) -> str:
    """Extract the section heading embedded in a chunk."""
    match = SECTION_PATTERN.search(text)

    if match:
        return match.group(1).strip()

    # Support Markdown headings in older chunks.
    for line in text.splitlines():
        line = line.strip()

        if line.startswith("### "):
            return line[4:].strip()

    return "Unknown"


def delete_source(source: str) -> None:
    """Remove previously indexed chunks for a document."""
    collection.delete(where={"source": source})

    logger.info("Removed previous indexed chunks for %s.", source)


def add_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    source: str,
    metadatas: list[dict] | None = None,
) -> int:
    """
    Store chunks, embeddings, and structural metadata.

    The metadatas argument is optional to preserve compatibility
    with existing callers.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            "Each chunk must have exactly one embedding."
        )

    if metadatas is not None and len(chunks) != len(metadatas):
        raise ValueError(
            "Each chunk must have exactly one metadata record."
        )

    if not chunks:
        return 0

    ids = [
        f"{source}::chunk_{index}"
        for index in range(len(chunks))
    ]

    prepared_metadata = []

    for index, chunk in enumerate(chunks):
        metadata = dict(metadatas[index]) if metadatas else {}

        # Keep standard fields consistent across all chunks.
        metadata["source"] = source
        metadata["chunk_index"] = index
        metadata.setdefault("chapter", extract_chapter(chunk))
        metadata.setdefault(
            "chapter_number",
            extract_chapter_number(metadata["chapter"]),
        )
        metadata.setdefault("page", extract_page(chunk))
        metadata.setdefault("section", extract_section(chunk))

        # ChromaDB metadata values must be scalar values.
        prepared_metadata.append({
            key: value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool))
        })

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=prepared_metadata,
    )

    logger.info(
        "Indexed %s chunks from %s.",
        len(chunks),
        source,
    )

    return len(chunks)


def search_chunks(
    query_embedding: list[float],
    n_results: int = 5,
    chapter: str | None = None,
    source: str | None = None,
    page: int | None = None,
    section: str | None = None,
) -> list[dict]:
    """Retrieve semantically similar chunks with optional metadata filters."""
    if not query_embedding:
        return []

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    conditions = []

    if chapter:
        conditions.append({"chapter": chapter})

    if source:
        conditions.append({"source": source})

    if page is not None:
        conditions.append({"page": page})

    if section:
        conditions.append({"section": section})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}
    else:
        where_filter = None

    if where_filter:
        matching = collection.get(
            where=where_filter,
            include=[],
        )
        available_chunks = len(matching["ids"])
    else:
        available_chunks = total_chunks

    if available_chunks == 0:
        return []

    result_count = min(max(1, n_results), available_chunks)

    query_arguments = {
        "query_embeddings": [query_embedding],
        "n_results": result_count,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where_filter:
        query_arguments["where"] = where_filter

    results = collection.query(**query_arguments)

    matches = []

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    for document, metadata, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        metadata = metadata or {}

        matches.append({
            "text": document or "",
            "source": metadata.get("source", "Unknown"),
            "chunk_index": metadata.get("chunk_index", -1),
            "chapter": metadata.get("chapter", "Unknown"),
            "chapter_number": metadata.get("chapter_number", -1),
            "page": metadata.get("page", -1),
            "section": metadata.get("section", "Unknown"),
            "distance": (
                float(distance) if distance is not None else 1.0
            ),
        })

    return matches


def search_chunks_multi(
    query_embeddings: list[list[float]],
    n_results_each: int = 4,
    chapter: str | None = None,
    source: str | None = None,
    page: int | None = None,
    section: str | None = None,
) -> list[dict]:
    """Search using multiple embeddings and deduplicate results."""
    if not query_embeddings or collection.count() == 0:
        return []

    merged = {}

    for embedding in query_embeddings:
        results = search_chunks(
            query_embedding=embedding,
            n_results=n_results_each,
            chapter=chapter,
            source=source,
            page=page,
            section=section,
        )

        for result in results:
            key = (result["source"], result["chunk_index"])
            previous = merged.get(key)

            if (
                previous is None
                or result["distance"] < previous["distance"]
            ):
                merged[key] = result

    return sorted(
        merged.values(),
        key=lambda item: item["distance"],
    )


def get_collection_count() -> int:
    """Return the total number of indexed chunks."""
    return collection.count()


def list_indexed_sources() -> list[dict]:
    """Return indexed source names and their chunk counts."""
    result = collection.get(include=["metadatas"])

    counts = {}

    for metadata in result.get("metadatas") or []:
        if not metadata:
            continue

        source = metadata.get("source", "Unknown")
        counts[source] = counts.get(source, 0) + 1

    return [
        {"source": source, "chunks": count}
        for source, count in sorted(counts.items())
    ]
