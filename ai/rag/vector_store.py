"""ChromaDB storage and chapter-aware semantic search."""

import logging
import re
from pathlib import Path

import chromadb

logger = logging.getLogger("JustAnotherBot.RAG")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "justanotherbot_documents"

CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

CHAPTER_TAG = re.compile(
    r"^\[Chapter:\s*(.*?)\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)

NUMBER_TAG = re.compile(
    r"^\[ChapterNumber:\s*(-?\d+)\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)

PAGE_TAG = re.compile(
    r"^\[Page\s*:?\s*(\d+)\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)

SECTION_TAG = re.compile(
    r"^\[Section:\s*(.*?)\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)

CHAPTER_RE = re.compile(
    r"\b(?:chapter|chap\.?|part|book)\s+"
    r"(one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|first|second|third|fourth|fifth|sixth|"
    r"seventh|eighth|ninth|tenth|[IVXLCDM]+|\d+)\b",
    re.IGNORECASE,
)

WORDS = {
    "one": 1, "first": 1,
    "two": 2, "second": 2,
    "three": 3, "third": 3,
    "four": 4, "fourth": 4,
    "five": 5, "fifth": 5,
    "six": 6, "sixth": 6,
    "seven": 7, "seventh": 7,
    "eight": 8, "eighth": 8,
    "nine": 9, "ninth": 9,
    "ten": 10, "tenth": 10,
    "eleven": 11, "twelve": 12,
}

ROMANS = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4,
    "v": 5, "vi": 6, "vii": 7, "viii": 8,
    "ix": 9, "x": 10, "xi": 11, "xii": 12,
}


def _number(value):
    if value is None:
        return -1

    value = str(value).strip().lower()

    if value.isdigit():
        return int(value)

    return WORDS.get(
        value,
        ROMANS.get(value, -1),
    )


def extract_chapter(text: str) -> str:
    """Extract the chapter label from a chunk."""

    match = CHAPTER_TAG.search(text or "")

    if match:
        return match.group(1).strip() or "Unknown"

    for line in (text or "").splitlines():
        line = line.strip().lstrip("#").strip()
        match = CHAPTER_RE.search(line)

        if match:
            return line

    return "Unknown"


def extract_chapter_number(chapter: str) -> int:
    """Extract the chapter number from a chapter label."""

    match = NUMBER_TAG.search(chapter or "")

    if match:
        return int(match.group(1))

    match = CHAPTER_RE.search(chapter or "")

    if match:
        return _number(match.group(1))

    return -1


def extract_page(text: str) -> int:
    match = PAGE_TAG.search(text or "")

    return int(match.group(1)) if match else -1


def extract_section(text: str) -> str:
    match = SECTION_TAG.search(text or "")

    if match:
        return match.group(1).strip() or "Unknown"

    for line in (text or "").splitlines():
        if line.strip().startswith(("## ", "### ")):
            return line.strip().lstrip("#").strip()

    return "Unknown"


def delete_source(source: str) -> None:
    """Remove all indexed chunks belonging to one source document."""

    collection.delete(
        where={"source": source}
    )

    logger.info(
        "Removed previous indexed chunks for %s.",
        source,
    )


def add_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    source: str,
    metadatas: list[dict] | None = None,
) -> int:
    """Store chunks, embeddings, and searchable metadata."""

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

    ids = []
    prepared_metadata = []

    for index, chunk in enumerate(chunks):
        ids.append(f"{source}::chunk_{index}")

        metadata = (
            dict(metadatas[index])
            if metadatas
            else {}
        )

        chapter = metadata.get("chapter") or extract_chapter(chunk)

        number = metadata.get("chapter_number")

        if number is None or number == -1:
            tagged_number = NUMBER_TAG.search(chunk)

            if tagged_number:
                number = int(tagged_number.group(1))
            else:
                number = extract_chapter_number(chapter)

        page = metadata.get("page")

        if page is None or page == -1:
            page = extract_page(chunk)

        section = metadata.get("section") or extract_section(chunk)

        metadata.update(
            source=source,
            chunk_index=index,
            chapter=str(chapter or "Unknown"),
            chapter_number=int(
                number if number is not None else -1
            ),
            page=int(page if page is not None else -1),
            section=str(section or "Unknown"),
        )

        # ChromaDB metadata values must be simple scalar types.
        metadata = {
            key: value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool))
        }

        prepared_metadata.append(metadata)

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
    chapter_number: int | None = None,
    source: str | None = None,
    page: int | None = None,
    section: str | None = None,
) -> list[dict]:
    """Search chunks, optionally restricting the search by metadata."""

    if not query_embedding or collection.count() == 0:
        return []

    conditions = []

    if chapter:
        conditions.append({"chapter": chapter})

    if chapter_number is not None:
        conditions.append(
            {"chapter_number": int(chapter_number)}
        )

    if source:
        conditions.append({"source": source})

    if page is not None:
        conditions.append({"page": int(page)})

    if section:
        conditions.append({"section": section})

    if len(conditions) == 1:
        where = conditions[0]
    elif conditions:
        where = {"$and": conditions}
    else:
        where = None

    total_count = collection.count()

    if where:
        available = len(
            collection.get(
                where=where,
                include=[],
            )["ids"]
        )

        if available == 0:
            return []

        n_results = min(
            max(1, n_results),
            available,
        )
    else:
        n_results = min(
            max(1, n_results),
            total_count,
        )

    arguments = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where:
        arguments["where"] = where

    found = collection.query(**arguments)

    documents = (found.get("documents") or [[]])[0]
    metadata_list = (found.get("metadatas") or [[]])[0]
    distances = (found.get("distances") or [[]])[0]

    results = []

    for document, metadata, distance in zip(
        documents,
        metadata_list,
        distances,
    ):
        metadata = metadata or {}

        results.append({
            "text": document or "",
            "source": metadata.get("source", "Unknown"),
            "chunk_index": metadata.get("chunk_index", -1),
            "chapter": metadata.get("chapter", "Unknown"),
            "chapter_number": metadata.get("chapter_number", -1),
            "page": metadata.get("page", -1),
            "section": metadata.get("section", "Unknown"),
            "distance": (
                float(distance)
                if distance is not None
                else 1.0
            ),
        })

    return results


def search_chunks_multi(
    query_embeddings: list[list[float]],
    n_results_each: int = 4,
    chapter: str | None = None,
    chapter_number: int | None = None,
    source: str | None = None,
    page: int | None = None,
    section: str | None = None,
) -> list[dict]:
    """Search multiple queries and merge duplicate chunks."""

    merged = {}

    for embedding in query_embeddings or []:
        results = search_chunks(
            query_embedding=embedding,
            n_results=n_results_each,
            chapter=chapter,
            chapter_number=chapter_number,
            source=source,
            page=page,
            section=section,
        )

        for result in results:
            key = (
                result["source"],
                result["chunk_index"],
            )

            if (
                key not in merged
                or result["distance"] < merged[key]["distance"]
            ):
                merged[key] = result

    return sorted(
        merged.values(),
        key=lambda item: item["distance"],
    )


def get_collection_count() -> int:
    return collection.count()


def list_indexed_sources() -> list[dict]:
    """Return the number of indexed chunks for each source."""

    result = collection.get(
        include=["metadatas"]
    )

    counts = {}

    for metadata in result.get("metadatas") or []:
        if metadata:
            source = metadata.get("source", "Unknown")
            counts[source] = counts.get(source, 0) + 1

    return [
        {"source": source, "chunks": count}
        for source, count in sorted(counts.items())
    ]


def chapter_metadata_report() -> dict:
    """Return indexed chunk counts grouped by chapter number."""

    result = collection.get(
        include=["metadatas"]
    )

    counts = {}

    for metadata in result.get("metadatas") or []:
        if metadata:
            number = metadata.get("chapter_number", -1)

            label = (
                str(number)
                if number not in (-1, None)
                else "Unknown"
            )

            counts[label] = counts.get(label, 0) + 1

    return dict(sorted(counts.items()))


def get_chapter_distribution(
    source: str | None = None,
) -> dict:
    """Return chapter labels and chunk counts for ingestion checks."""

    if source:
        result = collection.get(
            where={"source": source},
            include=["metadatas"],
        )
    else:
        result = collection.get(
            include=["metadatas"]
        )

    counts = {}

    for metadata in result.get("metadatas") or []:
        metadata = metadata or {}

        number = metadata.get("chapter_number", -1)
        chapter = metadata.get("chapter", "Unknown")

        key = f"{chapter} (number={number})"
        counts[key] = counts.get(key, 0) + 1

    return dict(sorted(counts.items()))