<<<<<<< HEAD
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
=======

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
>>>>>>> origin/main
