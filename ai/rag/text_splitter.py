
import re


CHAPTER_PATTERN = re.compile(
    r"(?m)^\s*##\s+"
    r"(?:Chapter|Chap\.?)\s+.+$",
    re.IGNORECASE,
)

HEADING_PATTERN = re.compile(
    r"(?m)^\s*(#{2,3})\s+(.+?)\s*$"
)

PAGE_PATTERN = re.compile(
    r"(?m)^\[Page\s+(\d+)\]\s*$",
    re.IGNORECASE,
)


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    """
    Split a document into overlapping chunks while preserving structure.

    Supports:
    - Chapter and section headings
    - PDF page markers
    - Documents without headings
    - Introductory material before the first heading

    Returns list[str] for compatibility with the existing ingestion pipeline.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    sections = _extract_sections(text)
    chunks = []

    for heading, page, section_text in sections:
        if not section_text.strip():
            continue

        prefix_parts = []

        if heading:
            prefix_parts.append(f"[Section: {heading}]")

        if page is not None:
            prefix_parts.append(f"[Page: {page}]")

        prefix = "\n".join(prefix_parts)

        # Reserve room for the metadata prefix.
        content_size = max(1, chunk_size - len(prefix) - 1)

        section_chunks = _split_section(
            section_text=section_text,
            chunk_size=content_size,
            chunk_overlap=min(chunk_overlap, content_size - 1)
            if content_size > 1
            else 0,
        )

        for content in section_chunks:
            chunk = f"{prefix}\n{content}" if prefix else content
            chunks.append(chunk)

    return chunks


def _extract_sections(text: str) -> list[tuple[str | None, int | None, str]]:
    """
    Split text at headings and page markers.

    A page marker updates the page metadata for subsequent content.
    Headings update the current section metadata.
    """

    sections = []

    current_heading = None
    current_page = None
    current_lines = []

    def flush():
        content = "\n".join(current_lines).strip()

        if content:
            sections.append(
                (current_heading, current_page, content)
            )

    for line in text.splitlines():
        stripped = line.strip()

        page_match = PAGE_PATTERN.fullmatch(stripped)

        if page_match:
            flush()
            current_lines.clear()
            current_page = int(page_match.group(1))
            continue

        heading_match = HEADING_PATTERN.fullmatch(stripped)

        if heading_match:
            flush()
            current_lines.clear()
            current_heading = heading_match.group(2).strip()
            current_lines.append(stripped)
            continue

        current_lines.append(line)

    flush()

    if not sections:
        return [(None, None, text.strip())]

    return sections


def _split_section(
    section_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split a section near natural text boundaries."""

    if len(section_text) <= chunk_size:
        return [section_text.strip()]

    chunks = []
    start = 0
    text_length = len(section_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            search_start = start + chunk_size // 2

            boundaries = [
                section_text.rfind("\n\n", search_start, end),
                section_text.rfind("\n", search_start, end),
                section_text.rfind(". ", search_start, end),
                section_text.rfind("? ", search_start, end),
                section_text.rfind("! ", search_start, end),
            ]

            boundary = max(boundaries)

            if boundary > start:
                end = boundary + 1

        content = section_text[start:end].strip()

        if content:
            chunks.append(content)

        if end >= text_length:
            break

        next_start = max(start + 1, end - chunk_overlap)

        # Prefer beginning the next chunk at a word boundary.
        whitespace = section_text.find(" ", next_start, end)

        if whitespace != -1:
            next_start = whitespace + 1

        start = next_start

    return chunks
