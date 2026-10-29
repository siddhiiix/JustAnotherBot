import sqlite3

from ai.conversation import DB_PATH


MAX_MEMORIES_PER_USER = 50
MAX_MEMORY_LENGTH = 500
MAX_PROMPT_MEMORY_CHARACTERS = 2500


def initialize_memory_database():
    """Create long-term memory and automatic-memory settings tables."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_long_term_memories_user
            ON long_term_memories (user_id, id)
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS auto_memory_settings (
                user_id INTEGER PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 0
            )
        """)


def add_memory(user_id: int, content: str) -> int:
    """Save a memory, returning its ID. Exact duplicates are not re-added."""
    content = content.strip()

    if not content:
        raise ValueError("A memory cannot be empty.")

    if len(content) > MAX_MEMORY_LENGTH:
        raise ValueError(
            f"Memories cannot exceed {MAX_MEMORY_LENGTH} characters."
        )

    with sqlite3.connect(DB_PATH) as connection:
        existing = connection.execute("""
            SELECT id
            FROM long_term_memories
            WHERE user_id = ?
              AND LOWER(TRIM(content)) = LOWER(TRIM(?))
            LIMIT 1
        """, (user_id, content)).fetchone()

        if existing:
            return existing[0]

        row = connection.execute("""
            SELECT COUNT(*)
            FROM long_term_memories
            WHERE user_id = ?
        """, (user_id,)).fetchone()

        if row[0] >= MAX_MEMORIES_PER_USER:
            raise ValueError(
                f"You may store up to {MAX_MEMORIES_PER_USER} memories. "
                "Remove an old memory before adding another."
            )

        cursor = connection.execute("""
            INSERT INTO long_term_memories (user_id, content)
            VALUES (?, ?)
        """, (user_id, content))

        return cursor.lastrowid


def get_memories(user_id: int) -> list[dict]:
    """Retrieve all memories belonging to a user."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT id, content, created_at
            FROM long_term_memories
            WHERE user_id = ?
            ORDER BY id ASC
        """, (user_id,)).fetchall()

    return [dict(row) for row in rows]


def get_memories_for_prompt(
    user_id: int,
    max_characters: int = MAX_PROMPT_MEMORY_CHARACTERS,
) -> str:
    """Format recent memories for inclusion in the AI prompt."""
    memories = get_memories(user_id)

    if not memories:
        return ""

    selected = []
    used_characters = 0

    for memory in reversed(memories):
        line = f"- [Memory #{memory['id']}] {memory['content']}"

        if used_characters + len(line) + 1 > max_characters:
            if not selected:
                prefix = f"- [Memory #{memory['id']}] "
                available = max_characters - len(prefix)

                if available > 0:
                    selected.append(
                        prefix + memory["content"][:available]
                    )
            break

        selected.append(line)
        used_characters += len(line) + 1

    selected.reverse()
    return "\n".join(selected)


def delete_memory(user_id: int, memory_id: int) -> bool:
    """Delete a memory only when it belongs to the requesting user."""
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute("""
            DELETE FROM long_term_memories
            WHERE id = ? AND user_id = ?
        """, (memory_id, user_id))

        return cursor.rowcount > 0


def clear_all_memories(user_id: int) -> int:
    """Delete all saved memories belonging to a user."""
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute("""
            DELETE FROM long_term_memories
            WHERE user_id = ?
        """, (user_id,))

        return cursor.rowcount


def is_auto_memory_enabled(user_id: int) -> bool:
    """Return whether automatic memory extraction is enabled."""
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute("""
            SELECT enabled
            FROM auto_memory_settings
            WHERE user_id = ?
        """, (user_id,)).fetchone()

    return bool(row[0]) if row else False


def set_auto_memory_enabled(user_id: int, enabled: bool) -> None:
    """Save a user's automatic-memory preference."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            INSERT INTO auto_memory_settings (user_id, enabled)
            VALUES (?, ?)
            ON CONFLICT(user_id)
            DO UPDATE SET enabled = excluded.enabled
        """, (user_id, int(enabled)))


initialize_memory_database()