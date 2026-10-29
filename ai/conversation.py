import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "bot_memory.db"

# Number of recent messages sent to the AI on each request.
MAX_HISTORY = 10


def initialize_database():
    """Create the conversation and summary tables."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_lookup
            ON conversation_messages (
                guild_id, channel_id, user_id, id
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                guild_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                last_message_id INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (guild_id, channel_id, user_id)
            )
        """)


def get_conversation(key, limit=MAX_HISTORY):
    """Return recent messages in chronological order."""
    guild_id, channel_id, user_id = key

    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute("""
            SELECT role, content
            FROM conversation_messages
            WHERE guild_id = ?
              AND channel_id = ?
              AND user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (
            guild_id, channel_id, user_id, limit
        )).fetchall()

    rows.reverse()

    return [
        {"role": role, "content": content}
        for role, content in rows
    ]


def save_message(key, role, content):
    """Save a message without deleting older history prematurely."""
    guild_id, channel_id, user_id = key

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            INSERT INTO conversation_messages (
                guild_id, channel_id, user_id, role, content
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            guild_id, channel_id, user_id, role, content
        ))


def get_conversation_summary(key):
    """Return the saved summary for this user's channel conversation."""
    guild_id, channel_id, user_id = key

    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute("""
            SELECT summary
            FROM conversation_summaries
            WHERE guild_id = ?
              AND channel_id = ?
              AND user_id = ?
        """, (
            guild_id, channel_id, user_id
        )).fetchone()

    return row[0] if row else ""


def get_messages_to_summarize(key, keep_recent=MAX_HISTORY):
    """Get unsummarised messages older than the recent-message window."""
    guild_id, channel_id, user_id = key

    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute("""
            SELECT last_message_id
            FROM conversation_summaries
            WHERE guild_id = ?
              AND channel_id = ?
              AND user_id = ?
        """, (
            guild_id, channel_id, user_id
        )).fetchone()

        last_message_id = row[0] if row else 0

        rows = connection.execute("""
            SELECT id, role, content
            FROM conversation_messages
            WHERE guild_id = ?
              AND channel_id = ?
              AND user_id = ?
              AND id > ?
              AND id NOT IN (
                  SELECT id
                  FROM conversation_messages
                  WHERE guild_id = ?
                    AND channel_id = ?
                    AND user_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            ORDER BY id ASC
        """, (
            guild_id,
            channel_id,
            user_id,
            last_message_id,
            guild_id,
            channel_id,
            user_id,
            keep_recent,
        )).fetchall()

    return [
        {"id": row[0], "role": row[1], "content": row[2]}
        for row in rows
    ]


def save_conversation_summary(key, summary, last_message_id):
    """
    Save the updated summary and remove messages it now represents.

    This operation is performed in one database transaction.
    """
    guild_id, channel_id, user_id = key

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            INSERT INTO conversation_summaries (
                guild_id,
                channel_id,
                user_id,
                summary,
                last_message_id,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(guild_id, channel_id, user_id)
            DO UPDATE SET
                summary = excluded.summary,
                last_message_id = excluded.last_message_id,
                updated_at = CURRENT_TIMESTAMP
        """, (
            guild_id,
            channel_id,
            user_id,
            summary,
            last_message_id,
        ))

        connection.execute("""
            DELETE FROM conversation_messages
            WHERE guild_id = ?
              AND channel_id = ?
              AND user_id = ?
              AND id <= ?
        """, (
            guild_id,
            channel_id,
            user_id,
            last_message_id,
        ))


def clear_conversation(key):
    """Clear one user's history and summary in the specified channel."""
    guild_id, channel_id, user_id = key

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute("""
            DELETE FROM conversation_messages
            WHERE guild_id = ?
              AND channel_id = ?
              AND user_id = ?
        """, (
            guild_id, channel_id, user_id
        ))

        deleted_count = cursor.rowcount

        connection.execute("""
            DELETE FROM conversation_summaries
            WHERE guild_id = ?
              AND channel_id = ?
              AND user_id = ?
        """, (
            guild_id, channel_id, user_id
        ))

    return deleted_count


initialize_database()