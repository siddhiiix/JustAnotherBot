import sqlite3

from ai.conversation import DB_PATH


def initialize_chaos_settings():
    """Create the chaos settings table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS chaos_settings (
                user_id INTEGER PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 0
            )
        """)


def is_chaos_enabled(user_id: int) -> bool:
    """Return whether Chaos Mode is enabled for a user."""
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute("""
            SELECT enabled
            FROM chaos_settings
            WHERE user_id = ?
        """, (user_id,)).fetchone()

    return bool(row[0]) if row else False


def set_chaos_enabled(user_id: int, enabled: bool) -> None:
    """Enable or disable Chaos Mode for a user."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            INSERT INTO chaos_settings (user_id, enabled)
            VALUES (?, ?)
            ON CONFLICT(user_id)
            DO UPDATE SET enabled = excluded.enabled
        """, (user_id, int(enabled)))


initialize_chaos_settings()