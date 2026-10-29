import sqlite3

from ai.conversation import DB_PATH


# Available personality modes.
PERSONALITIES = {
    "victorian": "Distinguished, eloquent, witty, and balanced.",
    "chaotic": "Mischievous, unpredictable, playful, and delightfully absurd.",
    "wholesome": "Warm, compassionate, encouraging, and reassuring.",
    "roaster": "Sharp-witted, sarcastic, teasing, and elegantly insulting.",
    "nerd": "Analytical, curious, precise, and enthusiastic about technical subjects.",
}

DEFAULT_PERSONALITY = "victorian"


def initialize_personality_settings():
    """Create the personality settings table without affecting chat history."""

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS personality_settings (
                user_id INTEGER PRIMARY KEY,
                personality TEXT NOT NULL DEFAULT 'victorian'
            )
        """)


def get_personality(user_id: int) -> str:
    """Return the user's saved personality, or Victorian by default."""

    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute("""
            SELECT personality
            FROM personality_settings
            WHERE user_id = ?
        """, (user_id,)).fetchone()

    if row and row[0] in PERSONALITIES:
        return row[0]

    return DEFAULT_PERSONALITY


def set_personality(user_id: int, personality: str) -> None:
    """Save or update a user's personality preference."""

    if personality not in PERSONALITIES:
        raise ValueError(f"Unknown personality: {personality}")

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            INSERT INTO personality_settings (user_id, personality)
            VALUES (?, ?)
            ON CONFLICT(user_id)
            DO UPDATE SET personality = excluded.personality
        """, (user_id, personality))


# Ensure the settings table exists when this module is imported.
initialize_personality_settings()