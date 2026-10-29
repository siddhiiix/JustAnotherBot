import sqlite3
from ai.conversation import DB_PATH

with sqlite3.connect(DB_PATH) as connection:
    cursor = connection.execute(
        "DELETE FROM conversation_messages WHERE role = ?",
        ("assistant",),
    )
    print(f"Removed {cursor.rowcount} saved bot replies.")

print("User messages and remembered facts are preserved.")