import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

MODEL = "qwen3:1.7b"
MAX_RESPONSE_TOKENS = 100
TEMPERATURE = 0.9
DISCORD_MESSAGE_LIMIT = 1900