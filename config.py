import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

MODEL = "qwen2.5:3b"
MAX_RESPONSE_TOKENS = 500
TEMPERATURE = 0.8
DISCORD_MESSAGE_LIMIT = 1900