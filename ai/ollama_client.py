import logging
import ollama

logger = logging.getLogger("JustAnotherBot")

MODEL = "qwen3:1.7b"
MAX_HISTORY = 10
MAX_RESPONSE_TOKENS = 100
TEMPERATURE = 0.9

ollama_client = ollama.AsyncClient()

KEEP_ALIVE = "10m"

async def generate_response(messages):
    """Generate a response using Ollama."""
    try:
        response = await ollama_client.chat(
            model=MODEL,
            messages=messages,
            options={
                "temperature": TEMPERATURE,
                "num_predict": MAX_RESPONSE_TOKENS,
            },
            think=False,
            keep_alive=KEEP_ALIVE,
        )

        return (response.message.content or "").strip()

    except ollama.ResponseError:
        logger.exception("Ollama response error.")
        raise