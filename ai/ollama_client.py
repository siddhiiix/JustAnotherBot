import logging
import time

import ollama

from config import MODEL, MAX_RESPONSE_TOKENS, TEMPERATURE

logger = logging.getLogger("JustAnotherBot")

OLLAMA_TIMEOUT_SECONDS = 120

ollama_client = ollama.AsyncClient(
    timeout=OLLAMA_TIMEOUT_SECONDS
)


async def generate_response(messages):
    """Generate an Ollama response with performance logging."""

    start_time = time.perf_counter()

    try:
        logger.info("Sending request to Ollama model: %s", MODEL)

        response = await ollama_client.chat(
            model=MODEL,
            messages=messages,
            options={
                "temperature": TEMPERATURE,
                "num_predict": min(MAX_RESPONSE_TOKENS, 180),
                "num_ctx": 2048,
            },
            keep_alive="10m",
        )

        elapsed_time = time.perf_counter() - start_time

        total_duration = getattr(response, "total_duration", 0) / 1e9
        load_duration = getattr(response, "load_duration", 0) / 1e9
        prompt_tokens = getattr(response, "prompt_eval_count", 0)
        generated_tokens = getattr(response, "eval_count", 0)
        done_reason = getattr(response, "done_reason", None)

        logger.info(
            "Ollama performance | elapsed=%.2fs | model_total=%.2fs | "
            "model_load=%.2fs | prompt_tokens=%s | generated_tokens=%s | "
            "done_reason=%s",
            elapsed_time,
            total_duration,
            load_duration,
            prompt_tokens,
            generated_tokens,
            done_reason,
        )

        answer = (response.message.content or "").strip()

        if not answer:
            logger.warning("Ollama returned an empty response.")
            return ""

        if done_reason == "length":
            logger.warning(
                "Response may be incomplete: reached the token limit."
            )

        return answer

    except Exception:
        logger.exception(
            "Ollama request failed after %.2f seconds.",
            time.perf_counter() - start_time,
        )
        raise