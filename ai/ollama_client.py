
import logging
import time
from functools import lru_cache

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_ollama import ChatOllama

from config import MODEL, MAX_RESPONSE_TOKENS, TEMPERATURE

logger = logging.getLogger("JustAnotherBot")

OLLAMA_TIMEOUT_SECONDS = 120
DEFAULT_MAX_TOKENS = 180
CONTEXT_WINDOW = 2048


def convert_messages(messages):
    """Convert existing message dictionaries into LangChain messages."""
    converted = []

    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content", ""))

        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))

    return converted


@lru_cache(maxsize=8)
def get_chat_model(temperature, max_tokens):
    """Create and reuse a configured LangChain Ollama model."""
    return ChatOllama(
        model=MODEL,
        temperature=temperature,
        num_predict=max_tokens,
        num_ctx=CONTEXT_WINDOW,
        keep_alive="10m",
        client_kwargs={
            "timeout": OLLAMA_TIMEOUT_SECONDS,
        },
    )


async def generate_response(messages, max_tokens=None, temperature=None):
    """
    Generate an asynchronous response through LangChain ChatOllama.

    Keeps the existing generate_response() interface so the main bot
    and RAG pipeline do not need to be rewritten.
    """
    start_time = time.perf_counter()

    if max_tokens is None:
        token_limit = min(MAX_RESPONSE_TOKENS, DEFAULT_MAX_TOKENS)
    else:
        token_limit = min(max_tokens, MAX_RESPONSE_TOKENS)

    selected_temperature = (
        TEMPERATURE if temperature is None else temperature
    )

    try:
        chat_model = get_chat_model(
            selected_temperature,
            token_limit,
        )

        langchain_messages = convert_messages(messages)

        logger.info(
            "Generating response through LangChain | model=%s | "
            "messages=%s | max_tokens=%s",
            MODEL,
            len(langchain_messages),
            token_limit,
        )

        response = await chat_model.ainvoke(langchain_messages)

        answer = response.content

        if isinstance(answer, list):
            answer = "".join(
                block.get("text", "")
                if isinstance(block, dict)
                else str(block)
                for block in answer
            )

        answer = str(answer).strip()

        metadata = getattr(response, "response_metadata", {}) or {}
        usage = getattr(response, "usage_metadata", None) or {}

        logger.info(
            "LangChain response completed | elapsed=%.2fs | "
            "input_tokens=%s | output_tokens=%s | done_reason=%s",
            time.perf_counter() - start_time,
            usage.get("input_tokens", "unknown"),
            usage.get("output_tokens", "unknown"),
            metadata.get("done_reason", "unknown"),
        )

        if not answer:
            logger.warning("LangChain returned an empty response.")

        return answer

    except Exception:
        logger.exception(
            "LangChain generation failed after %.2f seconds.",
            time.perf_counter() - start_time,
        )
        raise
def get_tool_calling_model(
    tools,
    temperature=0.0,
    max_tokens=None,
):
    """
    Return a ChatOllama model configured with MCP-backed
    LangChain tools for agent tool calling.
    """

    token_limit = (
        min(MAX_RESPONSE_TOKENS, DEFAULT_MAX_TOKENS)
        if max_tokens is None
        else min(max_tokens, MAX_RESPONSE_TOKENS)
    )

    selected_temperature = (
        TEMPERATURE if temperature is None else temperature
    )

    chat_model = get_chat_model(
        selected_temperature,
        token_limit,
    )

    return chat_model.bind_tools(tools)