
import asyncio

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ai.rag.rag_pipeline import (
    retrieve_relevant_chunks,
    build_context,
)
from ai.rag.vector_store import get_collection_count


# --------------------------------------------------
# 1. Define the document search tool
# --------------------------------------------------


@tool
async def search_documents(
    question: str = "",
    query: str = "",
    context: str = "",
) -> str:
    """Search indexed documents and return relevant excerpts."""

    search_question = " ".join(
        part.strip()
        for part in [question, query, context]
        if part and part.strip()
    )

    if not search_question:
        return "No search question was provided."

    print(f"\nSearching: {search_question}")

    if get_collection_count() == 0:
        return "No documents have been indexed."

    results = await retrieve_relevant_chunks(search_question)

    if not results:
        return "No relevant document excerpts found."

    context_text, sources = build_context(results)

    print("\n========== RETRIEVED EXCERPTS ==========\n")
    print(context_text)
    print("\n========== END OF EXCERPTS ==========\n")

    source_list = "\n".join(
        f"[{s['citation']}] {s['source']}"
        for s in sources
    )

    return (
        f"DOCUMENT EXCERPTS:\n{context_text}\n\n"
        f"SOURCES:\n{source_list}\n\n"
        "Explain the actual events and information found in "
        "the excerpts. Do not merely list page numbers or "
        "section names. Cite factual claims using [1], [2], etc."
    )
# --------------------------------------------------
# 2. Configure Qwen with the tool
# --------------------------------------------------

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)

tools = [search_documents]

llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------
# 3. Run the tool-calling workflow
# --------------------------------------------------

async def main():

    question = "What are the important events in Chapter 3?"

    messages = [
        SystemMessage(
            content=(
             "You are JustAnotherBot, a refined Victorian gentleman.\n"
             "You have access to a document search tool named search_documents.\n\n"
            "IMPORTANT RULES:\n"
            "1. For EVERY question about indexed documents, "
            "you MUST call search_documents before answering.\n"
            "2. Never ask the user which document they mean. "
            "Search the indexed collection first.\n"
            "3. The search tool handles chapter filtering.\n"
            "4. After receiving search results, explain the actual "
            "information in the excerpts.\n"
            "5. Do not merely list page numbers or section names.\n"
            "6. Include citations such as [1] where appropriate.\n"
            "7. Never invent facts or events.\n"
             )
        ),
        HumanMessage(content=question),
    ]

    print(f"\nUser: {question}")
    print("\nJustAnotherBot is thinking...\n")

    # First model call: decide whether to use the tool.
    response = await llm_with_tools.ainvoke(messages)

    messages.append(response)

    # Execute requested tools.
   if not response.tool_calls:
    print("Qwen did not request a tool.")
    print("Forcing document search...\n")

    result = await search_documents.ainvoke({
        "question": question
    })

    messages.append(
        HumanMessage(
            content=(
                "Here are the mandatory document search results. "
                "Answer the original question using them:\n\n"
                + result
            )
        )
    )

    final_response = await llm_with_tools.ainvoke(messages)

    print("\nJustAnotherBot says:\n")
    print(final_response.content)
    return
    for tool_call in response.tool_calls:

        print(f"Tool requested: {tool_call['name']}")
        print(f"Arguments: {tool_call['args']}")

        if tool_call["name"] == "search_documents":

            result = await search_documents.ainvoke(
                tool_call["args"]
            )

            print("\nDocument retrieval completed.")

            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                )
            )

    # Second model call: formulate the final answer.
    final_response = await llm_with_tools.ainvoke(messages)

    print("\nJustAnotherBot says:\n")
    print(final_response.content)


if __name__ == "__main__":
    asyncio.run(main())
