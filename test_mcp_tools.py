
"""Smoke test for all JustAnotherBot MCP tools."""

import asyncio
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_PATH = PROJECT_ROOT / "mcp_servers" / "tools_server.py"


def print_tool_listing(result):
    print("\nDiscovered MCP tools:")

    if hasattr(result, "tools"):
        entries = result.tools

    elif isinstance(result, tuple):
        entries = next(
            (
                part for part in result
                if isinstance(part, (list, tuple))
            ),
            [],
        )

    else:
        entries = result

    if not isinstance(entries, (list, tuple)):
        entries = [entries]

    for entry in entries:
        if hasattr(entry, "name"):
            print(f"- {entry.name}: {entry.description}")

        elif isinstance(entry, tuple):
            tool = next(
                (
                    part for part in entry
                    if hasattr(part, "name")
                ),
                None,
            )

            if tool:
                print(f"- {tool.name}: {tool.description}")
            else:
                print(entry)

        else:
            print(entry)


def print_result(label, result):
    print(f"\n{'=' * 55}")
    print(label)
    print("=" * 55)

    if getattr(result, "is_error", False):
        print("TOOL RETURNED AN ERROR")

    for block in getattr(result, "content", []):
        text = getattr(block, "text", None)

        if text:
            print(text[:3000])


async def main():

    if not SERVER_PATH.exists():
        raise FileNotFoundError(
            f"MCP server not found: {SERVER_PATH}"
        )

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        cwd=str(PROJECT_ROOT),
    )

    async with Client(params) as client:

        # 1. Discover available tools.
        tools = await client.list_tools()
        print_tool_listing(tools)

        # 2. Calculator.
        calculator = await client.call_tool(
            "calculator",
            {
                "expression": "(25 * 4) + 10"
            },
        )

        print_result("CALCULATOR TEST", calculator)

        # 3. Date and time.
        datetime_result = await client.call_tool(
            "get_current_datetime",
            {
                "timezone_name": "Asia/Tokyo"
            },
        )

        print_result("DATE/TIME TEST", datetime_result)

        # 4. Tavily web search.
        search = await client.call_tool(
            "web_search",
            {
                "query": "Python official website",
                "max_results": 2,
            },
        )

        print_result("TAVILY SEARCH TEST", search)

        # 5. Weather.
        weather = await client.call_tool(
            "get_weather",
            {
                "location": "Ahmedabad, India",
            },
        )

        print_result("WEATHER TEST", weather)

        # 6. Currency conversion.
        currency = await client.call_tool(
            "convert_currency",
            {
                "amount": 100,
                "from_currency": "USD",
                "to_currency": "INR",
            },
        )

        print_result("CURRENCY TEST", currency)


if __name__ == "__main__":
    asyncio.run(main())
