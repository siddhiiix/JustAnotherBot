
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# 1. Define a tool
@tool
def calculator(a: float, b: float, operation: str) -> float:
    """
    Perform basic mathematical calculations.

    Args:
        a: First number.
        b: Second number.
        operation: One of add, subtract, multiply, divide.
    """

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":
        if b == 0:
            return "Cannot divide by zero."
        return a / b

    return "Unsupported operation."


# 2. Initialise Ollama
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)

# 3. Register the tool with the model
tools = [calculator]
llm_with_tools = llm.bind_tools(tools)

# 4. Ask a question
messages = [
    SystemMessage(
        content=(
            "You are JustAnotherBot, a Victorian gentleman. "
            "Use the calculator tool for mathematical calculations. "
            "Do not calculate these answers mentally."
        )
    ),
    HumanMessage(
        content="What is 245 multiplied by 36?"
    ),
]

print("Sending question to JustAnotherBot...\n")

try:
    # 5. Model decides whether to call a tool
    response = llm_with_tools.invoke(messages)

    if not response.tool_calls:
        print("No tool was called.")
        print(response.content)

    else:
        # 6. Execute requested tool calls
        messages.append(response)

        for tool_call in response.tool_calls:
            print(f"Tool requested: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")

            selected_tool = next(
                tool for tool in tools
                if tool.name == tool_call["name"]
            )

            result = selected_tool.invoke(tool_call["args"])

            print(f"Tool result: {result}\n")

            # 7. Return tool result to the model
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

        # 8. Model generates final response
        final_response = llm_with_tools.invoke(messages)

        print("JustAnotherBot says:\n")
        print(final_response.content)

except Exception as error:
    print(f"Tool calling failed: {error}")
