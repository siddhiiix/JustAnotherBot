
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Connect LangChain to your local Ollama model
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.7,
)

# Define conversation messages
messages = [
    SystemMessage(
        content="You are JustAnotherBot, a witty Victorian gentleman."
    ),
    HumanMessage(
        content="Explain what LangChain is in two sentences."
    ),
]

# Generate response
print("Connecting to Ollama through LangChain...\n")

try:
    response = llm.invoke(messages)

    print("JustAnotherBot says:\n")
    print(response.content)

except Exception as e:
    print(f"Something went wrong: {e}")