# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_community.tools import DuckDuckGoSearchRun

# print("✅ LangChain imported successfully")
# print("✅ HuggingFace endpoint ready")
# print("✅ DuckDuckGo tool ready")
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

ddg = DuckDuckGoSearchRun()

@tool
def search(query: str) -> str:
    """Search the web for current information."""
    return ddg.run(query)

agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt=(
        "You are a research assistant. "
        "You have exactly one tool: search. "
        "Use only that tool when needed. "
        "Never invent tool names."
    )
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What are the latest developments in quantum computing in 2025?"}
    ]
})

print("\nFinal Answer:\n")
print(response["messages"][-1].content)