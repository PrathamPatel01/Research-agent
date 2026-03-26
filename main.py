# # from langchain_huggingface import HuggingFaceEndpoint
# # from langchain_community.tools import DuckDuckGoSearchRun

# # print("✅ LangChain imported successfully")
# # print("✅ HuggingFace endpoint ready")
# # print("✅ DuckDuckGo tool ready")
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# 1. LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. Search tool
ddg = DuckDuckGoSearchRun()

# 3. Step 1 — Decide what to search for
search_query_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Given a topic, return ONLY a search query. Nothing else."),
    ("human", "Topic: {topic}")
])

query_chain = search_query_prompt | llm

# 4. Step 2 — Search the web
def run_search(topic):
    search_query = query_chain.invoke({"topic": topic}).content
    print(f"\n🔍 Searching for: {search_query}")
    return ddg.run(search_query)

# 5. Step 3 — Synthesize into a digest
digest_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant. Given search results, write a structured digest with:
    - 📌 Summary (2-3 sentences)
    - 🔑 Key Developments (3-5 bullet points)
    - 🔭 What to Watch Next (1-2 sentences)
    """),
    ("human", "Topic: {topic}\n\nSearch Results: {results}")
])

digest_chain = digest_prompt | llm

# 6. Run the full pipeline
topic = "latest developments in quantum computing in 2025"

results = run_search(topic)
print("\n📄 Raw Search Results:\n", results[:500], "...")

digest = digest_chain.invoke({"topic": topic, "results": results})

print("\n✅ Research Digest:\n")
print(digest.content)