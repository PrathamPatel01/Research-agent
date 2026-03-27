
from langchain_groq import ChatGroq
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

# 3. Query generator chain
search_query_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Given a topic, return ONLY a search query. Nothing else."),
    ("human", "Topic: {topic}")
])
query_chain = search_query_prompt | llm

# 4. Digest generator chain
digest_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant. Given search results, write a structured digest with:
    - 📌 Summary (2-3 sentences)
    - 🔑 Key Developments (3-5 bullet points)
    - 🔭 What to Watch Next (1-2 sentences)
    """),
    ("human", "Topic: {topic}\n\nSearch Results: {results}")
])
digest_chain = digest_prompt | llm

# 5. Full pipeline as a function
def research_digest(topic: str):
    print(f"\n🔎 Researching: {topic}")
    
    search_query = query_chain.invoke({"topic": topic}).content
    print(f"🔍 Search query: {search_query}")
    
    results = ddg.run(search_query)
    
    digest = digest_chain.invoke({"topic": topic, "results": results})
    
    print("\n" + "="*50)
    print("📋 RESEARCH DIGEST")
    print("="*50)
    print(digest.content)
    print("="*50)

# 6. Interactive loop
print("🤖 Research Digest Agent")
print("Type a topic to research, or 'quit' to exit\n")

while True:
    topic = input("📝 Enter topic: ").strip()
    
    if topic.lower() == "quit":
        print("👋 Goodbye!")
        break
    
    if not topic:
        print("⚠️  Please enter a topic\n")
        continue
    
    research_digest(topic)
    print()