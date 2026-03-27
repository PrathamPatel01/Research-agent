# 🤖 Research Digest Agent

An AI-powered research assistant built with LangChain and Groq that searches 
the web in real time and generates structured research digests on any topic.

## 🔧 Tech Stack
- **LangChain** — LLM framework and prompt chaining
- **Groq** — Fast LLM inference (LLaMA 3.3 70B)
- **DuckDuckGo Search** — Real-time web search
- **Python** — Core language

## 🧠 Concepts Demonstrated
- Prompt Templates & LCEL chains
- Multi-step reasoning pipeline
- Tool integration (web search)
- Agent-style Think → Search → Synthesize loop

## 🚀 How to Run

1. Clone the repo
2. Create a virtual environment
```bash
   python3.11 -m venv langchain-env
   source langchain-env/bin/activate
```
3. Install dependencies
```bash
   pip install -r requirements.txt
```
4. Add your API key to `.env`
```
   GROQ_API_KEY=your_key_here
```
5. Run the agent
```bash
   python main.py
```

## 📸 Example Output
```
📝 Enter topic: artificial intelligence in healthcare

🔎 Researching: artificial intelligence in healthcare
🔍 Search query: "AI healthcare breakthroughs 2025"

==================================================
📋 RESEARCH DIGEST
==================================================
📌 Summary: ...
🔑 Key Developments: ...
🔭 What to Watch Next: ...
==================================================
```