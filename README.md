SHL Assessment Recommender
Conversational AI agent that recommends SHL Individual Test Solutions via a stateless
FastAPI service.
Endpoints
Method Path Description
GET /health Returns {"status": "ok"}
POST /chat Stateless conversational recommender
Quick Start (Local)
1. Clone & install
git clone <your-repo>
cd shl-recommender
pip install -r requirements.txt
2. Get a free Gemini API key
1. Go to https://aistudio.google.com/
2. Click Get API Key → Create API key
3. Copy the key
3. Set environment variable
# Mac/Linux
export GEMINI_API_KEY=your_key_here
# Windows
set GEMINI_API_KEY=your_key_here
4. Build the catalog
python scrape_shl.py
This scrapes the SHL product catalog and writes catalog.json.
If scraping fails (network issues), a fallback catalog of ~30 key assessments is used
automatically.
5. Run the server
uvicorn main:app --reload --port 8000
6. Test it
# Health check
curl http://localhost:8000/health
# Chat
curl -X POST http://localhost:8000/chat \
-H "Content-Type: application/json" \
-d '{
"messages": [
{"role": "user", "content": "I am hiring a mid-level Java developer"}
]
}'
# Run full behavior probe suite
python test_agent.py --base-url http://localhost:8000
Deploy to Render (Free)
1. 2. Push this repo to GitHub
Go to https://render.com → New → Web Service → connect your repo
3. Set:
Build Command: pip install -r requirements.txt && python scrape_shl.py
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
4. Add environment variable: GEMINI_API_KEY = your_key_here
5. Deploy → copy your URL → submit to SHL
Request / Response Schema
POST /chat Request
{
"messages": [
{"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
{"role": "assistant", "content": "What seniority level?"},
{"role": "user", "content": "Mid-level, around 4 years"}
]
}
POST /chat Response
{
"reply": "Here are 5 assessments that fit a mid-level Java developer...",
"recommendations": [
{"name": "Java (New)", "url": "https://www.shl.com/...", "test_type": "K"},
{"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
],
"end_of_conversation": false
}
recommendations is an empty array when clarifying or refusing.
end_of_conversation is true only when the agent considers the task complete.
Agent Behaviors
Scenario Agent Action
Vague query (“I need an assessment”) Asks clarifying questions, no recs
Enough context (role + level + goal) Returns 1–10 matched assessments
User refines (“add personality tests”) Updates shortlist in-place
Comparison (“OPQ vs MQ?”) Answers from catalog data
Off-topic / prompt injection Politely refuses, no recs
> 8 turns Graceful end message
Project Structure
shl-recommender/
├── main.py ├── catalog.py ├── scrape_shl.py ├── test_agent.py ├── requirements.txt
├── render.yaml ├── APPROACH.md └── .env.example # FastAPI app, agent logic, Gemini calls
# Catalog loader + fallback data
# SHL website scraper → catalog.json
# Behavior probe test suite
# Render deployment config
# 2-page design document (submission requirement)
# Environment variable template