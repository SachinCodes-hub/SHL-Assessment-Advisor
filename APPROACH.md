SHL Assessment Recommender — Approach
Document
Candidate: [Your Name]
Role: AI Intern, SHL Labs
1. Design Overview
The system is a stateless conversational agent exposed as a FastAPI service. Every POST
/chat call receives the full conversation history and returns a structured response — no
server-side session state. The agent follows a Clarify → Recommend → Refine →
Compare flow.
Architecture
POST /chat
└─ Full conversation history
└─ build_system_prompt() ← injects catalog JSON as context
└─ Gemini 1.5 Flash ← LLM reasoning + structured JSON output
└─ validate_recommendations() ← URL whitelist check
└─ ChatResponse (reply, recommendations, end_of_conversation)
No vector store used. The entire catalog (~30–300 items) fits comfortably within Gemini
1.5 Flash’s 1M-token context window. This eliminates retrieval latency, avoids embedding
drift, and means catalog accuracy is 100% — the model sees every item on every call. For a
catalog of this size, RAG adds complexity with no benefit.
2. Catalog Construction
Scraper ( scrape_shl.py ):
Paginates https://www.shl.com/solutions/products/product-catalog/?type=1
(Individual Test Solutions only)
Extracts: name, URL, test_type, remote_testing, adaptive flags
Enriches each item by fetching its detail page: description, job_levels, languages
Output: catalog.json — written once at build time, loaded at startup
Fallback: A hardcoded get_fallback_catalog() in catalog.py covers ~30 key
assessments (Java, OPQ32r, Verify suite, MQ, SQL, SJTs, etc.) so the service works even if
scraping fails at deploy time.
3. Context Engineering
The system prompt injects the full catalog JSON and defines four behaviors:
Trigger Action
Vague query (no role/level/goal) Ask 1–2 clarifying questions. recommendations: []
Sufficient context Return 1–10 matched assessments
User edits constraints Update existing shortlist in-place
Comparison question Answer from catalog data, not model prior
Off-topic / injection Politely refuse. recommendations: []
JSON-only output: The model is instructed to return only a valid JSON object — no prose
outside it. This eliminates parsing fragility and ensures schema compliance. A regex-based
JSON extractor handles rare markdown-fence wrapping.
Temperature = 0.2: Low temperature for deterministic, consistent recommendations. High
enough to allow natural phrasing in replies.
4. Hallucination Prevention
validate_recommendations() checks every URL against catalog_urls (a set built from
catalog.json )
Any URL not in the set is dropped before the response is returned
Name-based fallback matching catches near-misses
The model is explicitly told: “Never invent names or URLs. Every assessment must come
from the catalog provided.”
5. Constraints Met
Requirement Implementation
GET /health → {"status": "ok"} ✓
POST /chat stateless ✓ Full history sent each call
Schema: reply, recommendations[],
end
of
conversation
_
_
✓ Enforced via Pydantic
Max 8 turns
✓ Early exit with graceful
message
30s timeout ✓ Gemini Flash typically < 5s
Catalog URLs only ✓ Whitelist validation post-LLM
Individual Test Solutions only ✓ Scraper uses type=1 filter
6. Evaluation Approach
test_agent.py runs 10 behavior probe categories locally before submission:
1. Health check
2. Schema compliance (all fields, correct types)
3. Vague query → no recommendations on turn 1
4. Off-topic → refusal
5. Prompt injection resistance
6. Recommend with context (1–10 items, shl.com URLs)
7. Refinement updates shortlist (adds personality type)
8. Comparison answers reference catalog data
9. Turn cap graceful handling
10. Recall@10 on 4 persona traces
7. What Didn’t Work
RAG with FAISS — initially attempted but recall was worse than full-catalog injection
because chunking split related metadata across vectors. Dropped in favor of full catalog
in context.
LangChain — added abstraction that made JSON output parsing unreliable. Replaced
with raw SDK calls.
Scraper HTML structure — SHL catalog uses dynamically-rendered tables. The
scraper has multiple selector fallbacks and degrades gracefully to the hardcoded
catalog.
8. Stack
Component Choice Reason
LLM Gemini 1.5 Flash Free tier, fast (~3–5s), 1M context
Framework
FastAPI + raw Gemini
SDK Defensible, no magic
Catalog
storage
JSON flat file Sufficient for catalog size
Retrieval Full context injection No retrieval errors
Deployment Render (free) Familiar, auto-deploy from GitHub
AI tools used Claude (code
scaffolding)
Used for boilerplate; all design choices are
mine