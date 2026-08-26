# Aster & Row Support Agent

A reliability-first RAG customer-support agent for the Aster & Row AI Agent Intern take-home assignment.

The agent answers policy and product questions from the supplied knowledge base, retrieves order status through a privacy-safe tool, supports multi-turn conversations, cites sources, and recommends human support when information is insufficient or authoritative sources conflict.

## Features

- BM25 retrieval over the supplied Markdown knowledge base
- Front-matter metadata preservation and authority-aware retrieval
- Source citations for policy and product answers
- Privacy-safe order lookup tool
- Safe handling of unknown, malformed, cancelled, returned, shipped, and exception orders
- Multi-turn session memory
- Prompt-injection resistance for retrieved internal/draft content
- Human-handoff signalling for conflicts, insufficient information, privacy requests, and order exceptions
- CLI interface
- Deterministic evaluation suite with visible and original cases

## Setup

```bash
git clone <YOUR_REPOSITORY_URL>
cd aster-row-support-agent

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
Add your Groq key to .env:
GROQ_API_KEY=your_groq_api_key_here
Never commit .env or API keys.
Run the CLI
python3 -m src.cli
Available commands:
/reset
/quit
Example questions:
How long do I have to return an unused backpack?
Where is ORD-1007?
Do you ship internationally?
What about Canada?
Can I put the entire Breeze Tumbler in the dishwasher?
Run Evaluations
python3 -m evaluation.run_eval
The evaluator reports every case individually and summarizes results by category.
Current Evaluation Result
Latest visible/custom evaluation result:
Category	Result
Retrieval	2/2
Conversation	3/3
Groundedness	3/3
Multi-source grounding	1/1
Privacy	1/1
Prompt security	1/1
Abstention	1/1
Tool reliability	5/5
Tool use	2/2
Source conflict	0/1
Total	19/20


The remaining source-conflict case is a response wording mismatch around the Breeze Tumbler’s hand-wash guidance. The agent still correctly identifies the conflict, cites both sources, and flags human support.
Architecture
User message
    |
    v
Retriever (BM25 + document metadata weighting)
    |
    v
Relevant Markdown passages + metadata
    |
    v
Agent orchestration and Groq LLM
    |
    +--> order_lookup tool --> sanitized order result
    |
    +--> flag_for_human_handoff tool
    |
    v
Final answer + source citations + handoff status
Technical Choices
Area	Choice	Reason
Chat model	openai/gpt-oss-20b through Groq	Supports tool calling and produces concise support responses
Retrieval	BM25 using rank_bm25	Small, vocabulary-focused knowledge base; fast and deterministic
Embeddings	Not used	BM25 was sufficient for this small corpus and avoids extra services/cost
Framework	Plain Python	Keeps tool routing, retrieval, and safety behaviour easy to inspect
Storage	In-memory	The supplied documents and order snapshot are static for this assignment
Session memory	In-memory, per session	Supports follow-ups while preventing cross-session leakage


Retrieval and Source Precedence
The knowledge base is split into heading-level chunks. Each chunk preserves document metadata such as:
- status
- policy_authority
- supersedes
- superseded_by
Active, official policy receives higher retrieval priority than superseded, draft, or internal material.
Retrieved content is treated as untrusted data. The agent does not follow instructions embedded in knowledge-base files and does not treat internal migration notes as authoritative policy.
Order Lookup and Privacy
OrderStore.lookup() is the only code path that reads data/orders.json.
The model receives only a sanitized result containing safe customer-facing fields. It never receives:
- Customer names
- Email addresses
- Shipping addresses
- Internal notes
- Risk scores
- Support tags
The tool also:
- Normalizes harmless ID differences such as lowercase IDs and missing hyphens
- Safely handles malformed and unknown IDs
- Suppresses stale carrier, tracking, and ETA fields for cancelled or returned orders
- Does not invent missing delivery dates
- Escalates shipping exceptions to human support
Bug Diary
Bug 1: Order requests without status keywords did not call the tool
How reproduced: Asked: Please check ORD-9999.
Root cause: Earlier logic relied on words such as “delivery” or “tracking” before considering an order lookup.
Fix: Any order-ID-shaped token is now enough to trigger a lookup.
Regression coverage: unknown-order and malformed-order-id-no-prefix evaluation cases.
Bug 2: Cancelled orders could expose stale delivery information
How reproduced: Asked when cancelled order ORD-1004 would arrive.
Root cause: Old carrier and ETA fields remained in the order data even after cancellation.
Fix: OrderStore.lookup() removes carrier, tracking number, and delivery estimate for cancelled and returned orders.
Regression coverage: cancelled-order-stale-eta.
Bug 3: Privacy requests containing a valid order ID could bypass refusal
How reproduced: Asked for the email, address, internal note, and risk score of ORD-1007.
Root cause: The order lookup path had priority over privacy handling.
Fix: Privacy-sensitive requests are detected before order lookup and are escalated without exposing data.
Regression coverage: order-data-privacy.
Bug 4: Exception-status orders were answered without a human handoff
How reproduced: Asked to check ORD-1010, which has a shipping exception.
Root cause: The agent reported the status but did not consistently invoke a handoff.
Fix: Exception orders now produce a deterministic safe response and human-handoff result.
Regression coverage: exception-status-handoff.
Original Evaluation Cases
Additional cases were added beyond the supplied visible suite:
- Exception-status order handoff
- Session isolation
- TrailPlus membership not assumed without confirmation
- Malformed order ID without the ORD- prefix
- Policy question followed by a final-sale exception
Known Limitations
- BM25 can miss heavily paraphrased questions with little vocabulary overlap. A production system could add embeddings or a reranker.
- Session memory is in-process only and disappears when the application restarts.
- The order data is a static mock snapshot, not a live fulfillment system.
- The human-handoff tool records the handoff state but does not create a real support ticket.
- The current source-conflict evaluation has one remaining wording-level failure, despite the agent identifying the conflict and recommending safe interim guidance.
- Groq free-tier rate limits can interrupt full evaluation runs; waiting for quota reset or using a paid plan may be required.
AI Coding Tools Used
I used AI coding assistance for:
- Designing the RAG and tool-use structure
- Debugging evaluation failures
- Improving order-data safety handling
- Drafting tests and README content
One AI-generated suggestion that was incomplete was relying only on the language model to decide when an order required escalation. This was unreliable for unknown, cancelled, and exception orders. The fix was to add deterministic safety handling around order-tool results.
Demo
Replace this placeholder with your recorded GIF or video link before submission.
Watch the demo
The demo shows:
- A knowledge-base answer with source citations
- An order lookup
- A multi-turn Canada follow-up
- A safe handoff/refusal
- The evaluation suite running
