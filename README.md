# Aster & Row Support Agent

A reliability-first Retrieval-Augmented Generation (RAG) customer-support agent developed for the **Aster & Row AI Agent Intern take-home assignment**.

The agent answers policy and product-related questions using a supplied knowledge base, retrieves order information through a privacy-safe tool, supports multi-turn conversations, provides source citations, and recommends human support when information is insufficient, conflicting, or requires manual intervention.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Technical Stack](#technical-stack)
5. [Project Setup](#project-setup)
6. [Running the CLI](#running-the-cli)
7. [Running Evaluations](#running-evaluations)
8. [Retrieval and Source Precedence](#retrieval-and-source-precedence)
9. [Order Lookup and Privacy](#order-lookup-and-privacy)
10. [Multi-Turn Conversation](#multi-turn-conversation)
11. [Safety and Prompt-Injection Resistance](#safety-and-prompt-injection-resistance)
12. [Evaluation Results](#evaluation-results)
13. [Bug Diary](#bug-diary)
14. [Original Evaluation Cases](#original-evaluation-cases)
15. [Known Limitations](#known-limitations)
16. [AI Coding Tools Used](#ai-coding-tools-used)
17. [Demo](#demo)

---

## Project Overview

The **Aster & Row Support Agent** is a customer-support system built around a Retrieval-Augmented Generation (RAG) architecture.

The system combines:

* Deterministic BM25-based document retrieval
* Authority-aware knowledge-base processing
* Groq-hosted language model inference
* Privacy-safe order lookup
* Multi-turn session memory
* Source citations
* Human-handoff mechanisms
* Prompt-injection resistance
* Deterministic evaluation and regression testing

The primary design goal is **reliability**. When the available information is incomplete, conflicting, sensitive, or requires human intervention, the system is designed to avoid guessing and instead provide a safe response or recommend human support.

---

## Features

### Knowledge Base Retrieval

* BM25 retrieval over the supplied Markdown knowledge base.
* Heading-level document chunking.
* Preservation of front-matter metadata.
* Authority-aware retrieval and ranking.
* Support for active, superseded, draft, and internal documents.
* Source citations in policy and product responses.

### Order Support

* Privacy-safe order lookup through a dedicated tool.
* Handles valid, unknown, malformed, cancelled, returned, shipped, and exception-status orders.
* Normalizes harmless order-ID variations such as lowercase IDs and missing hyphens.
* Prevents exposure of internal order information.
* Automatically recommends human support for shipping exceptions.

### Conversation Support

* Multi-turn conversation handling.
* Session-specific memory.
* Follow-up question handling.
* Session isolation to prevent information leakage between conversations.

### Safety

* Prompt-injection resistance for retrieved knowledge-base content.
* Retrieved documents are treated as untrusted data.
* Internal or draft documents are not automatically treated as authoritative.
* Privacy-sensitive requests are refused or escalated.
* The system avoids inventing unavailable information.

### Human Handoff

Human support is recommended when:

* Knowledge-base sources conflict.
* The available information is insufficient.
* A request involves sensitive or private information.
* An order has a shipping exception.
* The issue requires manual investigation.

### Evaluation

* Deterministic evaluation suite.
* Individual case reporting.
* Category-level result summaries.
* Regression coverage for previously identified bugs.
* Additional original test cases beyond the supplied visible evaluation suite.

---

## Architecture

The overall system follows the architecture below:

```text
User Message
     |
     v
Retriever
(BM25 + Metadata Weighting)
     |
     v
Relevant Knowledge-Base Passages
+ Metadata
     |
     v
Agent Orchestration
+ Groq LLM
     |
     +----------------------+
     |                      |
     v                      v
Order Lookup Tool     Human Handoff Tool
     |                      |
     v                      v
Sanitized Order       Handoff Status
Result
     |
     v
Final Response
+ Source Citations
+ Handoff Status
```

### Main Components

| Component           | Responsibility                                                      |
| ------------------- | ------------------------------------------------------------------- |
| Retriever           | Finds relevant knowledge-base passages using BM25                   |
| Metadata weighting  | Gives priority to authoritative and active documents                |
| Agent orchestration | Coordinates retrieval, reasoning, tool use, and response generation |
| Groq LLM            | Generates concise customer-support responses                        |
| Order Lookup Tool   | Retrieves and sanitizes order information                           |
| Human Handoff Tool  | Records when manual support is required                             |
| Session Memory      | Maintains context within an individual conversation                 |
| Evaluation Suite    | Tests retrieval, safety, privacy, grounding, and tool behaviour     |

---

## Technical Stack

| Area                 | Technology                        | Reason                                                                     |
| -------------------- | --------------------------------- | -------------------------------------------------------------------------- |
| Programming Language | Python                            | Simple, readable, and suitable for rapid development                       |
| Chat Model           | `openai/gpt-oss-20b` through Groq | Supports tool calling and concise support responses                        |
| Retrieval            | BM25 using `rank_bm25`            | Suitable for the small, vocabulary-focused knowledge base                  |
| Embeddings           | Not used                          | BM25 was sufficient for the supplied corpus and avoids additional services |
| Framework            | Plain Python                      | Keeps retrieval, tool routing, and safety behaviour transparent            |
| Storage              | In-memory                         | The supplied documents and order snapshot are static                       |
| Session Memory       | In-memory, per session            | Supports follow-up questions while preventing cross-session leakage        |
| Knowledge Base       | Markdown                          | Easy to inspect, modify, and preserve metadata                             |

---

## Project Setup

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd aster-row-support-agent
```

### 2. Create a Virtual Environment

For Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Add the Groq API key to `.env`:

```text
GROQ_API_KEY=your_groq_api_key_here
```

> **Important:** Never commit `.env`, API keys, or other secrets to the repository.

---

## Running the CLI

Start the support agent using:

```bash
python3 -m src.cli
```

The CLI supports the following commands:

```text
/reset
/quit
```

### Example Questions

```text
How long do I have to return an unused backpack?
```

```text
Where is ORD-1007?
```

```text
Do you ship internationally?
```

```text
What about Canada?
```

```text
Can I put the entire Breeze Tumbler in the dishwasher?
```

The system can also maintain context across related follow-up questions.

---

## Running Evaluations

Run the complete evaluation suite with:

```bash
python3 -m evaluation.run_eval
```

The evaluator reports:

* Individual test-case results
* Evaluation categories
* Overall score
* Failures and mismatches

The evaluation suite is designed to test both the supplied visible cases and additional original cases created during development.

---

## Retrieval and Source Precedence

The knowledge base is divided into heading-level chunks.

Each chunk preserves important document metadata, including:

* `status`
* `policy_authority`
* `supersedes`
* `superseded_by`

### Authority-Aware Retrieval

The retrieval system gives higher priority to:

1. Active official policy
2. Current authoritative product information
3. Other relevant active sources
4. Superseded material
5. Draft or internal material

This prevents outdated or internal content from being treated as the current customer-facing policy.

### Source Conflict Handling

When authoritative sources conflict, the agent does not silently choose one source and present it as unquestionably correct.

Instead, it:

1. Identifies the conflict.
2. Considers the authority and status of the sources.
3. Provides appropriate interim guidance where possible.
4. Cites the relevant sources.
5. Recommends human support when the conflict cannot be safely resolved.

---

## Order Lookup and Privacy

`OrderStore.lookup()` is the only code path that reads:

```text
data/orders.json
```

The language model does not receive the complete raw order record.

Instead, the order tool returns a sanitized result containing only customer-facing information.

### Information Never Exposed to the Model

The system prevents exposure of:

* Customer names
* Email addresses
* Shipping addresses
* Internal notes
* Risk scores
* Support tags

### Order-ID Normalization

The tool safely handles harmless formatting differences, including:

* Lowercase order IDs
* Missing hyphens
* Other supported formatting variations

For example, an order ID can be normalized before lookup rather than being treated as a completely different order.

### Order Status Handling

The system handles:

* Shipped orders
* Cancelled orders
* Returned orders
* Unknown orders
* Malformed order IDs
* Shipping exceptions

### Cancelled and Returned Orders

Cancelled and returned orders may contain stale carrier or delivery information in the underlying data.

To prevent misleading responses, the tool suppresses:

* Carrier information
* Tracking numbers
* Delivery estimates

for cancelled or returned orders.

The system does not invent a delivery date when one is unavailable.

### Shipping Exceptions

Orders with exception statuses are reported safely and trigger human-support escalation rather than being treated as normal deliveries.

---

## Multi-Turn Conversation

The agent supports contextual follow-up questions within the same session.

For example:

```text
User: Do you ship internationally?

Agent: [Provides international shipping policy.]

User: What about Canada?

Agent: [Uses the previous conversation context to answer specifically about Canada.]
```

Session memory is maintained in memory and is associated with the current session.

Different sessions remain isolated to prevent accidental cross-session information leakage.

---

## Safety and Prompt-Injection Resistance

Retrieved knowledge-base content is treated as **untrusted data**.

The agent does not automatically follow instructions contained inside retrieved documents.

This is particularly important for:

* Internal notes
* Draft documents
* Migration notes
* Superseded policies
* Other non-authoritative material

The system distinguishes between:

1. Instructions from the application and system logic.
2. Customer-facing policy information.
3. Retrieved content that may contain untrusted instructions.

This prevents a malicious or accidental instruction embedded in a knowledge-base document from overriding the agent's intended behaviour.

---

## Human Handoff

The agent uses human support escalation when automated handling is unsafe or insufficient.

Human handoff can occur for:

| Situation                                  | Action                                          |
| ------------------------------------------ | ----------------------------------------------- |
| Conflicting authoritative sources          | Flag for human support                          |
| Insufficient information                   | Avoid guessing and recommend support            |
| Privacy-sensitive request                  | Refuse disclosure and escalate when appropriate |
| Shipping exception                         | Provide safe status and recommend human support |
| Other cases requiring manual investigation | Flag for human support                          |

The current handoff implementation records the handoff state but does not create a real external support ticket.

---

## Evaluation Results

The latest visible/custom evaluation produced the following results:

| Category               |    Result |
| ---------------------- | --------: |
| Retrieval              |       2/2 |
| Conversation           |       3/3 |
| Groundedness           |       3/3 |
| Multi-source grounding |       1/1 |
| Privacy                |       1/1 |
| Prompt security        |       1/1 |
| Abstention             |       1/1 |
| Tool reliability       |       5/5 |
| Tool use               |       2/2 |
| Source conflict        |       0/1 |
| **Total**              | **19/20** |

### Source-Conflict Result

The remaining source-conflict case is a **response wording mismatch** related to the Breeze Tumbler's hand-wash guidance.

The agent still:

* Correctly identifies the conflict.
* Retrieves both relevant sources.
* Cites both sources.
* Avoids confidently presenting conflicting information as settled policy.
* Flags the issue for human support.

Therefore, the remaining failure is primarily an evaluation wording mismatch rather than a failure to detect or safely handle the underlying conflict.

---

## Bug Diary

### Bug 1: Order Requests Without Status Keywords Did Not Trigger the Tool

**How it was reproduced:**

```text
Please check ORD-9999.
```

**Root Cause:**

The earlier implementation relied on words such as `delivery` or `tracking` before considering an order lookup.

**Fix:**

Any token matching the expected order-ID pattern is now sufficient to trigger the order lookup process.

**Regression Coverage:**

* `unknown-order`
* `malformed-order-id-no-prefix`

---

### Bug 2: Cancelled Orders Could Expose Stale Delivery Information

**How it was reproduced:**

The agent was asked when cancelled order `ORD-1004` would arrive.

**Root Cause:**

Old carrier and ETA fields remained in the order data even after cancellation.

**Fix:**

`OrderStore.lookup()` now removes:

* Carrier
* Tracking number
* Delivery estimate

for cancelled and returned orders.

**Regression Coverage:**

* `cancelled-order-stale-eta`

---

### Bug 3: Privacy Requests Containing a Valid Order ID Could Bypass Refusal

**How it was reproduced:**

A request was made for the email address, shipping address, internal note, and risk score associated with `ORD-1007`.

**Root Cause:**

The order lookup path had priority over privacy-sensitive request handling.

**Fix:**

Privacy-sensitive requests are detected before order lookup and are escalated without exposing protected information.

**Regression Coverage:**

* `order-data-privacy`

---

### Bug 4: Exception-Status Orders Were Answered Without Human Handoff

**How it was reproduced:**

The agent was asked to check `ORD-1010`, which has a shipping exception.

**Root Cause:**

The agent reported the order status but did not consistently invoke human handoff.

**Fix:**

Exception-status orders now produce a deterministic safe response and a human-handoff result.

**Regression Coverage:**

* `exception-status-handoff`

---

## Original Evaluation Cases

Additional evaluation cases were created beyond the supplied visible test suite.

These include:

### 1. Exception-Status Order Handoff

Verifies that an order with a shipping exception is reported safely and escalated to human support.

### 2. Session Isolation

Verifies that information from one conversation session does not leak into another session.

### 3. TrailPlus Membership

Verifies that the agent does not assume a customer has TrailPlus membership without sufficient evidence or confirmation.

### 4. Malformed Order ID Without the `ORD-` Prefix

Verifies that malformed order identifiers are handled safely rather than causing incorrect lookups or hallucinated results.

### 5. Policy Question Followed by a Final-Sale Exception

Verifies that the agent can handle a normal policy question followed by a specific exception in the same conversation.

---

## Known Limitations

### 1. BM25 Retrieval

BM25 performs well for the supplied small knowledge base but may miss heavily paraphrased questions with very little vocabulary overlap.

A production implementation could improve this by combining BM25 with:

* Embedding-based retrieval
* Hybrid search
* A reranking model

### 2. In-Memory Session Memory

Session memory exists only in the running process.

It is lost when the application restarts.

A production system could use persistent session storage such as a database or dedicated session store.

### 3. Static Order Data

The current order information comes from a static mock snapshot rather than a live fulfillment system.

### 4. Human Handoff

The handoff tool records the handoff state but does not currently create an actual support ticket or connect to a customer-service platform.

### 5. Source Conflict Evaluation

One source-conflict evaluation case currently fails because of a wording-level mismatch related to the Breeze Tumbler's hand-wash guidance.

The underlying safety behaviour remains correct: the agent identifies the conflict, cites both sources, and recommends human support.

### 6. Groq Rate Limits

Groq free-tier rate limits may interrupt complete evaluation runs.

If this occurs, the evaluation can be retried after the quota resets or run using an appropriate paid plan.

---

## AI Coding Tools Used

AI coding assistance was used during development for:

* Designing the RAG and tool-use structure
* Debugging evaluation failures
* Improving order-data privacy and safety handling
* Drafting tests
* Improving README documentation

### Example of an AI-Generated Approach That Required Improvement

One AI-generated suggestion relied primarily on the language model to determine when an order required escalation.

This approach was unreliable for cases such as:

* Unknown orders
* Cancelled orders
* Shipping exceptions

The approach was therefore improved by adding **deterministic safety handling around order-tool results**.

This ensures that important safety decisions are not dependent solely on probabilistic language-model behaviour.

---

## Demo

A recorded GIF or video demonstrating the project can be linked below.

**Watch demo here:** (https://drive.google.com/file/d/1lCRFzmBgZLnG8uzL47Cxov2criOFE2CV/view?usp=sharing)

The demonstration covers:

1. A knowledge-base question with source citations.
2. An order lookup.
3. A multi-turn Canada shipping follow-up.
4. A safe refusal/human handoff.
5. The evaluation suite running.

---

## Conclusion

The Aster & Row Support Agent demonstrates a reliability-focused approach to building a RAG-based customer-support system.

The implementation combines deterministic retrieval, authority-aware source handling, privacy-safe tool use, session isolation, prompt-injection resistance, and human escalation mechanisms.

The current evaluation score of **19/20** demonstrates strong performance across retrieval, conversation handling, groundedness, privacy, prompt security, abstention, and tool reliability, with the remaining issue limited to a wording-level source-conflict evaluation case.
