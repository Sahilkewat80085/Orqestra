# Orqestra

**Production-grade Multi-Agent LLM Orchestration & Evaluation Platform**

> A real AI infrastructure platform — not a chatbot wrapper, not a tutorial demo.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Orqestra coordinates five specialized AI agents through a **typed shared context object**, exposes five documented FastAPI endpoints, streams all activity over SSE, and includes a complete evaluation harness with a self-improving prompt loop.

**Key properties:**
- **Dynamic routing** — no static agent chains; the Orchestrator decides at runtime
- **Typed shared context** — agents ONLY communicate through `SharedContext`
- **Full observability** — every tool call, retry, budget update, and policy violation is streamed
- **Reproducible evals** — 15 test cases, 6-dimension custom scorer, full trace storage
- **Human-in-the-loop** — MetaAgent proposes prompt rewrites; humans approve via API

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI Server (port 8000)                 │
│   POST /api/v1/query     GET /api/v1/query/{id}/stream (SSE)     │
│   GET  /api/v1/trace/{id}      GET /api/v1/evals/latest          │
│   POST /api/v1/evals/rerun     POST /api/v1/prompts/approve       │
└──────────────────────┬───────────────────────────────────────────┘
                       │ SSE + REST
            ┌──────────▼──────────┐
            │  OrchestratorRouter  │  ← LLM call → RoutingDecision
            │  (LangGraph entry)   │  ← No static chains
            └──┬────────┬────┬────┘
               │        │    │  conditional edges from routing_decision
    ┌──────────▼─┐  ┌───▼──┐  ┌─▼──────────┐  ┌───────────┐
    │Decomposition│  │Retr- │  │Critique    │  │Synthesis  │
    │Agent        │  │ieval │  │Agent       │  │Agent      │
    └─────┬───────┘  └──┬───┘  └─────┬──────┘  └─────┬─────┘
          └─────────────┴─────────────┴────────────────┘
                              │
                   SharedContext (Pydantic typed)
                   ← Only channel for inter-agent comms
                              │
         ┌────────────────────────────────────────┐
         │  Tool Layer                            │
         │  WebSearch  │ PySandbox │ NL2SQL       │
         │  SelfReflection                        │
         │  All: retryable • logged • auditable   │
         └────────────────────────────────────────┘
                              │
         ┌────────────────────────────────────────┐
         │  Persistence                           │
         │  PostgreSQL — traces, evals, prompts  │
         │  Redis — job queue + SSE pub/sub       │
         │  ChromaDB — vector store (RAG)         │
         └────────────────────────────────────────┘
```

---

## Project Structure

```
orqestra/
├── app/
│   ├── agents/          # 5 agents: decomposition, retrieval, critique, synthesis, meta
│   ├── orchestrator/    # LangGraph graph, context budget, retry policy, router
│   ├── tools/           # 4 tools: web_search, python_sandbox, nl2sql, self_reflection
│   ├── evals/           # 15 test cases, runner, custom 6-dim scorer, storage
│   ├── streaming/       # Typed SSE events, Redis publisher, SSE subscriber
│   ├── logging/         # structlog JSON logger, FastAPI middleware
│   ├── database/        # SQLAlchemy models, session, repositories
│   ├── api/             # 5 FastAPI routes
│   ├── schemas/         # Pydantic models: SharedContext, agents, tools, evals
│   ├── services/        # Query, eval, prompt services
│   ├── utils/           # SHA-256 hashing, tiktoken counter
│   ├── config.py        # pydantic-settings, all env vars
│   └── main.py          # FastAPI app factory
├── tests/               # pytest test suite
├── scripts/             # seed_db, export_trace
├── docker/              # Dockerfile.api, Dockerfile.worker
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — set GOOGLE_API_KEY and POSTGRES_PASSWORD at minimum
```

### 2. Start all services

```bash
docker compose up --build
```

Zero manual setup. All services start automatically with healthchecks.

### 3. Seed the knowledge base

```bash
python scripts/seed_db.py
```

### 4. Submit a query and stream results

```bash
# Submit query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation and why is it used?"}'

# Stream SSE events (replace with returned query_id)
curl -N http://localhost:8000/api/v1/query/{query_id}/stream
```

### 5. View API docs

```
http://localhost:8000/docs
```

---

## Agents

| Agent | Responsibility | Key Constraint |
|-------|---------------|----------------|
| **Orchestrator** | Dynamic routing, execution plan, context budget init | No static chains; LLM decides routing at runtime |
| **Decomposition** | Break query into typed sub-tasks with dependency graph | Output must be typed `TaskGraph` |
| **Retrieval** | Multi-hop vector search from ChromaDB | ≥2 hops, ≥2 chunks, chunk→claim provenance required |
| **Critique** | Span-level review of all agent outputs | Must target specific text spans, not entire outputs |
| **Synthesis** | Merge outputs, resolve contradictions, final answer | Sentence-level provenance on every output sentence |
| **MetaAgent** | Analyze failed evals, propose prompt rewrites | Rewrites require human approval — never auto-applied |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/query` | Submit query, get `query_id` and `stream_url` |
| `GET` | `/api/v1/query/{id}/stream` | SSE stream — real-time agent/tool events |
| `GET` | `/api/v1/trace/{query_id}` | Full execution trace for any completed query |
| `GET` | `/api/v1/evals/latest` | Latest eval run with per-case scores + justifications |
| `POST` | `/api/v1/evals/rerun` | Targeted re-eval for specific case IDs |
| `POST` | `/api/v1/prompts/approve` | Approve or reject a MetaAgent prompt rewrite |
| `GET` | `/api/v1/prompts/pending` | List pending prompt rewrites |

All endpoints return structured errors with machine-readable `code` fields.

---

## Evaluation Methodology

Orqestra includes 15 concrete test cases across three categories:

| Category | Count | Description |
|----------|-------|-------------|
| **Baseline** | 5 | Well-defined factual queries with known keywords |
| **Ambiguous** | 5 | Underspecified queries requiring decomposition |
| **Adversarial** | 5 | Prompt injection, false premises, contradiction induction, instruction override, SQL injection |

Each case is scored on **6 dimensions**:

| Dimension | What it measures |
|-----------|-----------------|
| `correctness` | Keyword coverage + adversarial detection behavior |
| `citation_accuracy` | Citations provided when required |
| `contradiction_handling` | Critique agent invoked; contradictions flagged |
| `tool_efficiency` | No unnecessary tool calls for simple queries |
| `context_compliance` | Zero policy violations (budget not exceeded) |
| `critique_agreement` | High-severity critique findings addressed in synthesis |

**Every score includes a written justification string** — no black-box evaluation.

---

## Observability

Every SSE event includes:

```json
{
  "query_id": "...",
  "event_type": "agent_completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "payload": {
    "agent_id": "retrieval",
    "latency_ms": 423.5,
    "token_count": 1240
  },
  "sequence": 4
}
```

Event types: `agent_started`, `agent_completed`, `tool_call_started`, `tool_call_completed`, `tool_retry`, `budget_update`, `policy_violation`, `routing_decision`, `final_answer`, `pipeline_complete`.

All structured logs include: `timestamp`, `agent_id`, `event_type`, `latency_ms`, `token_count`, `input_hash`, `output_hash`, `policy_violations`.

---

## Self-Improving Prompt Loop

1. **Trigger**: Eval run completes; MetaAgent analyzes failed cases
2. **Analysis**: Root cause analysis identifies weak prompts by agent
3. **Proposal**: Structured `PromptRewriteDiff` with before/after content and reasoning
4. **Review**: Human approves/rejects via `POST /api/v1/prompts/approve`
5. **Re-eval**: On approval, ONLY failed cases are re-run
6. **Delta**: Score delta is computed and logged

Nothing is auto-applied. All changes are fully auditable with approver identity and timestamps.

---

## Known Limitations

- LangGraph state is in-memory per request; horizontal scaling requires external state store
- Python sandbox is subprocess-based; production would use gVisor or Firecracker
- Web search tool is stubbed by default (`WEB_SEARCH_MOCK=true`)
- MetaAgent prompt rewrites are not applied across running instances without restart
- ChromaDB is single-node; production would use Weaviate or Qdrant cluster

---

## Future Improvements

- [ ] Parallel agent execution for independent steps in the task graph
- [ ] Redis-backed LangGraph state for horizontal scaling
- [ ] gVisor/Firecracker-based Python sandbox
- [ ] Real-time token streaming at character level (not just agent-level events)
- [ ] Grafana dashboard for eval score trends
- [ ] A/B testing for prompt versions before promotion to active
- [ ] Webhook notifications on policy violations

---

## License

MIT
