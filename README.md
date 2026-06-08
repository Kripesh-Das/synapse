# synapse

> Hospital AI Agent Architecture — MiniCPM5-1B optimized  
> *Local deployment · Deterministic clinical logic · LangGraph orchestration*

---

## Architecture Diagram

![synapse Architecture](diagrams/synapse-arch.svg)

> **Edit this diagram:** open [`diagrams/synapse-arch.excalidraw`](diagrams/synapse-arch.excalidraw) in [excalidraw.com](https://excalidraw.com) or VS Code with the Excalidraw extension.

---

## Overview

synapse is a hospital triage assistant that separates **conversation** (handled by MiniCPM5-1B) from **clinical decisions** (handled by deterministic Python code). The LLM talks to patients — it never makes clinical decisions.

### Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Determinism by default** | 6 of 10 nodes are pure Python; only 4 use LLM |
| **SDB (Stochastic-Deterministic Boundary)** | LLM proposes, deterministic code verifies |
| **Emergency-first** | Keyword matcher runs BEFORE the LLM sees the message |
| **Progressive tool disclosure** | 3-10 tools visible per phase (not all 15) |
| **Per-node checkpointing** | PostgreSQL checkpoints after every node; resume from crash |
| **Guardrails in code** | Validation layer blocks medical advice, score leakage |

### What the Model Does NOT Do

- ❌ Emergency detection
- ❌ Triage scoring
- ❌ Medical diagnosis
- ❌ Department routing
- ❌ Clinical summaries (template-based, not generated)

---

## Quick Start

```bash
# Prerequisites
pip install langgraph langchain-core
# Add MiniCPM5-1B via SGLang or llama.cpp
```

```python
from synapse.graph import build_graph

graph = build_graph(checkpointer=PostgresSaver(pg_pool))
result = graph.invoke({"patient_id": "PT-001"}, config={"thread_id": "session-abc"})
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (StateGraph) |
| LLM | MiniCPM5-1B (SGLang) |
| Session state | Redis (TTL 3600s) |
| Checkpointing | PostgreSQL |
| Knowledge | RAG (Vector DB) |
| Observability | Langfuse |
| Safety | Deterministic keyword matcher + rules engine |

---

## Architecture Spec

Full design document: [`AGENTS.md`](AGENTS.md)

