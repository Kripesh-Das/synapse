# synapse — Hospital AI Triage Assistant

## What this is

LangGraph-based triage system using MiniCPM5-1B (local, 1B params). The LLM handles patient conversation only. All clinical logic (emergency detection, triage scoring, department routing) is deterministic Python — never the LLM.

**Status**: Core agent implemented. 175 tests passing. Graph compiles. CLI runner works with mock LLM. WebSocket API ready. LLM integration requires running SGLang server.

## Non-negotiable constraints

- **LLM never makes clinical decisions.** Emergency detection, triage scoring, and routing are keyword matchers and rules engines. If you're tempted to ask the model "how urgent is this?", stop.
- **Emergency detection runs BEFORE the LLM sees the message.** Keyword matcher first, LLM second. Zero exceptions.
- **MiniCPM5-1B outputs XML-style tool calls**, parsed by SGLang to OpenAI-compatible format. Not standard OpenAI function calling.
- **Progressive tool disclosure**: Each workflow phase exposes only 3-10 of the 15 tools. The harness decides what the model can see, not the model.
- **All LLM outputs must be validated as JSON.** The model occasionally produces malformed output. Parse with retry (1 max), fallback to deterministic template on failure.
- **`triage_score` must never appear in patient-facing output.** This is enforced in the validation layer.

## Project structure

```
synapse/
├── synapse/
│   ├── graph.py              # build_graph() — StateGraph wiring
│   ├── state.py              # SessionState TypedDict
│   ├── config.py             # Settings from .env
│   ├── cli.py                # CLI runner (--mock mode available)
│   ├── nodes/                # 10 workflow nodes
│   ├── tools/                # 13 tool definitions + registry
│   ├── extractors/           # Emergency detector (keyword matcher)
│   ├── triage/               # Rules engine + department routing
│   ├── prompts/              # LLM prompt templates
│   ├── llm/                  # OpenAI-compatible client (SGLang)
│   ├── validation/           # JSON parse + patient safety checks
│   ├── db/                   # PostgreSQL schemas + repository
│   ├── cache/                # Redis session cache
│   └── api/                  # FastAPI + WebSocket server
├── tests/                    # 175 tests
├── pyproject.toml
├── langgraph.json
├── docker-compose.yml        # PostgreSQL + Redis + SGLang
└── .env.example
```

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Run tests (no LLM needed)
python -m pytest tests/ -v

# Run CLI with mock LLM (no SGLang needed)
python -m synapse.cli --mock

# Single message through CLI
python -m synapse.cli --mock -m "I have chest pain"

# Start API server (WebSocket + REST)
python -m synapse.api

# Start infrastructure
docker compose up -d

# Start SGLang (requires NVIDIA GPU)
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM5-1B \
    --port 30000 \
    --tool-call-parser minicpm5

# Run agent with real LLM
python -m synapse.cli
```

## Workflow (10 nodes)

```
session_manager → welcome_agent → symptom_collector → triage_engine → summary_generator → data_persistence → session_close
                     ↑                   ↓ (emergency)          ↓ (score<=2)
                     └─────────── emergency_handler        urgent_notification → summary_generator
```

Key edges:
- `symptom_collector` → `emergency_handler` if keyword matcher fires (P10 priority)
- `symptom_collector` loops until `sufficient_data` or `turn_count > 20`
- `triage_engine` → `urgent_notification` if score <= 2

## What's built vs. what's TODO

### Done
- SessionState TypedDict with all fields
- Emergency keyword detector (6 categories, 50+ keywords, comprehensive tests)
- Triage rules engine (critical/high-severity scoring, age/pregnancy adjustments)
- Department routing table (12 departments)
- JSON validator (parse, retry, patient safety checks)
- All 10 node implementations
- Graph wiring with conditional edges
- Tool registry with progressive disclosure
- LLM client wrapper (OpenAI-compatible for SGLang)
- PostgreSQL persistence (schemas, conversation/triage records)
- Redis session cache (TTL 3600s)
- WebSocket API for mobile app integration
- CLI runner with mock LLM mode

### TODO (Phase 4+)
- Langfuse observability integration
- spaCy-based medical NER (currently keyword-only)
- RAG vector DB for hospital protocols
- Real-time department wait times
- React Native mobile app

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/session/{id}` | Get session status |
| WS | `/ws/triage` | WebSocket for real-time triage conversations |

### WebSocket Protocol

```json
// Client → Server
{"type": "message", "content": "I have chest pain", "language": "en"}

// Server → Client
{"type": "response", "content": "...", "session_id": "...", "metadata": {...}}
{"type": "emergency", "content": "...", "metadata": {"emergency_type": "cardiac"}}
```

## Tech stack

| Layer | Tech |
|-------|------|
| Orchestration | LangGraph (StateGraph + PostgresSaver checkpointing) |
| LLM | MiniCPM5-1B via SGLang (`tool_call_parser="minicpm5"`) |
| API | FastAPI + WebSocket |
| Session cache | Redis (TTL 3600s) |
| Primary DB | PostgreSQL |
| Knowledge | Vector DB (RAG for hospital protocols) |
| Observability | Langfuse |

## Conventions

- Deterministic nodes: pure Python functions, no LLM calls
- LLM nodes: prompts must be short, structured JSON output, explicit "Do NOT diagnose" constraints
- Checkpointing: save after every node completion (PostgresSaver)
- Retry: 1 retry for JSON parse failures, 3 retries with exponential backoff for transient tool errors
- Emergency messages are pre-written templates, never generated by the model
- Symptom extraction uses spaCy/regex NER, not the LLM
