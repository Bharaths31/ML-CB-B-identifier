# 9. Memory Layer (Mem0)

The **Memory Layer** is an optional module in the `ML-CB-B-identifier` ecosystem designed to augment your interactions with an LLM (Large Language Model) or agent by persisting long-term context. It utilizes **Mem0**, a memory layer for AI assistants.

---

## Core Concept

When integrating this classification model with chat agents (like a veterinary assistant bot), the agent needs to remember previous conversations, user preferences, or specific facts about the predicted breeds. 

The Memory Layer provides a **ChromaDB-backed vector store** to handle this context persistence.

### Directory Structure
```text
memory/                 # Mem0 vector store module
├── __init__.py
└── service.py          # The core MemoryService class
outputs/
└── memory/             # Local ChromaDB SQLite storage (persisted data)
```

---

## Scoping Memories

To ensure privacy and relevance, all memories are strictly scoped. When storing or retrieving a memory, you must provide:
- `user_id`: The ID of the human interacting with the system.
- `agent_id`: The ID of the specific AI agent.
- `run_id`: The session or thread identifier.

This prevents cross-contamination of context between different users or different assistant workflows.

---

## API Endpoints

The FastAPI webapp automatically mounts endpoints for the memory layer (located in `webapp/server.py`). You can test these via the "Memory" tab in the UI, or hit them programmatically:

### 1. Store a Memory
```bash
curl -X POST http://localhost:8000/api/memory \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The user is primarily interested in Gir cattle and lives in Gujarat.",
    "user_id": "user_123",
    "agent_id": "vet_bot",
    "run_id": "session_001"
  }'
```

### 2. Retrieve Relevant Context
When the user asks a question, query the memory to inject context into the LLM prompt:
```bash
curl -X POST http://localhost:8000/api/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which breed was I asking about earlier?",
    "user_id": "user_123",
    "agent_id": "vet_bot",
    "run_id": "session_001",
    "limit": 5
  }'
```
*This returns a list of the most semantically relevant memories based on vector similarity.*

---

## Integrating with the Classifier

A typical pipeline utilizing both the model and the memory layer looks like this:

1. **User uploads an image.**
2. The ML model predicts **Sahiwal cattle**.
3. **Agent queries memory:** `search("User's history with Sahiwal cattle")`.
4. **Memory returns:** `"User previously treated a Sahiwal for a tick infection."`
5. **Agent response:** *"This is a Sahiwal. Do you want me to pull up the tick treatment plan we discussed last month?"*
6. **Agent stores memory:** `store("User uploaded a new photo of their Sahiwal on Sept 6.")`

---

## Configuration

No extra configuration is required. The vector store is created automatically in `outputs/memory/` the first time you invoke the service. Ensure you have the required dependencies (`mem0ai`, `chromadb`) installed via `requirements.txt`.
