import os
import threading
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "false")

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EMBEDDER_MODEL = "all-MiniLM-L6-v2"
PLACEHOLDER_KEY = "sk-local-placeholder"
PLACEHOLDER_MODEL = "unused"


class Mem0Layer:
    """Mem0-backed context memory layer.

    Stores durable, retrievable memories keyed by user / agent / session
    (run_id) and turns them into a small, token-bounded context block so
    only the relevant memories are injected into a prompt instead of a
    growing full transcript. Works across sessions (new run_id keeps the
    user's memories), across agents (agent_id isolation) and across models
    (the recalled context is model-agnostic; the LLM provider is swappable
    via environment variables).
    """

    def __init__(self, store_dir=None, collection="breed_context",
                 embedder_model=DEFAULT_EMBEDDER_MODEL, env=None):
        self.store_dir = Path(store_dir or ROOT / "outputs" / "memory")
        self.collection = collection
        self.embedder_model = embedder_model
        self._env = dict(os.environ if env is None else env)
        self._memory = None
        self._lock = threading.Lock()
        self._op_lock = threading.Lock()

    @property
    def llm_provider(self):
        return self._env.get("MEM0_LLM_PROVIDER", "openai")

    @property
    def llm_api_key(self):
        return (self._env.get("MEM0_LLM_API_KEY")
                or self._env.get("OPENAI_API_KEY") or "").strip()

    @property
    def llm_base_url(self):
        return (self._env.get("MEM0_LLM_BASE_URL")
                or self._env.get("OPENAI_BASE_URL") or "").strip()

    @property
    def llm_model(self):
        return (self._env.get("MEM0_LLM_MODEL")
                or self._env.get("OPENAI_MODEL") or "gpt-4o-mini")

    @property
    def llm_configured(self):
        return bool(self.llm_api_key)

    def _client(self):
        with self._lock:
            if self._memory is None:
                from mem0 import Memory
                self.store_dir.mkdir(parents=True, exist_ok=True)
                llm_cfg = {
                    "model": self.llm_model if self.llm_configured else PLACEHOLDER_MODEL,
                    "api_key": self.llm_api_key or PLACEHOLDER_KEY,
                }
                if self.llm_base_url:
                    llm_cfg["base_url"] = self.llm_base_url
                config = {
                    "vector_store": {
                        "provider": "chroma",
                        "config": {"collection_name": self.collection,
                                   "path": str(self.store_dir)},
                    },
                    "llm": {"provider": self.llm_provider, "config": llm_cfg},
                    "embedder": {
                        "provider": "huggingface",
                        "config": {"model": self.embedder_model},
                    },
                    "history_db_path": str(self.store_dir / "history.db"),
                }
                self._memory = Memory.from_config(config)
            return self._memory

    @staticmethod
    def estimate_tokens(text):
        return max(1, len(text) // 4)

    @staticmethod
    def token_savings(full_text, context_text):
        full = Mem0Layer.estimate_tokens(full_text)
        used = Mem0Layer.estimate_tokens(context_text)
        return {
            "tokens_full_transcript": full,
            "tokens_context": used,
            "tokens_saved": max(0, full - used),
            "saved_pct": round(100 * (1 - used / full), 1) if full else 0.0,
        }

    def _filters(self, user_id=None, agent_id=None, run_id=None):
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id
        return filters or None

    def remember(self, text, *, user_id=None, agent_id=None, run_id=None,
                 metadata=None, infer=None):
        """Store a memory. infer=True (default when an LLM is configured)
        lets mem0 extract, merge and retire facts with the LLM; infer=False
        stores the raw text without any LLM."""
        if infer is None:
            infer = self.llm_configured
        with self._op_lock:
            result = self._client().add(
                text, user_id=user_id, agent_id=agent_id, run_id=run_id,
                metadata=metadata, infer=infer)
        return result.get("results", [])

    def recall(self, query, *, user_id=None, agent_id=None, run_id=None,
               top_k=5, threshold=0.1):
        """Semantic search for memories relevant to query within the given
        scopes. This is what avoids dumping the whole history into a prompt."""
        with self._op_lock:
            result = self._client().search(
                query, filters=self._filters(user_id, agent_id, run_id),
                top_k=top_k, threshold=threshold)
        return result.get("results", [])

    def build_context(self, query, *, user_id=None, agent_id=None, run_id=None,
                      top_k=5, threshold=0.1, max_tokens=600,
                      per_memory_tokens=160):
        """Return a compact context block (bounded by max_tokens) plus token
        accounting showing how much prompt space was saved versus using every
        stored memory."""
        hits = self.recall(query, user_id=user_id, agent_id=agent_id,
                           run_id=run_id, top_k=top_k, threshold=threshold)
        retrieved_tokens = self.estimate_tokens(
            " ".join(h["memory"] for h in hits)) if hits else 0
        block = []
        used_tokens = 0
        for hit in hits:
            text = hit["memory"]
            if len(text) > per_memory_tokens * 4:
                text = text[:per_memory_tokens * 4 - 3] + "..."
            tokens = self.estimate_tokens(text)
            if used_tokens + tokens > max_tokens:
                break
            block.append(text)
            used_tokens += tokens
        context = "\n".join(block)
        return {
            "query": query,
            "context": context,
            "memories_retrieved": len(hits),
            "memories_in_context": len(block),
            "tokens_used": used_tokens,
            "tokens_retrieved": retrieved_tokens,
            "tokens_saved": max(0, retrieved_tokens - used_tokens),
            "saved_pct": round(100 * (1 - used_tokens / retrieved_tokens), 1)
            if retrieved_tokens else 0.0,
        }

    def chat(self, message, *, user_id=None, agent_id=None, run_id=None,
             system_hint=None, top_k=5, max_tokens=600, persist=True):
        """Reply with only the recalled memories injected as context. Requires
        an OpenAI-compatible LLM (OPENAI_API_KEY or MEM0_LLM_* env vars)."""
        if not self.llm_configured:
            raise RuntimeError(
                "No LLM configured. Set OPENAI_API_KEY (+ OPENAI_BASE_URL for "
                "local/OpenAI-compatible endpoints) or MEM0_LLM_* to enable "
                "memory-aware chat. remember/recall work without an LLM.")
        context = self.build_context(
            message, user_id=user_id, agent_id=agent_id, run_id=run_id,
            top_k=top_k, max_tokens=max_tokens)
        from litellm import completion
        system = (system_hint or
                  "You are a helpful assistant for cattle & buffalo breed "
                  "recognition. Use the relevant memories below; if none are "
                  "relevant, say so and answer from your own knowledge.")
        if context["context"]:
            system += "\n\nRelevant memories:\n" + context["context"]
        else:
            system += "\n\nRelevant memories: none found."
        response = completion(
            model=self.llm_model, api_key=self.llm_api_key,
            **({"api_base": self.llm_base_url} if self.llm_base_url else {}),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ])
        reply = response.choices[0].message.content
        if persist:
            for text in (f"user asked: {message}", f"assistant said: {reply}"):
                self.remember(text, user_id=user_id, agent_id=agent_id,
                              run_id=run_id)
        return {"reply": reply, "context": context}

    def list_memories(self, *, user_id=None, agent_id=None, run_id=None,
                      top_k=200):
        with self._op_lock:
            result = self._client().get_all(
                filters=self._filters(user_id, agent_id, run_id), top_k=top_k)
        return result.get("results", [])

    def delete(self, memory_id):
        with self._op_lock:
            self._client().delete(memory_id)

    def clear(self, *, user_id=None, agent_id=None, run_id=None):
        with self._op_lock:
            return self._client().delete_all(user_id=user_id,
                                             agent_id=agent_id, run_id=run_id)

    def reset(self):
        with self._op_lock:
            self._client().reset()

    def status(self, *, user_id=None, agent_id=None, run_id=None):
        try:
            count = len(self.list_memories(user_id=user_id, agent_id=agent_id,
                                           run_id=run_id))
        except ValueError:
            count = None
        return {
            "store_dir": str(self.store_dir),
            "collection": self.collection,
            "embedder_model": self.embedder_model,
            "llm_configured": self.llm_configured,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model if self.llm_configured else None,
            "llm_base_url": self.llm_base_url or None,
            "memories": count,
            "memories_scoped": bool(user_id or agent_id or run_id),
        }
