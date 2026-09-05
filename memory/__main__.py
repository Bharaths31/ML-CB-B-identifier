import argparse
import sys

from .service import Mem0Layer


def _print_token_stats(stats):
    print(f"  memories retrieved : {stats['memories_retrieved']}")
    print(f"  memories in context: {stats['memories_in_context']}")
    print(f"  context tokens     : {stats['tokens_used']}")
    print(f"  retrieved tokens   : {stats['tokens_retrieved']}")
    print(f"  tokens saved       : {stats['tokens_saved']} "
          f"({stats['saved_pct']}%)")


def demo():
    layer = Mem0Layer(collection="demo_context")
    layer.reset()

    print("== store facts scoped to user 'alice' ==")
    layer.remember("Alice's farm in Kaimoor keeps Sahiwal cows for milk",
                   user_id="alice", run_id="session-1")
    layer.remember("Sahiwal gives ~2200 kg milk per lactation",
                   user_id="alice", run_id="session-1")
    layer.remember("Alice also has Murrah buffalo for sale",
                   user_id="alice", run_id="session-1")
    layer.remember("Alice's son is 10 years old and likes cricket",
                   user_id="alice", run_id="session-1")
    layer.remember("Alice prefers morning milking sessions",
                   user_id="alice", run_id="session-1")
    layer.remember("The vet visits Alice's farm every month",
                   user_id="alice", run_id="session-1")
    print("  stored 5 memories (session-1)\n")

    print("== another user 'bob' is isolated ==")
    layer.remember("Bob breeds kosali, a small disease-resistant breed",
                   user_id="bob", run_id="session-1")
    print("  stored 1 memory (bob/session-1)\n")

    print("== cross-session recall (new run_id, same user) ==")
    stats = layer.build_context("What dairy cows does Alice keep?",
                                user_id="alice", top_k=3)
    print(f"  context:\n{stats['context']}\n")
    _print_token_stats(stats)

    print("\n== cross-agent isolation (agent_id) ==")
    layer.remember("vet-agent note: treat Sahiwal mastitis with X",
                   user_id="alice", agent_id="vet-agent", run_id="session-1")
    hits = layer.recall("Alice's cow health notes", user_id="alice",
                        agent_id="sales-agent")
    print(f"  sales-agent sees {len(hits)} memories "
          f"(vet-agent notes hidden)\n")

    print("== what a model sees per turn ==")
    full = layer.list_memories(user_id="alice")
    baseline = " ".join(m["memory"] for m in full)
    cmp = layer.token_savings(baseline, stats["context"])
    print(f"  naive full-transcript  -> all {len(full)} alice memories "
          f"({cmp['tokens_full_transcript']} tokens), incl. cricket/vet trivia")
    print(f"  memory-layer context   -> only {stats['memories_in_context']} "
          f"relevant ({cmp['tokens_context']} tokens)")
    print(f"  tokens saved per turn  -> {cmp['tokens_saved']} "
          f"({cmp['saved_pct']}%)")

    layer.reset()
    print("\n== demo store cleared ==")


def main():
    parser = argparse.ArgumentParser(
        description="Mem0 context memory layer demo")
    parser.add_argument("--recall", help="recall context for a query")
    parser.add_argument("--add", help="store a raw memory")
    parser.add_argument("--user-id", default="demo-user")
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--run-id", default="session-1")
    parser.add_argument("--list", action="store_true", help="list memories")
    parser.add_argument("--reset", action="store_true", help="reset store")
    args = parser.parse_args()

    layer = Mem0Layer()
    if args.reset:
        layer.reset()
        print("reset ok")
    if args.add:
        results = layer.remember(args.add, user_id=args.user_id,
                                 agent_id=args.agent_id, run_id=args.run_id)
        for r in results:
            print(f"{r['event']} {r['id'][:8]} {r['memory']}")
    if args.recall:
        stats = layer.build_context(args.recall, user_id=args.user_id,
                                    agent_id=args.agent_id, run_id=args.run_id)
        print("context:")
        print(stats["context"] or "  (none)")
        _print_token_stats(stats)
    if args.list:
        for m in layer.list_memories(user_id=args.user_id,
                                     agent_id=args.agent_id,
                                     run_id=args.run_id):
            print(f"{m['id'][:8]} {m['memory']}")
    if not any((args.reset, args.add, args.recall, args.list)):
        demo()


if __name__ == "__main__":
    sys.exit(main())
