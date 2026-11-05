# Agent Memory

This folder is the durable, git-tracked memory for work on the BreedID livestock
vision project. It records what changed, why it changed, how it was verified, and
what remains open.

It is written for a future coding agent or maintainer who has none of the current
chat context. Read `CURRENT_STATE.md` first, then the relevant topic file.

## Index

| File | Purpose |
|---|---|
| `CURRENT_STATE.md` | Live URLs, APK metadata, test status, environment, caveats |
| `PROGRESS.md` | Chronological progress and commit map |
| `DECISIONS.md` | Architecture/decision log with rationale and trade-offs |
| `SESSION_2026-11-05.md` | Detailed narrative of the 2026-11-05 session |
| `RUNBOOK.md` | Commands for build, test, APK refresh, Vercel deploy and device checks |
| `GUARDRAILS.md` | On-device input guardrail design, thresholds and calibration results |
| `API.md` | Vercel advisory API contract, data model and app integration |
| `OPEN_ITEMS.md` | Known gaps, risks and next actions |

## Maintenance rules

- Update `CURRENT_STATE.md` whenever a deploy, APK refresh or test result changes.
- Add a dated session file for substantial multi-step work.
- Add a decision entry when a threshold, architecture or deployment choice changes.
- Never store secrets, Vercel tokens, OAuth tokens, device serial numbers or user
  personal data in this folder.
- Do not use this folder as a second changelog; keep it focused on context that
  is hard to reconstruct from `git log` alone.

## Source of truth

The code and tests remain authoritative. This folder explains intent and history;
if it conflicts with the code, trust the code and update this folder.
