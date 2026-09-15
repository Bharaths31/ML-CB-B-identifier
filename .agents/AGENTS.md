# Project Rules — Cattle & Buffalo Breed Classifier

## Context

Before making any changes to this project, read `.agents/knowledge/CONTEXT.md` for the full architecture, data flow, and implementation details. It contains everything you need to understand the project.

## Code Conventions

- All ML source code lives in `src/` and is run as `python -m src.<module>` (relative imports).
- Model testing GUI is provided via `test_model.py` (custom PyTorch inference browser application).
- Configuration constants are centralized in `src/config.py` — never hardcode paths or hyperparameters elsewhere.
- Training output goes to `outputs/` (checkpoints, exports, metrics).
- Data lives in `data/raw/` (source images) and `data/splits/` (generated CSVs).

## Training

- Always support both CUDA and CPU devices gracefully.
- Use `torch.amp` for mixed precision on CUDA, skip on CPU.
- Smoke tests (`--smoke-test`) must use a real mini-dataset, not batch limits.
- Training auto-exports a portable model bundle after completion.

## Model Testing & Verification

- Interactive testing is run using `python test_model.py`.
- `python -m src.verify` checks backbone loading and forward pass shapes.
- `python -m src.train --smoke-test --skip-qat` is the fast integration test.
