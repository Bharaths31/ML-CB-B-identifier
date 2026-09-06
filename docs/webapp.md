# 8. Web Application

The project includes a lightweight, built-in **FastAPI** web application with a vanilla JavaScript frontend. It is designed to be a complete dashboard for your classification model, allowing you to run predictions, trigger training jobs, evaluate the model, and interact with the optional memory layer.

---

## Architecture Overview

The webapp lives entirely in the `webapp/` directory:
- `server.py`: The FastAPI backend handling REST API endpoints.
- `static/index.html`: A single-page application (SPA) providing the user interface.
- `static/app.js`: Vanilla JS logic (no framework like React or Vue required).
- `static/style.css`: Styling and responsive design.

The architecture is built for simplicity. When a long-running process (like training or evaluation) is triggered, the backend uses a `JobRunner` component that spawns a subprocess (e.g., `python -m src.train`). It captures the stdout and stderr in real-time, allowing the frontend to poll for progress updates (roughly every 1.2 seconds) and display dynamic progress bars.

---

## Starting the Web Server

To start the server, activate your virtual environment and run the main entry point:

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Run the server
python webapp/server.py
```

By default, the server listens on `http://localhost:8000`. Open this URL in your web browser to access the dashboard.

---

## Dashboard Features

The webapp is organized into tabs, each serving a distinct workflow:

### 1. Predict
Upload any image (Cattle or Buffalo) to test the active model.
- The UI fetches the latest model checkpoint from `outputs/checkpoints/`.
- Displays the predicted species (Cattle or Buffalo).
- Displays the predicted breed.
- Shows confidence scores.

> **Note on Model Cache**: The `ModelBox` class in the backend caches the model weights in memory. If a new training job completes and overwrites the checkpoint, `ModelBox` detects the modified timestamp (mtime) and automatically invalidates the cache, loading the fresh model on the next prediction.

### 2. Train
Launch a training job directly from your browser.
- Uses `JobRunner` to spawn `python -m src.train`.
- Real-time progress bars and loss metrics are extracted from the subprocess standard output and displayed dynamically.
- Automatically stops polling when the job signals completion.

### 3. Evaluate
Run the evaluation suite on the current checkpoint.
- Spawns `python -m src.evaluate`.
- Generates detailed confusion matrices and per-class metrics.

### 4. Memory
Interact with the Mem0 vector store.
- A chat-like interface to store facts, retrieve context, or query the LLM memory.
- Useful if you are extending the classifier with an AI assistant.

### 5. Debug
Technical details for developers.
- View raw subprocess logs.
- System information (CUDA availability, RAM, CPU).
- Inspect currently stored checkpoints and exports in the `outputs/` directory.

---

## API Endpoints

The webapp can also be used headlessly via its REST API. Some of the core endpoints include:

- `POST /api/predict`: Upload an image file for classification.
- `POST /api/train/start`: Start a training job.
- `GET /api/train/status`: Get current training progress and logs.
- `POST /api/evaluate/start`: Start an evaluation job.
- `GET /api/evaluate/status`: Get evaluation progress.

For a full list of payloads and schemas, refer to the [API Reference](api-reference.md).
