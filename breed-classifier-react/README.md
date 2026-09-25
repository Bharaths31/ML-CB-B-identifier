# 🐄 Breed Classifier — React Frontend

User-centric React UI for the breed-classifier model, backed by `test_model.py`.

## Features

**User Mode (default):**
- 📸 Single image — drag & drop, browse, or **live camera capture**
- 📁 Batch upload with progress and summary stats
- Clean breed classification results with animated top-5 confidence bars

**Dev Mode (header toggle):**
- 🛠️ Dev Tools tab: model spec inspector, class maps viewer, image metadata
  analyzer, presenter-view configurator, session logs
- 📄 ODT export for single & batch results (mirrors the `--dev` backend)
- Raw species confidence percentages

## Setup

1. Start the model backend (from your project root):

   ```bash
   python test_model.py            # serves http://localhost:8501
   ```

2. Install and run the React app:

   ```bash
   cd breed-classifier-react
   npm install
   npm run dev                     # http://localhost:5173
   ```

Vite proxies `/api/*` to `http://localhost:8501`, so no CORS setup is needed.

> Note: ODT export and dev-info endpoints are only enabled when the backend
> runs in its default dev mode (not `--present`).

## Camera capture

`getUserMedia` requires a secure context — `http://localhost` qualifies, so
camera capture works out of the box in development. For production, serve the
app over HTTPS.
