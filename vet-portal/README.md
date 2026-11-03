# BreedID — Veterinary Review Portal (standalone)

The expert escalation queue is **not part of the farmer-facing app**. It runs as
its own Vite application on **port 5174** and shares the backend at port 5000.

## Run

```bash
# terminal 1 — backend + farmer app
python run.py

# terminal 2 — vet portal (separate page)
cd vet-portal
npm install        # once
npm run dev        # http://localhost:5174
```

The **Developer Settings** page (`/dev`) has an **“Vet Portal ↗”** button that
opens this app. Override the URL with:

```bash
# .env.local in breed-classifier-react/
VITE_VET_PORTAL_URL=http://localhost:5174
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VITE_VET_PORTAL_URL` | `http://localhost:5174` | Link used by the dev menu |
| `VITE_MAIN_APP_URL` | `http://localhost:5173` | “Back to user app” link |
| `VITE_DEV_APP_URL` | `http://localhost:5173/dev` | “Developer settings” link |

## What it shares

API client, socket hook, i18n context, theme switcher and styles are imported
from `../breed-classifier-react/src/` (Vite resolves React from the main app to
avoid duplicate copies). The escalation workflow itself is unchanged:
pending / verified / rejected queues, breed verification, clinical notes and
promotion into the active-learning dataset.

Build with `npm run build` inside `vet-portal/`; the static output is
`vet-portal/dist/` and can be served by any web server on the same origin as the
API.
