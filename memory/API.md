# Advisory API

Folder: `advisory-api/`
Deployment: Vercel, project `advisory-api`, Root Directory `advisory-api`.
Runtime: Node 20+ (Vercel currently builds with Node 26.x).
Dependencies: none.

## Endpoint

```http
GET /api/advisory?species=Cattle&breed=gir&gender=Female&age=Adult&lang=en
```

| Parameter | Values | Notes |
|---|---|---|
| `species` | `Cattle`, `Buffalo` | aliases `cow`, `buffalo`; case-insensitive |
| `breed` | slug, e.g. `gir`, `murrah`, `bhadwari` | spaces/hyphens normalized; aliases supported |
| `gender` | `Female`, `Male`, `Unknown` | localized labels are accepted |
| `age` | `Calf`, `Young`, `Adult` | localized labels are accepted |
| `lang` | `en`, `ta`, `hi` | unknown falls back to `en` |

Other accepted query aliases: `species_name`, `breed_slug`, `sex`, `age_group`,
`ageGroup`, `language`.

## Response shape

```json
{
  "ok": true,
  "query": { "species": "Cattle", "breed": "gir", "gender": "Female", "age": "Adult", "lang": "en" },
  "breed": {
    "slug": "gir",
    "name": "Gir",
    "local_names": { "en": "Gir", "ta": "கிர்", "hi": "गीर" },
    "species": "Cattle",
    "origin": "...",
    "utility": "Milch (Dairy)",
    "utility_type": "Milch (Dairy)",
    "milk_yield": {
      "daily_liters": "8 - 12 L/day",
      "lactation_yield": "1800 - 3000 kg",
      "fat_percentage": "4.5% - 5.0%"
    },
    "physical_markers": ["..."],
    "feeding": { "...": "..." },
    "climate": { "...": "..." },
    "market_price": "₹45,000 – ₹1,00,000",
    "price_range_inr": [45000, 100000],
    "schemes": [{ "code": "rgm", "name": "...", "description": "..." }],
    "labels": { "...": "..." }
  },
  "advisory": {
    "title": "Lactation nutrition and udder care",
    "summary": "...",
    "context": {
      "gender": "Female",
      "age": "Adult (> 3 years)",
      "milk_yield_visible": true,
      "draught_power_visible": false
    },
    "cards": [
      { "id": "lactation_diet", "title": "...", "body": "..." },
      { "id": "milk_yield", "title": "...", "body": "..." }
    ],
    "vaccination_schedule": { "title": "...", "items": [] },
    "schemes": { "title": "...", "note": "...", "items": [] },
    "disclaimer": "..."
  },
  "meta": {
    "source": "static-breed-knowledge",
    "generated_at": "...",
    "latency_ms": 1,
    "breed_count": 68,
    "counts": { "Cattle": 50, "Buffalo": 18 },
    "supported_languages": ["en", "ta", "hi"],
    "disclaimer": "..."
  }
}
```

`vaccination_schedule` is present for `Calf` and `Young` only.

## Dynamic logic

| Gender / age | Milk yield | Draught power | Cards |
|---|---|---|---|
| Female + Adult | Visible | Hidden | lactation diet, milk/fat, calcium/minerals, mastitis, heat/AI |
| Male + Adult | Hidden | Visible | work nutrition, draught, breeding value, bull health |
| Calf | Hidden | Hidden | colostrum, starter, deworming, vaccination, housing |
| Young | Hidden | Hidden | growth diet, first breeding, deworming, routine health |
| Unknown gender + Adult | Hidden | Hidden | general feed, general health |

Calf/young responses include a vaccination/deworming timeline:
- colostrum, 0–3 days;
- first deworming, 2–3 weeks;
- FMD first dose, 4–6 months;
- brucellosis, 4–8 months for female calves;
- HS/BQ, 6 months.

## CORS and caching

```text
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, OPTIONS
Access-Control-Allow-Headers: Content-Type, Accept
```

- `OPTIONS` → `204`
- `200` → `Cache-Control: public, s-maxage=86400, stale-while-revalidate=604800`
- errors → `no-store`
- `405` for non-GET/HEAD
- `400` for missing species/breed
- `404` for unknown breed with suggestions
- `HEAD` returns `200` with no body

## Data

File: `advisory-api/data/breed_knowledge.json`

- 68 breeds: 50 cattle, 18 buffalo.
- Fields: `origin`, `utility_type`, `milk_yield`, `physical_markers`,
  `feeding`, `climate`, `market_price`, `price_range_inr`, `schemes`,
  `scheme_codes`, `data_confidence`, `note`.
- Source: project `data/breed_knowledge.json` (curated NBAGR/ICAR descriptions
  and Indian feeding norms).
- Regenerate:

```powershell
cd advisory-api
cmd /c "npm run build:data"
```

The extra crossbred classes `holstein_friesian` and `jersey` are excluded from
the 68-breed API.

## Code layout

```text
advisory-api/
├─ api/advisory.js          # handler
├─ data/breed_knowledge.json
├─ lib/i18n.js              # en/ta/hi copy, cards, price/scheme formatting
├─ lib/knowledge.js         # lookup, aliases, species normalization
├─ public/index.html        # live landing page
├─ scripts/build-data.js
├─ scripts/dev-server.js
└─ test/advisory.test.js
```

## Local development

```powershell
cd advisory-api
cmd /c "npm test"
cmd /c "npm run dev"
# http://localhost:3001/api/advisory?species=Cattle&breed=gir&gender=Female&age=Adult&lang=en
```

## Android client

`breed-classifier-react/src/services/onlineAdvisory.js`

```js
fetchAdvisoryData(species, breedSlug, gender, ageGroup, lang, apiUrl)
```

- 5-second `AbortController` timeout.
- `navigator.onLine === false` → offline fallback object.
- Never throws.
- Accepts a full endpoint or a project base URL.
- Default: `https://advisory-api-bharaths31s-projects.vercel.app/api/advisory`.
- Optional env: `VITE_ADVISORY_API_URL`.

UI: `breed-classifier-react/src/components/BreedTipsModal.jsx` renders the live
cards, milk visibility, feeding, climate, vaccination timeline and schemes.
Breed Tips no longer requires the legacy Online Mode or Python backend health;
only internet connectivity.

## Deployment

Preferred: commit and push to `main`; Vercel builds the `advisory-api` root.

Project settings:

```text
Root Directory: advisory-api
SSO protection: disabled
Project: advisory-api
Scope: bharaths31s-projects
```

Live:

```text
https://advisory-api-bharaths31s-projects.vercel.app/
https://advisory-api-bharaths31s-projects.vercel.app/api/advisory
```

Alias: `https://advisory-api-rho.vercel.app` (working, but the canonical team URL
is the app default).
