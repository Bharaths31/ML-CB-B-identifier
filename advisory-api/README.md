# BreedID Advisory API (Vercel)

A free, dependency-free Vercel Node.js serverless API that serves breed tips and
farmer advisory cards for the 68 BreedID classes (50 cattle + 18 buffalo). It
uses only static JSON and pure functions, so it fits comfortably inside the
Vercel free tier.

## Project structure

```text
advisory-api/
├─ api/
│  └─ advisory.js              # GET /api/advisory — CORS, validation, response
├─ data/
│  └─ breed_knowledge.json     # 68 breeds: origin, utility, milk, markers, price, schemes
├─ lib/
│  ├─ i18n.js                  # en / ta / hi copy, card logic, price/scheme formatting
│  └─ knowledge.js             # lookup, aliases, species normalization
├─ scripts/
│  ├─ build-data.js            # regenerate data/breed_knowledge.json from project data
│  └─ dev-server.js            # local HTTP server on port 3001
├─ test/
│  └─ advisory.test.js         # node:test suite (no dependencies)
├─ package.json
└─ vercel.json
```

## API

```http
GET /api/advisory?species=Cattle&breed=gir&gender=Female&age=Adult&lang=en
```

| Parameter | Values | Notes |
|---|---|---|
| `species` | `Cattle`, `Buffalo` (case-insensitive) | also accepts `cow` / `buffalo` |
| `breed` | slug, e.g. `gir`, `murrah`, `bhadwari` | spaces/hyphens and aliases are normalized |
| `gender` | `Female`, `Male`, `Unknown` | `Female` + `Adult` shows milk advice |
| `age` | `Calf`, `Young`, `Adult` | `Calf` hides milk and draught advice |
| `lang` | `en`, `ta`, `hi` | defaults to `en` |

The function returns:

- `breed`: localized name, origin, utility, milk yield, physical markers, market
  price and localized scheme names.
- `advisory`: context-specific title, summary, cards, vaccination schedule
  (calf/young), scheme list and disclaimer.
- `meta`: breed counts, language, source timestamp and latency.

### Logic summary

- **Female + Adult:** lactation diet, daily milk/fat, calcium/minerals, mastitis
  prevention and heat/AI advice. Milk yield is visible.
- **Male + Adult:** work nutrition, draught power, breeding value and bull
  health. Milk yield is hidden; draught power is visible.
- **Calf (< 1 year):** colostrum, calf starter, deworming, FMD/brucellosis
  vaccination and calf housing. Milk yield and draught power are hidden.
- **Young (1–3 years):** growth diet, first breeding, deworming and routine
  health.
- **Missing gender:** general adult care.

CORS is `Access-Control-Allow-Origin: *`; `OPTIONS` returns `204`. Successful
responses are cached with `s-maxage=86400, stale-while-revalidate=604800`.

### Example

```bash
curl "https://YOUR-PROJECT.vercel.app/api/advisory?species=Cattle&breed=gir&gender=Female&age=Adult&lang=ta"
```

## Local development

```powershell
cd advisory-api
npm test
npm run dev
# open http://localhost:3001/api/advisory?species=Cattle&breed=gir&gender=Female&age=Adult&lang=en
```

Regenerate the compact data file after changing the project knowledge base:

```powershell
npm run build:data
```

## Deploy to Vercel (free)

Run from the `advisory-api` folder:

```powershell
cd advisory-api
npm i -g vercel
vercel login
vercel
```

Accept the defaults. The first deploy is a preview; promote it to production:

```powershell
vercel --prod
```

Vercel prints the live URL, for example:

```text
https://breedid-advisory.vercel.app
```

Your endpoint is then:

```text
https://breedid-advisory.vercel.app/api/advisory
```

No environment variables, database, model, or paid add-on is required. The only
runtime dependency is Node.js 20+ (provided by Vercel).

## Android / Capacitor integration

The client lives at
`breed-classifier-react/src/services/onlineAdvisory.js`. It:

- returns a plain fallback object when `navigator.onLine === false`;
- aborts after **5 seconds** with `AbortController`;
- never throws, so the Android UI cannot crash on a rural network drop;
- normalizes a full endpoint or a project base URL.

Usage:

```js
import { fetchAdvisoryData } from "./services/onlineAdvisory.js";

const tips = await fetchAdvisoryData(
  result.species,          // "Cattle" | "Buffalo"
  result.breed_slug,       // "gir", "murrah"
  gender.key,              // "Female" | "Male" | "Unknown"
  age.key,                 // "Calf" | "Young" | "Adult"
  settings.language,       // "en" | "ta" | "hi"
  "https://breedid-advisory.vercel.app/api/advisory",
);

if (tips.ok) {
  // tips.breed, tips.advisory.cards, tips.advisory.vaccination_schedule
} else if (tips.offline) {
  // show tips.message; on-device breed result is still valid
} else if (tips.timeout) {
  // show tips.message; allow retry
}
```

Set the default in `breed-classifier-react/.env` (optional):

```text
VITE_ADVISORY_API_URL=https://breedid-advisory.vercel.app/api/advisory
```

## Data notes

The knowledge base is generated from the project's curated
`data/breed_knowledge.json` (NBAGR/ICAR breed descriptions and Indian feeding
norms). Yields, prices and vaccination timings are typical ranges; the API adds
a localized "confirm locally" disclaimer to every response.
