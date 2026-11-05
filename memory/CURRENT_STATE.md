# Current State

Last updated: 2026-11-05
Repository: `C:\Users\shiva304\Documents\ML-CB-B-identifier`
Branch: `main`, clean and synced with `origin/main`

## Headline

- Android app: simple/developer mode, native camera input, direct `260×260`
  downscale, dynamic age/gender, center-focused guardrail and selfie blocker.
- Online tips: live Vercel advisory API wired into the app.
- Tracked APK: current; installed and verified on the connected Android device.
- Vercel: public landing page and API, SSO protection disabled, root directory
  set to `advisory-api`.

## Live URLs

Landing page:

```text
https://advisory-api-bharaths31s-projects.vercel.app/
```

Canonical API:

```text
https://advisory-api-bharaths31s-projects.vercel.app/api/advisory
```

Legacy/short alias, also working:

```text
https://advisory-api-rho.vercel.app/api/advisory
```

Example:

```text
https://advisory-api-bharaths31s-projects.vercel.app/api/advisory?species=Cattle&breed=gir&gender=Female&age=Adult&lang=en
```

Verified: root `200 text/html`, API `200 application/json`, CORS `*`.

## Vercel project settings

| Setting | Value |
|---|---|
| Project | `advisory-api` |
| Scope/team | `bharaths31s-projects` |
| Root directory | `advisory-api` |
| SSO deployment protection | Disabled |
| Function | `api/advisory.js` |
| Runtime dependency | none (static JSON + Node built-ins) |
| Landing page | `public/index.html` |
| Default app endpoint | canonical URL above |

## APK metadata

```text
artifacts/android/breedid-android-debug.apk
size:   34,310,518 bytes
sha256: F01BC06D7C88412AE4265723901E80F09AD6A421FD59F883D9DEF0E81BDC6850
commit: 4c60daa
built:  2026-11-05T14:41:28+05:30
```

`artifacts/android/build-info.json` and `SHA256SUMS.txt` match the artifact.
The APK was installed on the connected iQOO I2302 (Android 16) and the live Breed
Tips flow was verified there.

Latest commit:

```text
edc4590 chore: refresh APK for online Breed Tips fix
```

## Verification status

| Check | Result |
|---|---|
| App build (`npm run build`) | Pass |
| App unit tests | Pass |
| i18n parity | 643/643 keys in en/ta/hi |
| Advisory API tests | 7/7 pass |
| Live API root + endpoint | 200 |
| Live Breed Tips on device | Pass |
| Guardrail test set | 34 cattle accepted; human selfie + monitor rejected |
| Device install | Pass |
| git status | Clean, `main...origin/main` |

## Known caveats

- The goat test image is accepted after the top-k loosening. This is a known
  trade-off and an open item; see `OPEN_ITEMS.md`.
- Browser `FaceDetector` is unavailable in the Electron harness and the Android
  WebView used here, so the YCbCr fallback is the path actually exercised.
- Vercel MCP is configured but still needs user OAuth (`/mcps`).
- The tracked APK is a debug build.
- Gender and age remain read-only; no override pill was added because the earlier
  product requirement was explicitly read-only.

## Do not commit

- `.env.local` at repository root (contains a Vercel OIDC token, created by
  `vercel link`).
- `.vercel/` project link folders.
- `advisory-api/node_modules/` if it is ever created locally.
