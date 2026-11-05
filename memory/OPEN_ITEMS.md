# Open Items

## 1. Vercel MCP authentication

Status: blocked on a user OAuth click.

`opencode mcp list` reports:

```text
⚠ vercel  needs authentication
```

Action: run `/mcps` in OpenCode, select `vercel`, sign in. Do not commit any
tokens. `opencode mcp auth vercel` was attempted; it waited for browser OAuth and
timed out.

## 2. Goat / non-cattle animal acceptance

Status: known trade-off.

The top-k relaxation fixed cattle false rejections but now accepts the goat test
image as “Mewati”. This is not a threshold-tuning problem: the model’s logits for
that goat overlap the cattle range, and the selfie gate does not fire because
eye/brow density is low.

Possible future work:
- add a lightweight animal-class / ruminant-shape check that is not silhouette
  segmentation;
- train or attach a small “cattle/buffalo vs other animal” classifier;
- use a model with better OOD separation;
- add a curated negative-animal test set and calibrate against it.

Do not tighten the existing entropy/top-k thresholds blindly; that reintroduces
real cattle false rejections.

## 3. FaceDetector fallback

Status: expected limitation.

`FaceDetector` is unavailable in the Electron harness and the Android WebView used
for verification. The YCbCr fallback is the effective path. A future Android
WebView/Chrome version may expose FaceDetector; the code already tries it.

## 4. Advisory API aliases

Status: functional but confusing.

Both work:

```text
https://advisory-api-bharaths31s-projects.vercel.app
https://advisory-api-rho.vercel.app
```

The app uses the canonical team URL. The short alias was transient during the
404 incident. If aliases are cleaned up, update `DEFAULT_ADVISORY_API_URL` and
the docs together.

## 5. Vercel deploy ergonomics

Status: workaround documented.

The project Root Directory is `advisory-api`, and the Git integration is the
tested deploy path. Running `vercel deploy` from inside `advisory-api` can fail
with “Root Directory does not exist”; running it from the repo root uploads large
artifacts. Prefer `git push`.

## 6. APK is a debug build

Status: intentional.

`artifacts/android/breedid-android-debug.apk` is a debug build. A release build
still needs signing, versioning and Play Store/enterprise distribution decisions.

## 7. Age/gender override

Status: intentionally not implemented.

The earlier product requirement said gender/age are read-only. A later request
mentioned a one-click override pill, but it was not added. If the product
decision changes, add it explicitly and update the result UI and validation.

## 8. Advisory API observability

Status: minimal.

There is no analytics or error tracking. Vercel function logs are available via
`vercel logs`/dashboard. For production, add:
- request metrics;
- 4xx/5xx alerts;
- optional `lang`/breed usage counts;
- response-time monitoring.

## 9. Automated guardrail tests

Status: browser harness only.

The guardrail was verified manually in a Vite dev page with prepared canvases.
There is no committed unit test harness for canvas/DOM behavior. A future test
could use Playwright or a small browser integration test with fixture images.

## 10. Advisory data freshness

Status: manual.

The API data is generated from `data/breed_knowledge.json` and committed. If the
curated source changes, run `npm run build:data`, review the diff, test and
redeploy. There is no scheduled refresh.
