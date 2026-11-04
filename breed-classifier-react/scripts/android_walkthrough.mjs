/**
 * Drive the installed Android WebView over CDP and capture a UI walkthrough.
 * Run from breed-classifier-react/ where the `ws` package is installed.
 */
import fs from "node:fs";
import path from "node:path";
import http from "node:http";
import WebSocket from "ws";

const OUT = path.resolve(process.env.WALKTHROUGH_DIR ||
  "../outputs/screenshots/android_walkthrough");
const SAMPLE = path.resolve(process.env.SAMPLE_IMAGE ||
  "../data/external/atharvadarpude_buffalo/buffalo/banni/1.jpg");

function getJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let data = "";
      res.on("data", (chunk) => { data += chunk; });
      res.on("end", () => {
        try { resolve(JSON.parse(data)); } catch (err) { reject(err); }
      });
    }).on("error", reject);
  });
}

const targets = await getJson("http://127.0.0.1:9222/json");
const target = targets.find((item) => item.type === "page");
if (!target) throw new Error("No WebView page target found on :9222");

const socket = new WebSocket(target.webSocketDebuggerUrl);
let nextId = 1;
const pending = new Map();

socket.on("message", (raw) => {
  const message = JSON.parse(raw.toString());
  if (!message.id) return;
  const entry = pending.get(message.id);
  if (!entry) return;
  pending.delete(message.id);
  if (message.error) entry.reject(new Error(message.error.message));
  else entry.resolve(message.result);
});

await new Promise((resolve, reject) => {
  socket.once("open", resolve);
  socket.once("error", reject);
});

function send(method, params = {}) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(script) {
  const result = await send("Runtime.evaluate", {
    expression: script,
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || "Runtime.evaluate failed");
  }
  return result.result?.value;
}

async function waitFor(expression, timeoutMs = 30000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (await evaluate(`Boolean(${expression})`)) return true;
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  throw new Error(`Timed out waiting for: ${expression}`);
}

async function screenshot(name) {
  await new Promise((resolve) => setTimeout(resolve, 350));
  const result = await send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: false,
  });
  const file = path.join(OUT, name);
  fs.writeFileSync(file, Buffer.from(result.data, "base64"));
  console.log("saved", file);
}

await send("Page.enable");
await send("Runtime.enable");
fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });

// 1. Home / input
await evaluate("window.scrollTo(0,0); 'ok'");
await waitFor("document.querySelector('.android-source-grid')");
await screenshot("01_home_input.png");

// 2. Settings drawer (top)
await evaluate(`
  document.querySelector('.android-menu-btn')?.click();
  'ok'
`);
await waitFor("document.querySelector('.settings-drawer')");
await screenshot("02_settings_online_theme.png");

// 3. Theme + language section
await evaluate(`
  (() => {
    const drawer = document.querySelector('.settings-drawer');
    const themes = document.querySelector('.settings-theme-grid');
    if (drawer && themes) drawer.scrollTop = themes.offsetTop - 70;
    return 'ok';
  })()
`);
await screenshot("03_settings_themes_language.png");

// 4. Advanced settings: model + thresholds
await evaluate(`
  document.querySelector('.settings-accordion-trigger')?.click();
  'ok'
`);
await waitFor("document.querySelector('.settings-advanced')");
await evaluate(`
  (() => {
    const drawer = document.querySelector('.settings-drawer');
    const advanced = document.querySelector('.settings-advanced');
    if (drawer && advanced) drawer.scrollTop = advanced.offsetTop - 80;
    return 'ok';
  })()
`);
await screenshot("04_settings_advanced_ood.png");

await evaluate(`
  (() => {
    const drawer = document.querySelector('.settings-drawer');
    if (drawer) drawer.scrollTop = drawer.scrollHeight;
    return 'ok';
  })()
`);
await screenshot("05_settings_advanced_models.png");

// 5. Close settings and prepare an uploaded gallery image
await evaluate(`
  (() => {
    const row = Array.from(document.querySelectorAll('.control-setting-row'))
      .find((item) => item.textContent.includes('Online Mode'));
    const toggle = row?.querySelector('input[type=checkbox]');
    if (toggle && toggle.checked) toggle.click();
    return 'ok';
  })()
`);
await evaluate("document.querySelector('.settings-close')?.click(); 'ok'");
await waitFor("!document.querySelector('.settings-drawer')");

const sampleB64 = fs.readFileSync(SAMPLE).toString("base64");
await evaluate(`
  (async () => {
    const b64 = ${JSON.stringify(sampleB64)};
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const file = new File([bytes], 'sample.jpg', { type: 'image/jpeg' });
    const input = Array.from(document.querySelectorAll('input[type=file]'))
      .find((el) => (el.accept || '').includes('.png'));
    if (!input) throw new Error('upload input not found');
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  })()
`);
await waitFor("document.querySelector('.android-preview-grid')");
await screenshot("06_upload_preview_pad_auto.png");

// 6. Center crop
await evaluate(`
  Array.from(document.querySelectorAll('.android-option-row button'))
    .find((b) => b.textContent.includes('Center Crop'))?.click();
  'ok'
`);
await waitFor("document.querySelector('.android-preview-grid')");
await screenshot("07_upload_center_crop.png");

// 7. Analyze and capture result
await evaluate(`
  Array.from(document.querySelectorAll('button'))
    .find((b) => b.textContent.includes('Analyze Breed'))?.click();
  'ok'
`);
await waitFor("document.querySelector('.android-results')", 60000);
await evaluate("window.scrollTo(0,0); 'ok'");
await screenshot("09_result_top.png");
await evaluate("window.scrollTo(0, document.body.scrollHeight); 'ok'");
await screenshot("10_result_bottom.png");

// 9. Offline breed-tips notice
await evaluate(`
  Array.from(document.querySelectorAll('button'))
    .find((b) => b.textContent.includes('Breed Tips'))?.click();
  'ok'
`);
await waitFor("document.querySelector('.android-toast')");
await screenshot("11_breed_tips_offline_notice.png");

// 10. Recent scans on the home page
await waitFor("!document.querySelector('.android-toast')", 10000);
await evaluate(`
  Array.from(document.querySelectorAll('button'))
    .find((b) => b.textContent.includes('Identify another animal'))?.click();
  'ok'
`);
await waitFor("document.querySelector('.android-recent')");
await screenshot("12_home_recent_scans.png");

// 11. Guided camera with the cattle silhouette overlay
await evaluate(`
  Array.from(document.querySelectorAll('button'))
    .find((b) => b.textContent.includes('Take Photo'))?.click();
  'ok'
`);
await waitFor("document.querySelector('.camera-modal')");
try {
  await waitFor("document.querySelector('.camera-silhouette')", 8000);
} catch { /* camera may be unavailable on the desktop browser */ }
await screenshot("13_camera_guided_capture.png");
await evaluate("document.querySelector('.camera-close-icon')?.click(); 'ok'");
await waitFor("!document.querySelector('.camera-modal')");

// 12. Settings with Online Mode enabled (still shows offline/online badge)
await evaluate("window.scrollTo(0,0); 'ok'");
await evaluate(`
  document.querySelector('.android-menu-btn')?.click();
  'ok'
`);
await waitFor("document.querySelector('.settings-drawer')");
await evaluate(`
  (() => {
    const toggle = Array.from(document.querySelectorAll('.control-setting-row'))
      .find((row) => row.textContent.includes('Online Mode'))
      ?.querySelector('input[type=checkbox]');
    if (toggle && !toggle.checked) toggle.click();
    return 'ok';
  })()
`);
await screenshot("14_settings_online_enabled.png");

fs.writeFileSync(path.join(OUT, "README.md"), `# Android UI walkthrough

Captured from the installed Capacitor WebView over Chrome DevTools Protocol.

| File | Screen |
|---|---|
| 01_home_input.png | Single-portal home: gallery upload / guided camera |
| 02_settings_online_theme.png | Settings drawer: online mode + backend URL |
| 03_settings_themes_language.png | 4 theme selector + language |
| 04_settings_advanced_ood.png | Advanced: OOD toggle, profile and thresholds |
| 05_settings_advanced_models.png | Advanced: on-device model, upload + OOD JSON |
| 06_upload_preview_pad_auto.png | Uploaded gallery preview: Pad to Square + Auto |
| 07_upload_center_crop.png | Uploaded gallery preview: Center Crop |
| 09_result_top.png | Result: Species, Breed + Top-3, Gender, Age |
| 10_result_bottom.png | Result: Breed Tips button and model metadata |
| 11_breed_tips_offline_notice.png | Offline Mode toast for Breed Tips |
| 12_home_recent_scans.png | Recent scans persisted on the home page |
| 13_camera_guided_capture.png | Guided camera with cattle silhouette + status pills |
| 14_settings_online_enabled.png | Settings with Online Mode enabled |
`);

socket.close();
console.log("walkthrough complete:", OUT);
