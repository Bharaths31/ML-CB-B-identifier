'use strict';

const $ = id => document.getElementById(id);

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML;
}

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

function chip(text, cls) {
  return `<span class="chip ${cls || ''}">${esc(text)}</span>`;
}

function setConsole(el, lines) {
  el.textContent = lines.map(l => l.text).join('\n');
  el.scrollTop = el.scrollHeight;
}

function jobStatusHTML(st) {
  if (!st.running && st.exit === null) return '';
  const parts = [];
  parts.push(st.running ? 'RUNNING' : `DONE exit=${st.exit}`);
  if (st.kind) parts.push('job=' + st.kind);
  if (st.phase != null) parts.push(`phase=${st.phase}`);
  if (st.epoch != null && st.epochs != null) parts.push(`epoch=${st.epoch}/${st.epochs}`);
  const m = st.metrics;
  if (m && Object.keys(m).length > 0) {
    if (m.loss != null) parts.push(`loss=${m.loss.toFixed(4)}`);
    if (m.binary_acc != null) parts.push(`binary=${(m.binary_acc * 100).toFixed(1)}%`);
    if (m.cattle_acc != null) parts.push(`cattle=${(m.cattle_acc * 100).toFixed(1)}%`);
    if (m.buffalo_acc != null) parts.push(`buffalo=${(m.buffalo_acc * 100).toFixed(1)}%`);
    if (m.combined_top1 != null) parts.push(`top1=${(m.combined_top1 * 100).toFixed(1)}%`);
  }
  if (st.best) parts.push(`best(top1)=${(st.best.top1 * 100).toFixed(1)}%`);
  if (st.progress != null && st.progress > 0) parts.push(`${st.progress.toFixed(1)}%`);
  return parts.join('  ');
}

/* ---------- tabs ---------- */

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.panel').forEach(p =>
    p.classList.toggle('active', p.id === 'tab-' + name));
  if (name === 'debug') loadDebug();
  if (name === 'evaluate') loadMetrics();
  if (name === 'memory') { loadMemoryStatus(); listMemories(); }
}

/* ---------- status chips ---------- */

async function loadStatus() {
  try {
    const s = await fetchJSON('/api/status');
    $('status-chips').innerHTML = [
      chip(s.device, s.device === 'cuda' ? 'ok' : ''),
      chip('torch ' + s.torch),
      chip(s.image_size + 'px'),
      chip(s.num_cattle + 'c / ' + s.num_buffalo + 'b'),
      s.model_loaded
        ? chip('model: ' + s.loaded_backbone, 'ok')
        : chip('model: not loaded', 'warn'),
      s.job_running
        ? chip('job: ' + (s.job_kind || 'running'), 'running')
        : '',
    ].filter(Boolean).join('');
    if (s.job_running) {
      startPolling();
    }
    return s;
  } catch (e) {
    $('status-chips').innerHTML = chip('status error: ' + e.message, 'warn');
    return null;
  }
}

/* ---------- predict ---------- */

let currentBytes = null;
let currentType = 'image/jpeg';

const dz = $('dropzone');
dz.addEventListener('click', e => {
  if (e.target.id !== 'browse-btn' && e.target.id !== 'sample-btn') $('image-input').click();
});
$('image-input').addEventListener('change', e => {
  const f = e.target.files[0];
  if (f) handleImage(f);
});
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) handleImage(f);
});

$('clear-btn').addEventListener('click', () => {
  currentBytes = null;
  $('preview-wrap').classList.add('hidden');
  $('result').classList.add('hidden');
  $('predict-error').classList.add('hidden');
  $('image-input').value = '';
});

$('sample-btn').addEventListener('click', async () => {
  try {
    const r = await fetch('/api/sample-image');
    if (!r.ok) throw new Error('no sample image (train first?)');
    const blob = await r.blob();
    currentBytes = await blob.arrayBuffer();
    currentType = blob.type || 'image/jpeg';
    showPreview(URL.createObjectURL(blob));
  } catch (e) { showPredictError(e.message); }
});

async function handleImage(file) {
  currentBytes = await file.arrayBuffer();
  currentType = file.type || 'image/jpeg';
  showPreview(URL.createObjectURL(file));
}

function showPreview(url) {
  $('preview').src = url;
  $('preview-wrap').classList.remove('hidden');
}

function showPredictError(msg) {
  const el = $('predict-error');
  el.textContent = msg;
  el.classList.remove('hidden');
}

$('analyze-btn').addEventListener('click', async () => {
  $('predict-error').classList.add('hidden');
  if (!currentBytes) { showPredictError('Choose an image first.'); return; }
  const btn = $('analyze-btn');
  btn.disabled = true;
  btn.textContent = 'Analyzing...';
  try {
    const fd = new FormData();
    fd.append('file', new Blob([currentBytes], { type: currentType }), 'image.jpg');
    fd.append('backbone', $('predict-backbone').value);
    const r = await fetchJSON('/api/predict', { method: 'POST', body: fd });
    renderResult(r);
  } catch (e) { showPredictError(e.message); }
  finally { btn.disabled = false; btn.textContent = 'Analyze'; }
});

function renderResult(r) {
  const el = $('result');
  const bars = r.top3.map((t, i) => {
    const pct = (t.confidence * 100).toFixed(1);
    return `<div style="font-size:12px;color:var(--muted)">
      <b>${i + 1}.</b> ${esc(t.label)}
      <span style="float:right">${pct}%</span>
      <div class="bar"><div style="width:${Math.max(1, pct)}%"></div></div>
    </div>`;
  }).join('');
  el.innerHTML = `
    <div class="result-species">${esc(r.species)} &middot; ${r.latencyMs} ms</div>
    <div class="result-breed">${esc(r.breed)}</div>
    <div style="color:var(--muted);font-size:12px">
      confidence ${(r.breedConfidence * 100).toFixed(1)}%
      &middot; species confidence ${(r.speciesConfidence * 100).toFixed(1)}%
    </div>
    <h3 style="margin:14px 0 6px;font-size:13px">Top 3</h3>
    ${bars}`;
  el.classList.remove('hidden');
}

/* ---------- train ---------- */

$('train-start').addEventListener('click', async () => {
  try {
    const body = {
      backbone: $('train-backbone').value,
      smoke_test: $('smoke').checked,
      skip_qat: $('skip-qat').checked,
    };
    for (const [k, v] of [['p1', 'phase1_epochs'], ['p2', 'phase2_epochs'], ['p3', 'phase3_epochs']]) {
      const val = parseInt($(k).value, 10);
      if (val > 0) body[v] = val;
    }
    const r = await fetchJSON('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    $('train-log').textContent = '> ' + r.command + '\n';
    $('train-status').textContent = 'started';
    startPolling();
  } catch (e) { $('train-status').textContent = 'error: ' + e.message; }
});

$('train-stop').addEventListener('click', async () => {
  try { await fetchJSON('/api/job/stop', { method: 'POST' }); }
  catch (e) { $('train-status').textContent = 'error: ' + e.message; }
});

/* ---------- evaluate ---------- */

$('eval-run').addEventListener('click', async () => {
  try {
    await fetchJSON('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backbone: $('eval-backbone').value }),
    });
    $('eval-log').textContent = 'evaluation started\n';
    startPolling();
  } catch (e) { $('eval-status').textContent = 'error: ' + e.message; }
});

async function loadMetrics() {
  try {
    const data = await fetchJSON('/api/metrics');
    const entries = Object.entries(data);
    if (!entries.length) {
      $('metrics-panel').textContent = 'no metrics yet (run evaluation)';
      return;
    }
    $('metrics-panel').innerHTML = entries.map(([file, m]) => {
      const rows = [
        ['backbone', m.backbone],
        ['val binary_acc', m.val && m.val.binary_acc],
        ['val cattle_acc', m.val && m.val.cattle_acc],
        ['val buffalo_acc', m.val && m.val.buffalo_acc],
        ['val combined_top1', m.val && m.val.combined_top1],
        ['val combined_top3', m.val && m.val.combined_top3],
        ['test binary_acc', m.test && m.test.binary_acc],
        ['test binary_f1', m.test && m.test.binary_f1],
        ['test cattle_acc', m.test && m.test.cattle_acc],
        ['test buffalo_acc', m.test && m.test.buffalo_acc],
        ['test combined_top1', m.test && m.test.combined_top1],
        ['test combined_top3', m.test && m.test.combined_top3],
        ['test_binary_f1_ok', m.test_binary_f1_ok],
      ].filter(r => r[1] !== undefined && r[1] !== null);
      const html = rows.map(([k, v]) =>
        `<tr><td>${esc(k)}</td><td>${typeof v === 'number' ? (v * 100).toFixed(2) + '%' : esc(v)}</td></tr>`).join('');
      return `<h3>${esc(file)}</h3><table>${html}</table>`;
    }).join('');
  } catch (e) { $('metrics-panel').textContent = 'error: ' + e.message; }
}

/* ---------- debug ---------- */

async function loadDebug() {
  const s = await loadStatus();
  if (s) {
    $('debug-status').innerHTML = `
      <b>device:</b> ${s.device}<br>
      <b>python:</b> ${s.python} &nbsp; <b>torch:</b> ${s.torch} &nbsp; <b>torchvision:</b> ${s.torchvision}<br>
      <b>input:</b> ${s.image_size}x${s.image_size} &nbsp; <b>classes:</b> ${s.num_cattle} cattle + ${s.num_buffalo} buffalo<br>
      <b>model loaded:</b> ${s.model_loaded ? s.loaded_backbone : 'no'}<br>
      <b>checkpoints:</b> ${Object.entries(s.checkpoints || {}).map(([k, v]) => `${k}(${v}MB)`).join(', ') || 'none'}<br>
      <b>exports:</b> ${Object.entries(s.exports || {}).map(([k, v]) => `${k}(${v}MB)`).join(', ') || 'none'}`;
  }
  try {
    const d = await fetchJSON('/api/dataset');
    const breedRows = [
      ...Object.entries(d.cattle).map(([b, n]) => ['cattle', b, n]),
      ...Object.entries(d.buffalo).map(([b, n]) => ['buffalo', b, n]),
    ];
    $('dataset-panel').innerHTML =
      `<b>cattle:</b> ${d.cattle_breeds} breeds, ${d.cattle_total} images (expected ${d.expected_cattle})<br>
       <b>buffalo:</b> ${d.buffalo_breeds} breeds, ${d.buffalo_total} images (expected ${d.expected_buffalo})<br><br>
       <input type="text" id="dataset-search" placeholder="Search breeds..." style="width:100%; margin:10px 0; padding:6px 8px; border:1px solid var(--border); border-radius:6px; background:var(--bg); color:var(--text); font-size:13px">
       <table id="dataset-table"><tr><th>species</th><th>breed</th><th>images</th></tr>
       ${breedRows.map(([sp, b, n]) => `<tr><td>${sp}</td><td>${esc(b)}</td><td>${n}</td></tr>`).join('')}</table>`;

    const sInput = $('dataset-search');
    if (sInput) {
      sInput.addEventListener('input', e => {
        const term = e.target.value.toLowerCase().trim();
        document.querySelectorAll('#dataset-table tr').forEach((row, i) => {
          if (i === 0) return;
          const text = row.textContent.toLowerCase();
          row.style.display = text.includes(term) ? '' : 'none';
        });
      });
    }
  } catch (e) { $('dataset-panel').textContent = 'error: ' + e.message; }
}

$('verify-btn').addEventListener('click', async () => {
  try {
    await fetchJSON('/api/verify', { method: 'POST' });
    $('util-log').textContent = 'verification started\n';
    startPolling();
  } catch (e) { $('util-status').textContent = 'error: ' + e.message; }
});

$('export-btn').addEventListener('click', async () => {
  try {
    await fetchJSON('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        backbone: $('train-backbone').value,
        mode: $('export-mode').value,
      }),
    });
    $('util-log').textContent = 'export started\n';
    startPolling();
  } catch (e) { $('util-status').textContent = 'error: ' + e.message; }
});

/* ---------- memory ---------- */

function scopeFrom(ids) {
  const out = {};
  for (const k of Object.keys(ids)) {
    const v = $(ids[k]).value.trim();
    if (v) out[k] = v;
  }
  return out;
}

async function loadMemoryStatus() {
  try {
    const params = new URLSearchParams(scopeFrom({
      user_id: 'list-user', agent_id: 'list-agent', run_id: 'list-run',
    }));
    const s = await fetchJSON('/api/memory/status?' + params.toString());
    const llm = s.llm_configured
      ? chip(s.llm_provider + ' / ' + s.llm_model, 'ok')
      : chip('no LLM (store/recall still work)', 'warn');
    const mem = s.memories == null
      ? 'n/a (pick a scope)'
      : s.memories + (s.memories_scoped ? ' (scoped)' : '');
    $('memory-status').innerHTML = `
      <b>store:</b> ${esc(s.store_dir)} &nbsp; <b>collection:</b> ${esc(s.collection)}<br>
      <b>embedder:</b> ${esc(s.embedder_model)}<br>
      <b>LLM:</b> ${llm} ${s.llm_base_url ? '&nbsp; base: ' + esc(s.llm_base_url) : ''}<br>
      <b>memories:</b> ${mem}`;
    return s;
  } catch (e) {
    $('memory-status').textContent = 'memory layer error: ' + e.message;
    return null;
  }
}

$('mem-add-btn').addEventListener('click', async () => {
  const text = $('mem-text').value.trim();
  if (!text) { $('mem-add-out').textContent = 'enter some text first'; return; }
  try {
    const r = await fetchJSON('/api/memory/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        ...scopeFrom({ user_id: 'mem-user', agent_id: 'mem-agent', run_id: 'mem-run' }),
        infer: $('mem-infer').checked,
      }),
    });
    $('mem-add-out').textContent = r.results
      .map(x => `${x.event} ${x.memory}`).join('\n') ||
      '(nothing stored)';
    $('mem-text').value = '';
    loadMemoryStatus();
    listMemories();
  } catch (e) { $('mem-add-out').textContent = 'error: ' + e.message; }
});

const SAMPLE_FACTS = [
  "Alice's farm in Kaimoor keeps Sahiwal cows for milk",
  'Sahiwal gives ~2200 kg milk per lactation',
  'Alice also has Murrah buffalo for sale',
  'Alice prefers morning milking sessions',
  "Alice's son is 10 years old and likes cricket",
  'The vet visits Alice farm every month',
  'Bob breeds kosali, a small disease-resistant breed',
];

$('mem-add-demo').addEventListener('click', async () => {
  try {
    for (const text of SAMPLE_FACTS) {
      await fetchJSON('/api/memory/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, user_id: 'alice', run_id: 'session-1', infer: false }),
      });
    }
    $('mem-add-out').textContent = 'stored ' + SAMPLE_FACTS.length +
      ' sample facts for alice/session-1';
    loadMemoryStatus();
    listMemories();
  } catch (e) { $('mem-add-out').textContent = 'error: ' + e.message; }
});

function renderTokenStats(el, s) {
  const total = s.tokens_retrieved || s.tokens_used || 1;
  const used_pct = ((s.tokens_used / total) * 100).toFixed(1);
  const saved_pct = ((s.tokens_saved / total) * 100).toFixed(1);

  el.innerHTML = `
    <div class="row" style="margin-top:0; font-size:12px; justify-content:space-between">
      <span>Retrieved: <b>${s.memories_retrieved}</b> (${total} tokens)</span>
      <span>Injected: <b>${s.memories_in_context}</b> (${s.tokens_used} tokens)</span>
      <span>Saved: <b style="color:var(--accent)">${s.tokens_saved} (${s.saved_pct}%)</b></span>
    </div>
    <div class="token-bar" style="margin-top:6px">
      <div class="used" style="width: ${Math.max(3, parseFloat(used_pct))}%" title="Injected Context: ${s.tokens_used} tokens (${used_pct}%)"></div>
      <div class="saved" style="width: ${Math.max(0, parseFloat(saved_pct))}%" title="Saved/Pruned: ${s.tokens_saved} tokens (${saved_pct}%)"></div>
    </div>
  `;
}

$('recall-btn').addEventListener('click', async () => {
  const query = $('recall-query').value.trim();
  if (!query) { $('recall-stats').textContent = 'enter a query'; return; }
  try {
    const s = await fetchJSON('/api/memory/recall', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        ...scopeFrom({ user_id: 'recall-user', agent_id: 'recall-agent', run_id: 'recall-run' }),
        top_k: parseInt($('recall-topk').value, 10) || 5,
      }),
    });
    renderTokenStats($('recall-stats'), s);
    $('recall-context').textContent = s.context || '(no relevant memories)';
  } catch (e) { $('recall-stats').textContent = 'error: ' + e.message; }
});

$('chat-btn').addEventListener('click', async () => {
  const message = $('chat-message').value.trim();
  if (!message) { $('chat-reply').textContent = 'enter a message'; return; }
  const btn = $('chat-btn');
  btn.disabled = true;
  btn.textContent = 'Asking...';
  $('chat-error').classList.add('hidden');
  try {
    const r = await fetchJSON('/api/memory/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        ...scopeFrom({ user_id: 'chat-user', agent_id: 'chat-agent', run_id: 'chat-run' }),
        system_hint: $('chat-hint').value || null,
      }),
    });
    $('chat-reply').textContent = r.reply;
    renderTokenStats($('chat-stats'), r.context);
    $('chat-context').textContent = r.context.context || '(no context)';
    loadMemoryStatus();
    listMemories();
  } catch (e) {
    const el = $('chat-error');
    el.textContent = e.message;
    el.classList.remove('hidden');
  } finally { btn.disabled = false; btn.textContent = 'Ask'; }
});

async function listMemories() {
  const params = new URLSearchParams(scopeFrom({
    user_id: 'list-user', agent_id: 'list-agent', run_id: 'list-run',
  }));
  try {
    const r = await fetchJSON('/api/memory/all?' + params.toString());
    const ms = r.results || [];
    const tableHTML = ms.length
      ? `<table style="margin-top:0">
          <thead><tr><th>Scope &amp; Memory Content</th><th style="text-align:right">Action</th></tr></thead>
          <tbody>
          ${ms.map(m => {
            const audit = {
              id: m.id,
              created_at: m.created_at,
              updated_at: m.updated_at,
              hash: m.hash,
              ...(m.metadata || {})
            };
            for (const k of Object.keys(audit)) {
              if (audit[k] === null || audit[k] === undefined || audit[k] === '') delete audit[k];
            }
            return `<tr style="border-bottom:1px solid var(--border)">
              <td style="padding:8px 0">
                <div style="font-size:11px; color:var(--muted); margin-bottom:4px">
                  id: <b style="font-family:var(--mono)">${esc(m.id.slice(0, 8))}</b> &middot;
                  user: <b>${esc(m.user_id || '-')}</b> &middot;
                  agent: <b>${esc(m.agent_id || '-')}</b> &middot;
                  run: <b>${esc(m.run_id || '-')}</b>
                </div>
                <div style="font-size:13px; font-weight:500; color:var(--text)">${esc(m.memory)}</div>
                <details class="mem-details">
                  <summary>Audit &amp; Metadata</summary>
                  <pre>${esc(JSON.stringify(audit, null, 2))}</pre>
                </details>
              </td>
              <td style="text-align:right; vertical-align:top; padding-top:8px">
                <button class="ghost" data-del="${esc(m.id)}" style="padding:3px 8px; font-size:11px">del</button>
              </td>
            </tr>`;
          }).join('')}
          </tbody>
         </table>`
      : '(no memories in scope)';

    $('mem-list').innerHTML = tableHTML;

    document.querySelectorAll('#mem-list [data-del]').forEach(b =>
      b.addEventListener('click', async () => {
        try {
          await fetchJSON('/api/memory/' + encodeURIComponent(b.dataset.del),
            { method: 'DELETE' });
          listMemories(); loadMemoryStatus();
        } catch (e) { $('mem-list').textContent = 'error: ' + e.message; }
      }));
  } catch (e) { $('mem-list').textContent = 'error: ' + e.message; }
}

$('mem-list-btn').addEventListener('click', listMemories);
$('mem-clear-btn').addEventListener('click', async () => {
  try {
    await fetchJSON('/api/memory/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scopeFrom({ user_id: 'list-user', agent_id: 'list-agent', run_id: 'list-run' })),
    });
    listMemories(); loadMemoryStatus();
  } catch (e) { $('mem-list').textContent = 'error: ' + e.message; }
});
$('mem-reset-btn').addEventListener('click', async () => {
  try {
    await fetchJSON('/api/memory/reset', { method: 'POST' });
    listMemories(); loadMemoryStatus();
  } catch (e) { $('mem-list').textContent = 'error: ' + e.message; }
});

/* ---------- job poller ---------- */

let pollInterval = null;
let prevRunning = false;
let prevKind = null;

function fmtEta(s) {
  if (s == null) return '';
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return m > 0 ? `${m}m ${r}s` : `${r}s`;
}

function renderPBar(el, st) {
  if (!el) return;
  if (!st.running && st.exit !== null) {
    // Show completion state
    if (st.exit === 0) {
      el.innerHTML = `
        <div class="pbar"><div style="width:100%" class="pbar-done"></div></div>
        <div class="pbar-text">Complete ✓</div>`;
      el.classList.remove('hidden');
    } else if (st.exit != null) {
      el.innerHTML = `
        <div class="pbar"><div style="width:100%" class="pbar-error"></div></div>
        <div class="pbar-text">Failed (exit ${st.exit})</div>`;
      el.classList.remove('hidden');
    } else {
      el.classList.add('hidden');
      el.innerHTML = '';
    }
    return;
  }
  if (!st.running) {
    el.classList.add('hidden');
    el.innerHTML = '';
    return;
  }
  const parts = [];
  let pct = st.progress || 0;
  let label = '';
  if (st.kind === 'train') {
    if (st.phase != null) parts.push(`Phase ${st.phase}/${st.phases || 3}`);
    if (st.epoch != null && st.epochs != null) parts.push(`Epoch ${st.epoch}/${st.epochs}`);
    if (st.batch) {
      label = `Batch ${st.batch.done}/${st.batch.total}`;
    }
    if (st.progress != null) {
      pct = st.progress;
      parts.push(`${st.progress.toFixed(1)}% total`);
    }
  } else if (st.bar) {
    pct = st.bar.pct;
    label = `${st.bar.label} ${st.bar.done}/${st.bar.total}`;
    parts.push(`${pct.toFixed(1)}%`);
  } else if (st.progress != null && st.progress > 0) {
    pct = st.progress;
    parts.push(`${pct.toFixed(1)}%`);
  }
  if (st.eta_s != null) parts.push(`ETA ${fmtEta(st.eta_s)}`);
  el.innerHTML = `
    <div class="pbar"><div style="width:${Math.max(1, pct)}%"></div></div>
    <div class="pbar-text">${parts.concat(label ? [label] : []).join(' &middot; ')}</div>`;
  el.classList.remove('hidden');
}

async function pollOnce() {
  let job;
  try { job = await fetchJSON('/api/job'); }
  catch (_) { return; }

  const st = job.status;
  const statusText = jobStatusHTML(st);

  // Update the correct panel based on job kind
  if (st.kind === 'train') {
    $('train-status').textContent = statusText;
    renderPBar($('train-pbar'), st);
    if (job.log.length) setConsole($('train-log'), job.log);
  } else if (st.kind === 'evaluate') {
    $('eval-status').textContent = statusText;
    renderPBar($('eval-pbar'), st);
    if (job.log.length) setConsole($('eval-log'), job.log);
  } else if (st.kind === 'verify' || st.kind === 'export') {
    $('util-status').textContent = statusText;
    renderPBar($('util-pbar'), st);
    if (job.log.length) setConsole($('util-log'), job.log);
  }

  // On job completion, refresh relevant data
  if (prevRunning && !st.running && st.exit !== null) {
    loadStatus();
    if (st.kind === 'evaluate') loadMetrics();
    if (st.kind === 'export') loadStatus();
    if (st.kind === 'train') {
      // Refresh everything after training: model may have changed
      loadStatus();
      loadMetrics();
    }
  }
  prevRunning = st.running;
  prevKind = st.kind;

  if (!st.running) {
    stopPolling();
  }
}

function startPolling() {
  if (pollInterval) return;
  prevRunning = true;
  pollOnce();
  pollInterval = setInterval(pollOnce, 1200);
}

function stopPolling() {
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}

/* init */
loadStatus();
