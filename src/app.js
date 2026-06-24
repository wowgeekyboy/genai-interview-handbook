/* =========================================================================
   THE GENAI HANDBOOK — app.js
   SPA navigation, command-palette search, theme, code tooling, animations.
   ========================================================================= */
'use strict';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const state = {
  manifest: null,
  search: [],
  fragmentCache: new Map(),
  current: null,
};

const FEATURES = [
  { i: '🧠', h: 'Concept → Visual → Answer', p: 'Each topic explained simply, drawn as a diagram, then phrased the way a senior engineer says it out loud.' },
  { i: '🔁', h: 'RAG, 7 ways', p: 'Vanilla plus Self-RAG, Corrective, Adaptive, Agentic, HippoRAG & GraphRAG — with when-to-use trade-offs.' },
  { i: '🕸️', h: 'Agents & LangGraph', p: 'State, nodes, edges, cycles, persistence, multi-agent and human-in-the-loop, with runnable code.' },
  { i: '🚀', h: 'Production-grade', p: 'Cost, latency, observability, evals, guardrails, OWASP LLM Top 10 — the stuff interviews really probe.' },
  { i: '🔐', h: 'Security & PII', p: 'A full 5-layer masking/unmasking defense, prompt-injection handling, and compliance you can speak to.' },
  { i: '⚡', h: 'Cutting-edge 2026', p: 'Reasoning models, MoE, Flash/Paged attention, quantization, distributed training, MCP at scale.' },
];

/* ---------- boot ---------- */
init();

async function init() {
  initTheme();
  initParticles();
  initProgress();
  bindGlobalUI();

  try {
    const [manifest, search] = await Promise.all([
      fetch('data/manifest.json').then(r => r.json()),
      fetch('data/search-index.json').then(r => r.json()),
    ]);
    state.manifest = manifest;
    state.search = search;
    renderHome();
    renderNav();
    initSearch();
    window.addEventListener('hashchange', route);
    route();
  } catch (e) {
    console.error('Failed to load handbook data', e);
    $('#proseContent').innerHTML = '<p>Could not load handbook data. If you opened this file directly, run it through a local server (npm run dev).</p>';
  }
}

/* ---------- theme ---------- */
function initTheme() {
  const saved = localStorage.getItem('gh-theme') || 'dark';
  setTheme(saved);
  $('#themeBtn').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
  });
}
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  document.body.dataset.theme = t;
  localStorage.setItem('gh-theme', t);
  $('#themeBtn').textContent = t === 'dark' ? '◐' : '◑';
}

/* ---------- particle network ---------- */
function initParticles() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const canvas = $('#particles');
  const ctx = canvas.getContext('2d');
  let w, h, pts, raf;
  const COUNT = Math.min(64, Math.floor(window.innerWidth / 24));

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = Math.min(window.innerHeight * 1.1, 900);
    pts = Array.from({ length: COUNT }, () => ({
      x: Math.random() * w, y: Math.random() * h,
      vx: (Math.random() - .5) * .35, vy: (Math.random() - .5) * .35,
    }));
  }
  function draw() {
    ctx.clearRect(0, 0, w, h);
    for (const p of pts) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
    }
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
        const d = Math.hypot(dx, dy);
        if (d < 130) {
          ctx.strokeStyle = `rgba(129,140,248,${(1 - d / 130) * 0.18})`;
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y); ctx.stroke();
        }
      }
    }
    for (const p of pts) {
      ctx.fillStyle = 'rgba(34,211,238,.5)';
      ctx.beginPath(); ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2); ctx.fill();
    }
    raf = requestAnimationFrame(draw);
  }
  resize(); draw();
  let to;
  window.addEventListener('resize', () => { clearTimeout(to); to = setTimeout(() => { cancelAnimationFrame(raf); resize(); draw(); }, 200); });
}

/* ---------- progress + back to top ---------- */
function initProgress() {
  const bar = $('#progress'), top = $('#toTop');
  const onScroll = () => {
    const st = document.documentElement.scrollTop || document.body.scrollTop;
    const sh = (document.documentElement.scrollHeight - document.documentElement.clientHeight) || 1;
    bar.style.width = (st / sh * 100) + '%';
    top.classList.toggle('show', st > 600);
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  top.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
}

/* ---------- global UI ---------- */
function bindGlobalUI() {
  $('#startBtn').addEventListener('click', () => { location.hash = '#/' + (state.manifest?.sections[1]?.id || 'intro'); });
  $('#brandHome').addEventListener('click', () => { location.hash = ''; goHome(); });
  $('#searchTrigger').addEventListener('click', openCmdk);
  $('#menuBtn').addEventListener('click', () => document.body.classList.toggle('nav-open'));
  $('#scrim').addEventListener('click', () => document.body.classList.remove('nav-open'));

  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openCmdk(); }
    if (e.key === '/' && !/input|textarea/i.test(document.activeElement.tagName) && !cmdkOpen()) { e.preventDefault(); openCmdk(); }
    if (e.key === 'Escape') closeCmdk();
  });
}

/* ---------- home ---------- */
function renderHome() {
  const s = state.manifest.stats;
  const statDefs = [
    { n: s.lines, l: 'Lines of depth', fmt: shortNum },
    { n: s.questions, l: 'Interview Q&A', suffix: '+' },
    { n: s.codeBlocks, l: 'Code snippets', suffix: '+' },
    { n: s.parts, l: 'Full parts' },
    { n: 12, l: 'Topic domains' },
  ];
  $('#heroStats').innerHTML = statDefs.map(d =>
    `<div class="stat"><div class="num" data-target="${d.n}" data-suffix="${d.suffix || ''}" data-fmt="${d.fmt ? 1 : 0}">0</div><div class="label">${d.l}</div></div>`
  ).join('');

  $('#featureGrid').innerHTML = FEATURES.map(f =>
    `<div class="feature reveal"><div class="fi">${f.i}</div><h3>${f.h}</h3><p>${f.p}</p></div>`
  ).join('');

  $('#currGrid').innerHTML = state.manifest.sections.map(sec => {
    const label = sec.num ? sec.num : '✦';
    const count = sec.subs.length ? `${sec.subs.length} topics` : 'Overview';
    return `<a class="curr-card reveal" data-go="${sec.id}"><div class="pnum">${label}</div><div><h4>${esc(sec.title)}</h4><p>${count}</p></div></a>`;
  }).join('');
  $$('#currGrid [data-go]').forEach(a => a.addEventListener('click', () => { location.hash = '#/' + a.dataset.go; }));

  initReveal();
  animateStats();
}

function animateStats() {
  $$('#heroStats .num').forEach(el => {
    const target = +el.dataset.target, suffix = el.dataset.suffix || '', fmt = el.dataset.fmt === '1';
    const dur = 1400, t0 = performance.now();
    const tick = (t) => {
      const p = Math.min((t - t0) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = Math.round(target * eased);
      el.textContent = (fmt ? shortNum(val) : val.toLocaleString()) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

/* ---------- nav tree ---------- */
function renderNav() {
  const nav = $('#navTree');
  nav.innerHTML = state.manifest.sections.map(sec => {
    const label = sec.num ? sec.num : '✦';
    const subs = sec.subs.map(sub =>
      `<a href="#/${sec.id}/${sub.slug}" data-part="${sec.id}" data-slug="${sub.slug}">${esc(stripNum(sub.title))}</a>`
    ).join('');
    return `<div class="nav-part" data-part="${sec.id}">
      <button data-part="${sec.id}"><span class="pn">${label}</span><span class="txt">${esc(sec.title)}</span>${sec.subs.length ? '<span class="caret">▶</span>' : ''}</button>
      <div class="nav-subs">${subs}</div>
    </div>`;
  }).join('');

  $$('#navTree .nav-part > button').forEach(btn => {
    btn.addEventListener('click', () => {
      const part = btn.dataset.part;
      const wrap = btn.closest('.nav-part');
      if (state.current === part) wrap.classList.toggle('open');
      else location.hash = '#/' + part;
    });
  });
  $$('#navTree .nav-subs a').forEach(a => {
    a.addEventListener('click', () => document.body.classList.remove('nav-open'));
  });
}

/* ---------- routing ---------- */
function route() {
  const hash = location.hash.replace(/^#\/?/, '');
  if (!hash) { goHome(); return; }
  const [partId, slug] = hash.split('/');
  const sec = state.manifest.sections.find(s => s.id === partId);
  if (!sec) { goHome(); return; }
  loadPart(sec, slug);
}

function goHome() {
  document.body.classList.remove('reading', 'nav-open');
  window.scrollTo(0, 0);
  document.title = 'The Complete GenAI & AI Engineering Handbook';
}

async function loadPart(sec, slug) {
  document.body.classList.add('reading');
  document.body.classList.remove('nav-open');
  state.current = sec.id;

  let html = state.fragmentCache.get(sec.file);
  if (!html) {
    html = await fetch('data/parts/' + sec.file).then(r => r.text());
    state.fragmentCache.set(sec.file, html);
  }
  const prose = $('#proseContent');
  prose.innerHTML = html;
  enhanceContent(prose);
  buildToc(sec);
  buildBreadcrumb(sec);
  buildPager(sec);
  highlightNav(sec.id, slug);
  document.title = `${sec.title} · GenAI Handbook`;

  // scroll
  if (slug) {
    requestAnimationFrame(() => {
      const target = document.getElementById(slug);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      else window.scrollTo(0, 0);
    });
  } else {
    window.scrollTo(0, 0);
  }
  observeHeadings();
}

function highlightNav(partId, slug) {
  $$('#navTree .nav-part').forEach(p => {
    const on = p.dataset.part === partId;
    p.classList.toggle('active', on);
    p.classList.toggle('open', on);
  });
  $$('#navTree .nav-subs a').forEach(a => a.classList.toggle('active', a.dataset.part === partId && a.dataset.slug === slug));
  const active = $(`#navTree .nav-part.active`);
  if (active) active.scrollIntoView({ block: 'nearest' });
}

/* ---------- content enhancement ---------- */
function enhanceContent(root) {
  // Code blocks: wrap, add toolbar, copy, detect diagrams, highlight
  $$('pre', root).forEach(pre => {
    const code = pre.querySelector('code');
    if (!code) return;
    const langClass = [...code.classList].find(c => c.startsWith('language-'));
    let lang = langClass ? langClass.replace('language-', '') : '';
    const text = code.textContent;
    const isDiagram = !lang && /[┌└┐┘├┤┬┴┼│─►▶▼◄→↓↑]/.test(text);

    if (isDiagram) {
      const wrap = document.createElement('div');
      wrap.className = 'diagram-wrap';
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);
      return;
    }

    const wrap = document.createElement('div');
    wrap.className = 'code-wrap';
    pre.parentNode.insertBefore(wrap, pre);

    const head = document.createElement('div');
    head.className = 'code-head';
    head.innerHTML = `<span class="dots"><i></i><i></i><i></i></span><span class="lang">${esc(lang || 'text')}</span>
      <button class="copy">⧉ Copy</button>`;
    wrap.appendChild(head);
    wrap.appendChild(pre);

    if (window.hljs) {
      try { lang && hljs.getLanguage(lang) ? hljs.highlightElement(code) : hljs.highlightElement(code); } catch (e) {}
    }

    head.querySelector('.copy').addEventListener('click', (ev) => {
      navigator.clipboard.writeText(text).then(() => {
        const b = ev.target; b.textContent = '✓ Copied'; b.classList.add('done');
        setTimeout(() => { b.textContent = '⧉ Copy'; b.classList.remove('done'); }, 1600);
      });
    });
  });

  // Wrap tables for horizontal scroll
  $$('table', root).forEach(t => {
    if (t.parentElement.classList.contains('table-wrap')) return;
    const wrap = document.createElement('div'); wrap.className = 'table-wrap';
    t.parentNode.insertBefore(wrap, t); wrap.appendChild(t);
  });

  // External links open in new tab
  $$('a[href^="http"]', root).forEach(a => { a.target = '_blank'; a.rel = 'noopener'; });
}

/* ---------- TOC ---------- */
function buildToc(sec) {
  const toc = $('#tocNav');
  const heads = $$('#proseContent h2, #proseContent h3').filter(h => h.id);
  toc.innerHTML = heads.map(h =>
    `<a href="#/${sec.id}/${h.id}" data-id="${h.id}" style="padding-left:${h.tagName === 'H3' ? 24 : 12}px">${esc(stripNum(h.textContent.replace(/^#/, '')))}</a>`
  ).join('');
  $$('#tocNav a').forEach(a => a.addEventListener('click', (e) => {
    e.preventDefault();
    const t = document.getElementById(a.dataset.id);
    if (t) { t.scrollIntoView({ behavior: 'smooth', block: 'start' }); history.replaceState(null, '', `#/${sec.id}/${a.dataset.id}`); }
  }));
}

let headingObserver;
function observeHeadings() {
  if (headingObserver) headingObserver.disconnect();
  const heads = $$('#proseContent h2, #proseContent h3').filter(h => h.id);
  headingObserver = new IntersectionObserver((entries) => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        $$('#tocNav a').forEach(a => a.classList.toggle('active', a.dataset.id === en.target.id));
      }
    });
  }, { rootMargin: '-80px 0px -70% 0px' });
  heads.forEach(h => headingObserver.observe(h));
}

/* ---------- breadcrumb + pager ---------- */
function buildBreadcrumb(sec) {
  $('#breadcrumb').innerHTML =
    `<a href="#" onclick="location.hash='';return false;">Home</a> <span>/</span> <span>${sec.num ? 'Part ' + sec.num : 'Start'}</span> <span>/</span> <span style="color:var(--text)">${esc(sec.title)}</span>`;
}

function buildPager(sec) {
  const list = state.manifest.sections;
  const idx = list.findIndex(s => s.id === sec.id);
  const prev = list[idx - 1], next = list[idx + 1];
  let html = '';
  if (prev) html += `<a href="#/${prev.id}"><div class="dir">← Previous</div><span class="ttl">${esc(prev.title)}</span></a>`;
  else html += `<a href="#" onclick="location.hash='';return false;"><div class="dir">← Home</div><span class="ttl">Overview</span></a>`;
  if (next) html += `<a class="next" href="#/${next.id}"><div class="dir">Next →</div><span class="ttl">${esc(next.title)}</span></a>`;
  $('#pager').innerHTML = html;
}

/* ---------- search / command palette ---------- */
let cmdkSel = 0, cmdkMatches = [];
function initSearch() {
  const input = $('#cmdkInput');
  input.addEventListener('input', () => runSearch(input.value));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); cmdkSel = Math.min(cmdkSel + 1, cmdkMatches.length - 1); paintSel(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); cmdkSel = Math.max(cmdkSel - 1, 0); paintSel(); }
    if (e.key === 'Enter') { e.preventDefault(); const m = cmdkMatches[cmdkSel]; if (m) gotoResult(m); }
  });
  $('#cmdk').addEventListener('click', (e) => { if (e.target.id === 'cmdk') closeCmdk(); });
}
function cmdkOpen() { return $('#cmdk').classList.contains('open'); }
function openCmdk() {
  $('#cmdk').classList.add('open');
  const input = $('#cmdkInput'); input.value = ''; input.focus();
  runSearch('');
}
function closeCmdk() { $('#cmdk').classList.remove('open'); }

function runSearch(q) {
  q = q.trim().toLowerCase();
  const results = $('#cmdkResults');
  if (!q) {
    cmdkMatches = state.manifest.sections.filter(s => s.num).slice(0, 8).map(s => ({
      part: s.id, partTitle: s.title, title: s.title, slug: '', text: `${s.subs.length} topics`, type: 'part', _score: 0,
    }));
  } else {
    const terms = q.split(/\s+/);
    cmdkMatches = state.search.map(item => {
      const hay = (item.title + ' ' + item.text + ' ' + item.partTitle).toLowerCase();
      let score = 0;
      for (const t of terms) {
        if (!hay.includes(t)) { score = -1; break; }
        if (item.title.toLowerCase().includes(t)) score += 10;
        if (item.title.toLowerCase().startsWith(t)) score += 6;
        if (item.text.toLowerCase().includes(t)) score += 2;
      }
      return { ...item, _score: score };
    }).filter(x => x._score > 0)
      .sort((a, b) => b._score - a._score)
      .slice(0, 24);
  }
  cmdkSel = 0;
  if (!cmdkMatches.length) { results.innerHTML = `<div class="cmdk-empty">No matches for “${esc(q)}”. Try “RAG”, “attention”, “LoRA”, “latency”…</div>`; return; }
  results.innerHTML = cmdkMatches.map((m, i) =>
    `<a class="cmdk-item${i === 0 ? ' sel' : ''}" data-i="${i}">
      <div class="ci-top"><span class="ci-part">${m.type === 'part' ? 'Part' : esc(shortPart(m.partTitle))}</span><span class="ci-title">${hl(stripNum(m.title), q)}</span></div>
      ${m.text ? `<div class="ci-text">${hl(m.text, q)}</div>` : ''}
    </a>`
  ).join('');
  $$('#cmdkResults .cmdk-item').forEach(el => {
    el.addEventListener('click', () => gotoResult(cmdkMatches[+el.dataset.i]));
    el.addEventListener('mousemove', () => { cmdkSel = +el.dataset.i; paintSel(); });
  });
}
function paintSel() {
  $$('#cmdkResults .cmdk-item').forEach((el, i) => el.classList.toggle('sel', i === cmdkSel));
  const sel = $('#cmdkResults .cmdk-item.sel');
  if (sel) sel.scrollIntoView({ block: 'nearest' });
}
function gotoResult(m) {
  closeCmdk();
  location.hash = '#/' + m.part + (m.slug ? '/' + m.slug : '');
}

/* ---------- reveal ---------- */
function initReveal() {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); obs.unobserve(e.target); } });
  }, { threshold: 0.12 });
  $$('.reveal').forEach(el => obs.observe(el));
}

/* ---------- utils ---------- */
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function hl(text, q) {
  text = esc(text);
  if (!q) return text;
  for (const t of q.split(/\s+/).filter(Boolean)) {
    try { text = text.replace(new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig'), '<mark>$1</mark>'); } catch (e) {}
  }
  return text;
}
function stripNum(t) { return String(t).replace(/^\s*(§?\d+(\.\d+)*)\s*[:.]?\s*/, '').trim() || t; }
function shortNum(n) { return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1).replace(/\.0$/, '') + 'k' : '' + n; }
function shortPart(t) { return t.length > 16 ? t.slice(0, 15) + '…' : t; }
