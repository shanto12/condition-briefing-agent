"use strict";
const $ = (id) => document.getElementById(id);
const API_BASE = ((window.APP_CONFIG && window.APP_CONFIG.API_BASE) || "").replace(/\/+$/, "");
const KEY_STORE = "demoKey";
let threadId = null;
let lastStatus = null;
let busyNow = false;

const PLACEHOLDERS = {
  awaiting_answers: "Answer the questions, e.g. Alzheimer's, service-line, none",
  awaiting_approval: "Approve or reject the draft above",
  done: "Ask a follow-up question about this briefing",
  rejected: "Ask a follow-up question about this briefing",
};

/* ---------- storage helpers ---------- */
function getKey() { try { return sessionStorage.getItem(KEY_STORE) || ""; } catch (e) { return ""; } }
function setKey(v) { try { if (v) sessionStorage.setItem(KEY_STORE, v); else sessionStorage.removeItem(KEY_STORE); } catch (e) { /* ignore */ } }

/* ---------- DOM helpers ---------- */
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
const SVGNS = "http://www.w3.org/2000/svg";
function icon(name) {
  const s = document.createElementNS(SVGNS, "svg");
  s.setAttribute("class", "ic");
  s.setAttribute("aria-hidden", "true");
  const u = document.createElementNS(SVGNS, "use");
  u.setAttribute("href", `#i-${name}`);
  s.appendChild(u);
  return s;
}
function toast(msg, isErr) {
  const t = el("div", "toast" + (isErr ? " err" : ""), msg);
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), 4500);
}

/* ---------- theme ---------- */
function currentTheme() {
  const t = document.documentElement.getAttribute("data-theme");
  if (t) return t;
  return window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function syncThemeIcon() {
  const dark = currentTheme() === "dark";
  $("theme-ic").firstElementChild.setAttribute("href", dark ? "#i-sun" : "#i-moon");
  $("theme").setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
}
$("theme").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
  syncThemeIcon();
});
syncThemeIcon();

/* ---------- markdown (escape first; only https links become anchors) ---------- */
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function chipKind(t) {
  if (/^PMID:/i.test(t)) return "pmid";
  if (/^NCT\d/i.test(t)) return "nct";
  if (/^FDA:/i.test(t)) return "fda";
  if (/^DOC:/i.test(t)) return "doc";
  return null;
}
const CITE_RE = /^(PMID:\S+|NCT\d+|FDA:\S+|DOC:\S+)$/i;

function inline(s) {
  return escapeHtml(s)
    .replace(/\[([^\]]+)\]\((https:\/\/[^\s)]+)\)/g, (_, t, u) => {
      const k = CITE_RE.test(t) ? chipKind(t) : null;
      return `<a class="${k ? "chip chip-" + k : "ext"}" href="${u}" target="_blank" rel="noopener noreferrer">${t}</a>`;
    })
    .replace(/`([^`]+)`/g, (_, c) => { const k = CITE_RE.test(c) ? chipKind(c) : null; return k ? `<span class="chip chip-${k}">${c}</span>` : `<code>${c}</code>`; })
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/~~([^~]+)~~/g, "<del>$1</del>")
    .replace(/(^|\s)_([^_]+)_(?=\s|$|[.,;:])/g, "$1<em>$2</em>");
}

function renderMarkdown(md) {
  const lines = String(md).split("\n");
  let html = "", list = null, table = [];
  const flushList = () => { if (list) { html += `</${list}>`; list = null; } };
  const flushTable = () => {
    if (!table.length) return;
    const rows = table.filter((r) => !/^\|\s*:?-/.test(r)).map((r) => r.replace(/^\||\|$/g, "").split("|"));
    html += '<div class="twrap"><table><thead><tr>' + rows[0].map((c) => `<th>${inline(c.trim())}</th>`).join("") + "</tr></thead><tbody>" +
      rows.slice(1).map((cells) => "<tr>" + cells.map((c) => `<td>${inline(c.trim())}</td>`).join("") + "</tr>").join("") + "</tbody></table></div>";
    table = [];
  };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith("|")) { flushList(); table.push(line); continue; }
    flushTable();
    let m;
    if ((m = line.match(/^(#{1,3}) (.*)/))) { flushList(); html += `<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`; }
    else if ((m = line.match(/^- (.*)/))) { if (list !== "ul") { flushList(); html += "<ul>"; list = "ul"; } html += `<li>${inline(m[1])}</li>`; }
    else if ((m = line.match(/^\d+\. (.*)/))) { if (list !== "ol") { flushList(); html += "<ol>"; list = "ol"; } html += `<li>${inline(m[1])}</li>`; }
    else if ((m = line.match(/^> (.*)/))) { flushList(); html += `<blockquote>${inline(m[1])}</blockquote>`; }
    else if ((m = line.match(/^_(.+)_$/))) { flushList(); html += `<p><em>${inline(m[1])}</em></p>`; }
    else if (line.trim() === "") { flushList(); }
    else { flushList(); html += `<p>${inline(line)}</p>`; }
  }
  flushList(); flushTable();
  return html;
}
function prose(md) { const d = el("div", "prose"); d.innerHTML = renderMarkdown(md); return d; }

/* ---------- briefing report (structured cards from the markdown the API returns) ---------- */
const SECTION_ICONS = [
  [/executive/i, "spark"], [/standard of care/i, "pill"], [/emerging|pipeline/i, "flask"],
  [/compan|institution/i, "building"], [/patient panel/i, "users"], [/patient context/i, "shield"],
  [/gap|caveat/i, "alert"], [/removed/i, "x"], [/run log|guardrail/i, "shield"], [/source/i, "book"],
];
function sectionIcon(title) { for (const [re, n] of SECTION_ICONS) if (re.test(title)) return n; return "file"; }

function isBriefing(text) { return /^# .+/.test(text) && /\n## (Executive summary|Current standard of care)/.test(text); }

function splitSections(md) {
  const lines = md.split("\n"); const head = []; const secs = []; let cur = null;
  for (const l of lines) {
    const m = l.match(/^## (.*)/);
    if (m) { cur = { title: m[1].trim(), body: [] }; secs.push(cur); } else if (cur) cur.body.push(l); else head.push(l);
  }
  return { head, secs: secs.map((s) => ({ title: s.title, body: s.body.join("\n").trim() })) };
}

function statusClass(s) {
  const k = String(s || "").toLowerCase();
  if (/final|approved|done/.test(k)) return "badge-ok";
  if (/reject|fail/.test(k)) return "badge-bad";
  if (/draft|await|pending/.test(k)) return "badge-warn";
  return "";
}

function metaTile(label, value, extra) {
  const m = el("div", "meta" + (extra ? " " + extra : ""));
  m.append(el("small", null, label), el("b", null, value));
  return m;
}

function buildHead(headLines) {
  const wrap = el("header", "report-head");
  let title = "Briefing", sub = "";
  const metas = el("div", "metas");
  for (const l of headLines) {
    let m;
    if ((m = l.match(/^# (.*)/))) { title = m[1]; continue; }
    for (const part of l.split(" | ")) {
      if ((m = part.trim().match(/^\*\*(.+?):\*\*\s*(.*)$/))) {
        const label = m[1], val = m[2].trim();
        if (label === "Retrieval") { metas.append(metaTile("Retrieval", val.replace(/^Grounded in /, ""))); continue; }
        if (label === "Status") {
          const st = val.split(" ")[0].toLowerCase();
          metas.append(metaTile("Status", val.replace(/\(awaiting analyst approval\)/, "").trim(), "status-" + st));
          if (/awaiting/.test(val)) sub = "Draft awaiting your approval";
          continue;
        }
        if (label === "Generated" && !isNaN(Date.parse(val))) { metas.append(metaTile(label, new Date(val).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }))); continue; }
        metas.append(metaTile(label, val));
      }
    }
  }
  wrap.append(el("div", "eyebrow", "Condition briefing" + (sub ? " | " + sub : "")), el("h2", null, title.replace(/ briefing/i, "")), metas);
  return wrap;
}

function mkCard(title, opts = {}) {
  const collapsible = opts.collapsible;
  const card = el(collapsible ? "details" : "section", "card" + (opts.cls ? " " + opts.cls : ""));
  if (collapsible && opts.open) card.open = true;
  const h = el(collapsible ? "summary" : "div", collapsible ? null : "card-h");
  const inner = collapsible ? h : h;
  const ic = el("span", "card-ic"); ic.setAttribute("aria-hidden", "true"); ic.appendChild(icon(sectionIcon(title)));
  const hd = el("h3", null, title);
  const row = collapsible ? el("div", "card-h") : null;
  if (collapsible) {
    const wrapper = el("span", "card-h"); wrapper.append(ic, hd);
    inner.append(wrapper);
    if (opts.count) inner.append(el("span", "count", opts.count));
    const chev = icon("chev"); chev.classList.add("chev"); inner.append(chev);
  } else {
    inner.append(ic, hd);
    if (opts.count) inner.append(el("span", "count", opts.count));
  }
  void row;
  card.appendChild(h);
  return card;
}

function parseNum(v) { const n = parseInt(String(v).replace(/[^\d]/g, ""), 10); return /^\s*\d+\s*$/.test(String(v)) && !isNaN(n) ? n : null; }

function buildPanel(body) {
  const frag = el("div", "card-b");
  const lines = body.split("\n");
  const noteLine = lines.find((l) => /^_.*_$/.test(l.trim()));
  if (noteLine) frag.append(el("p", "panel-note", noteLine.trim().replace(/^_|_$/g, "")));
  const tiles = el("div", "tiles");
  for (const l of lines) {
    let m;
    if ((m = l.match(/^- Matched patients: \*\*(.+?)\*\* of (.+)$/))) {
      const t = el("div", "tile big"); const n = el("div", "num", m[1]);
      t.append(n, el("div", "lbl", "Matched patients in our panel"), el("div", "of", "of " + m[2])); tiles.append(t);
    } else if ((m = l.match(/^- (By [^:]+): (.*)$/))) {
      const t = el("div", "tile"); t.append(el("div", "lbl", m[1]));
      const items = m[2].split(/,\s*/).map((p) => { const mm = p.match(/^(.*)\s(\S+)$/); return mm ? [mm[1], mm[2]] : [p, ""]; });
      const max = Math.max(1, ...items.map((i) => parseNum(i[1]) || 0));
      const bars = el("div", "bars");
      for (const [k, v] of items) {
        const row = el("div", "bar-row"); const bl = el("div", "bl"); bl.append(el("span", null, k), el("b", null, v));
        const track = el("div", "track"); const n = parseNum(v);
        const fill = el("div", "fill " + (n == null ? "hidden-cell" : "w" + (Math.max(1, Math.round((n / max) * 20)) * 5)));
        track.append(fill); row.append(bl, track); bars.append(row);
      }
      t.append(bars); tiles.append(t);
    }
  }
  frag.append(tiles);
  return frag;
}

function buildSources(body) {
  const b = el("div", "card-b");
  let ul = null;
  for (const raw of body.split("\n")) {
    const l = raw.trim(); if (!l) continue;
    let m;
    if ((m = l.match(/^### (.*)/))) { b.append(el("h4", "sub", m[1])); ul = null; continue; }
    if (l.startsWith("- ")) {
      if (!ul) { ul = el("ul", "sources-list"); b.append(ul); }
      const li = el("li"); li.innerHTML = inline(l.slice(2)); ul.append(li);
    }
  }
  return b;
}

function renderBriefing(md) {
  const { head, secs } = splitSections(md);
  const root = el("div", "report");
  root.append(buildHead(head));
  const grid = el("div", "report-grid");
  for (const s of secs) {
    const t = s.title;
    let card;
    if (/^Executive summary/i.test(t)) { card = mkCard(t, { cls: "hl" }); card.append(Object.assign(el("div", "card-b"), {})); card.lastChild.append(prose(s.body)); }
    else if (/patient panel/i.test(t)) { card = mkCard(t); card.append(buildPanel(s.body)); }
    else if (/^Patient context/i.test(t)) {
      card = mkCard(t, { cls: "clin" });
      const flag = el("div", "clin-flag"); flag.append(icon("shield"), document.createTextNode("Clinician review required. Decision support only."));
      card.append(flag);
      const b = el("div", "card-b"); b.append(prose(s.body.replace(/^> .*\n?/m, ""))); card.append(b);
    }
    else if (/^Sources/i.test(t)) { card = mkCard(t, { collapsible: true, open: false }); card.append(buildSources(s.body)); }
    else if (/^Gaps/i.test(t)) { card = mkCard(t, { cls: "warn" }); const b = el("div", "card-b"); b.append(prose(s.body)); card.append(b); }
    else if (/removed by the verifier|run log/i.test(t)) { card = mkCard(t, { collapsible: true }); const b = el("div", "card-b"); b.append(prose(s.body)); card.append(b); }
    else { card = mkCard(t); const b = el("div", "card-b"); b.append(prose(s.body)); card.append(b); }
    grid.append(card);
  }
  root.append(grid);
  return root;
}

/* ---------- chat messages ---------- */
function kindClass(type) {
  if (type === "refusal" || type === "error") return "refusal";
  if (type === "followup") return "followup";
  return "";
}
function hideHero() { const h = $("hero"); if (h) h.hidden = true; }
function scrollToEl(node) { node.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" }); }

function addMessage(text, who, extra = "", traceUrl = null) {
  hideHero();
  const row = el("div", `row ${who}`);
  const bubble = el("div", "bubble " + extra);
  if (who === "user") {
    bubble.textContent = text;
  } else {
    const av = el("span", "bot-av"); av.setAttribute("aria-hidden", "true"); av.append(icon("logo"));
    row.append(av);
    if (isBriefing(text)) { bubble.classList.add("report"); bubble.append(renderBriefing(text)); }
    else bubble.append(prose(text));
    if (traceUrl && /^https:\/\//.test(traceUrl)) {
      const a = el("a", "trace", "View trace");
      a.href = traceUrl; a.target = "_blank"; a.rel = "noopener noreferrer";
      a.prepend(icon("link"));
      bubble.append(a);
    }
  }
  row.append(bubble);
  $("chat").append(row);
  scrollToEl(row);
  return row;
}

/* ---------- progress stepper ---------- */
const BRIEFING_STEPS = [
  { t: "Intake and plan", s: "Condition, subtype and planning focus locked in", at: 0 },
  { t: "Supervisor dispatches workers", s: "Deterministic routing, one worker per job", at: 2 },
  { t: "Workers run in parallel", s: "Public evidence and our own data", at: 4, workers: ["Standard of care", "Pipeline", "Landscape", "Patient panel"] },
  { t: "Synthesize the draft", s: "Schema-locked claims with source IDs", at: 11 },
  { t: "Verify: cite or drop", s: "Claims without a returned source are removed", at: 15 },
];
const FOLLOW_STEPS = [
  { t: "Research agent plans", s: "Reads your question and the briefing", at: 0 },
  { t: "Searching documents and public sources", s: "Up to six tool calls", at: 2 },
  { t: "Verifying citations", s: "Unsupported statements are removed", at: 8 },
];
const SIMPLE_STEPS = [{ t: "Working on it", s: "This usually takes a few seconds", at: 0 }];

function addProgress(kind) {
  hideHero();
  const steps = kind === "briefing" ? BRIEFING_STEPS : kind === "followup" ? FOLLOW_STEPS : SIMPLE_STEPS;
  const row = el("div", "row bot");
  const av = el("span", "bot-av"); av.setAttribute("aria-hidden", "true"); av.append(icon("logo"));
  const bubble = el("div", "bubble progress");
  const h = el("div", "progress-h");
  const title = el("strong", null, kind === "briefing" ? "Building your briefing" : kind === "followup" ? "Researching your question" : "Working");
  const elapsed = el("span", "elapsed", "0s");
  h.append(el("span", "badge badge-info badge-run", "Running"), title, elapsed);
  const ol = el("ol", "steps");
  ol.setAttribute("aria-label", "Progress");
  const nodes = steps.map((st) => {
    const li = el("li", "step");
    const dot = el("span", "dot"); dot.setAttribute("aria-hidden", "true"); dot.append(icon("check"));
    const body = el("div"); body.append(el("b", null, st.t), el("span", "sub", st.s));
    if (st.workers) { const w = el("div", "workers"); for (const n of st.workers) w.append(el("span", "worker", n)); body.append(w); }
    li.append(dot, body); ol.append(li); return li;
  });
  bubble.append(h, ol);
  if (kind === "briefing") { const sh = el("div", "shimmer-lines"); sh.append(el("i"), el("i"), el("i")); bubble.append(sh); }
  row.append(av, bubble); $("chat").append(row); scrollToEl(row);
  const t0 = Date.now();
  const tick = () => {
    const sec = (Date.now() - t0) / 1000;
    elapsed.textContent = `${Math.floor(sec)}s`;
    let active = 0; steps.forEach((st, i) => { if (sec >= st.at) active = i; });
    nodes.forEach((n, i) => { n.classList.toggle("done", i < active); n.classList.toggle("active", i === active); });
  };
  tick();
  const iv = setInterval(tick, 500);
  return { remove() { clearInterval(iv); row.remove(); } };
}

/* ---------- API with demo gate ---------- */
let gatePromise = null;
function openGate(msg, isErr) {
  const dlg = $("gate");
  $("gate-msg").textContent = msg || "This demo is protected so the model budget is not drained. Ask the presenter for the passphrase.";
  $("gate-err").hidden = !isErr;
  if (isErr) $("gate-err").textContent = "That passphrase was not accepted. Try again.";
  if (gatePromise) return gatePromise;
  gatePromise = new Promise((resolve) => {
    const form = $("gate-form");
    const onSubmit = (e) => {
      e.preventDefault();
      const v = $("gate-input").value.trim();
      if (!v) return;
      setKey(v); $("gate-input").value = "";
      form.removeEventListener("submit", onSubmit);
      dlg.close(); gatePromise = null; resolve();
    };
    form.addEventListener("submit", onSubmit);
    if (!dlg.open) dlg.showModal();
    $("gate-input").focus();
  });
  return gatePromise;
}
$("gate").addEventListener("cancel", (e) => e.preventDefault());

function identity() {
  const h = { "X-User": $("user").value.trim() };
  const k = getKey(); if (k) h["X-Demo-Key"] = k;
  return h;
}

async function request(path, init = {}, retried = false) {
  const headers = { ...(init.json ? { "Content-Type": "application/json" } : {}), ...identity() };
  let res;
  try {
    res = await fetch(API_BASE + path, { method: init.method || "GET", headers, body: init.body });
  } catch (e) {
    throw new Error("Cannot reach the server. Check your connection and try again.");
  }
  if (res.status === 429) {
    throw new Error("You are sending requests too quickly. Please wait a minute and try again.");
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 401 && data && data.detail === "demo_key_required") {
    const hadKey = !!getKey(); setKey("");
    await openGate(null, hadKey || retried);
    return request(path, init, true);
  }
  if (!res.ok) {
    let d = data.detail || data.text || `HTTP ${res.status}`;
    if (typeof d !== "string") d = JSON.stringify(d);
    throw new Error(d);
  }
  return data;
}
const api = (path, body) => request(path, body ? { method: "POST", json: true, body: JSON.stringify(body) } : {});

$("signout").addEventListener("click", () => {
  setKey("");
  resetSession();
  toast("Passphrase cleared from this tab.");
  loadAll();
});

/* ---------- state ---------- */
function busy(on) {
  busyNow = on;
  for (const id of ["send", "approve", "reject", "new", "upload"]) $(id).disabled = on;
  $("quick").hidden = on || !$("quick").children.length;
  $("review").classList.toggle("is-busy", on);
}

function badgeFor(status) {
  const b = el("span", "badge " + statusClass(status), String(status || "").replace(/_/g, " "));
  if (/running/.test(status || "")) b.classList.add("badge-run", "badge-info");
  return b;
}

function setSession(id) {
  threadId = id;
  const url = new URL(location.href);
  if (id) url.searchParams.set("session", id); else url.searchParams.delete("session");
  history.replaceState(null, "", url);
  for (const li of $("sessions").children) {
    const b = li.firstElementChild; if (b && b.dataset) b.classList.toggle("active", b.dataset.id === id);
  }
}

function setStatus(status, pendingType) {
  lastStatus = status;
  $("review").hidden = !(pendingType === "approval" || pendingType === "approval_pending");
  $("message").placeholder = PLACEHOLDERS[status] || "Enter a condition, e.g. dementia";
  const pill = $("session-status");
  pill.hidden = !status;
  pill.className = "badge " + statusClass(status);
  pill.textContent = (status || "").replace(/_/g, " ");
  const quick = $("quick"); quick.replaceChildren();
  if (status === "awaiting_answers") {
    for (const q of ["Alzheimer's, service-line, none", "Alzheimer's, infusion capacity, none", "Alzheimer's, service-line, P001"]) {
      const b = el("button", "qchip", q); b.type = "button";
      b.addEventListener("click", () => { $("message").value = q; $("message").focus(); });
      quick.append(b);
    }
  } else if (status === "done" || status === "rejected") {
    for (const q of ["What does our payer policy say about infusion coverage?", "Which sites in our market are enrolling?"]) {
      const b = el("button", "qchip", q); b.type = "button";
      b.addEventListener("click", () => { $("message").value = q; $("message").focus(); });
      quick.append(b);
    }
  }
  quick.hidden = !quick.children.length;
}

function resetSession() {
  setSession(null);
  setStatus(null, null);
  $("chat").replaceChildren($("hero"));
  $("hero").hidden = false;
  closeNav();
}

async function openSession(id) {
  closeNav();
  try {
    const v = await api(`/api/sessions/${encodeURIComponent(id)}`);
    $("chat").replaceChildren($("hero"));
    for (const m of v.messages) {
      const user = m.role === "user";
      addMessage(m.text || "", user ? "user" : "bot", user ? "" : kindClass(m.type), m.trace_url);
    }
    setSession(id);
    setStatus(v.status, v.pending && v.pending.type);
  } catch (e) {
    resetSession();
    addMessage(`Could not open that session: ${e.message}`, "bot", "refusal");
  }
}

async function send(payload, label) {
  if (busyNow) return;
  if (label) addMessage(label, "user");
  const kind = lastStatus === "awaiting_answers" ? "briefing" : (lastStatus === "done" || lastStatus === "rejected") ? "followup" : "simple";
  const waiting = addProgress(kind);
  busy(true);
  try {
    const r = await api("/api/chat", { thread_id: threadId, ...payload });
    waiting.remove();
    if (r.thread_id) setSession(r.thread_id);
    addMessage(r.text || "(no content)", "bot", kindClass(r.type), r.trace_url);
    setStatus(r.status, r.type);
    loadSessions();
    loadHistory();
  } catch (e) {
    waiting.remove();
    addMessage(`Error: ${e.message}`, "bot", "refusal");
  } finally {
    busy(false);
    $("message").focus({ preventScroll: true });
  }
}

/* ---------- side panels ---------- */
function skeletons(ul, n = 3) { ul.replaceChildren(...Array.from({ length: n }, () => { const li = el("li", "skeleton"); return li; })); }
function emptyList(ul, msg) { ul.replaceChildren(Object.assign(el("li", "empty-list", msg))); }
const fmtTime = (s) => (s || "").slice(0, 16).replace("T", " ");

async function loadSessions() {
  const ul = $("sessions");
  if (!ul.children.length) skeletons(ul);
  try {
    const items = await api("/api/sessions");
    ul.replaceChildren();
    if (!items.length) return emptyList(ul, "No sessions yet. Start a briefing.");
    for (const s of items) {
      const li = el("li");
      const b = el("button", "item", null); b.type = "button"; b.dataset.id = s.thread_id;
      b.append(el("span", "item-title", s.condition || s.title || "(untitled)"));
      const meta = el("span", "item-meta"); meta.append(badgeFor(s.status), el("span", null, fmtTime(s.updated_at)));
      b.append(meta);
      b.classList.toggle("active", s.thread_id === threadId);
      b.addEventListener("click", () => openSession(s.thread_id));
      li.append(b); ul.append(li);
    }
  } catch (e) { emptyList(ul, e.message); }
}

async function loadDocuments() {
  const ul = $("documents");
  if (!ul.children.length) skeletons(ul);
  try {
    const docs = await api("/api/documents");
    ul.replaceChildren();
    if (!docs.length) return emptyList(ul, "No documents in the library.");
    for (const d of docs) {
      const li = el("li"); const b = el("div", "item");
      b.append(el("span", "item-title", d.title || d.doc_id));
      const meta = el("span", "item-meta");
      meta.append(el("span", "chip chip-doc", d.doc_type || "document"), el("span", null, `${d.chunks} chunks`));
      if (d.synthetic) meta.append(el("span", null, "synthetic"));
      b.append(meta); li.append(b); ul.append(li);
    }
  } catch (e) { emptyList(ul, e.message); }
}

async function loadHistory() {
  const ul = $("history");
  if (!ul.children.length) skeletons(ul);
  try {
    const items = await api("/api/briefings");
    ul.replaceChildren();
    if (!items.length) return emptyList(ul, "No saved briefings yet.");
    for (const b of items) {
      const li = el("li"); const btn = el("button", "item"); btn.type = "button";
      btn.append(el("span", "item-title", `${b.condition} (${b.subtype})`));
      const meta = el("span", "item-meta"); meta.append(badgeFor(b.status), el("span", null, `${b.created_by} | ${fmtTime(b.created_at)}`));
      btn.append(meta);
      btn.addEventListener("click", async () => {
        closeNav();
        try { const rec = await api(`/api/briefings/${b.id}`); addMessage(rec.markdown, "bot"); }
        catch (e) { toast(e.message, true); }
      });
      li.append(btn); ul.append(li);
    }
  } catch (e) { emptyList(ul, e.message); }
}

async function loadAudit() {
  const box = $("audit-log");
  box.replaceChildren(el("div", "skeleton"));
  try {
    const rows = await api("/api/audit");
    box.replaceChildren();
    if (!rows.length) return box.append(el("div", "empty-list", "No patient-context access recorded."));
    const t = el("table"); const thead = el("thead"); const hr = el("tr");
    for (const h of ["Time", "User", "Patient", "Action"]) hr.append(el("th", null, h));
    thead.append(hr); t.append(thead);
    const tb = el("tbody");
    for (const r of rows.slice(0, 25)) {
      const tr = el("tr");
      tr.append(el("td", "mono", (r.ts || "").slice(5, 16).replace("T", " ")), el("td", null, r.user), el("td", "mono", r.patient_ref), el("td", null, r.action));
      tb.append(tr);
    }
    t.append(tb); box.append(t);
  } catch (e) { box.replaceChildren(el("div", "empty-list", e.message)); }
}

function loadAll() { loadSessions(); loadDocuments(); loadHistory(); }

/* ---------- tabs ---------- */
const TABS = ["sessions", "history", "docs", "audit"];
function selectTab(name, focus) {
  for (const t of TABS) {
    const btn = $("tab-" + t), on = t === name;
    btn.setAttribute("aria-selected", on ? "true" : "false");
    btn.tabIndex = on ? 0 : -1;
    $("panel-" + t).hidden = !on;
    if (on && focus) btn.focus();
  }
  if (name === "audit") loadAudit();
}
for (const t of TABS) {
  $("tab-" + t).addEventListener("click", () => selectTab(t));
  $("tab-" + t).addEventListener("keydown", (e) => {
    const i = TABS.indexOf(t);
    if (e.key === "ArrowRight") { e.preventDefault(); selectTab(TABS[(i + 1) % TABS.length], true); }
    if (e.key === "ArrowLeft") { e.preventDefault(); selectTab(TABS[(i + TABS.length - 1) % TABS.length], true); }
  });
}

/* ---------- mobile drawer ---------- */
function openNav() { $("app").classList.add("nav-open"); $("scrim").hidden = false; $("menu").setAttribute("aria-expanded", "true"); }
function closeNav() { $("app").classList.remove("nav-open"); $("scrim").hidden = true; $("menu").setAttribute("aria-expanded", "false"); }
$("menu").addEventListener("click", openNav);
$("scrim").addEventListener("click", closeNav);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeNav(); });

/* ---------- events ---------- */
$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("message").value.trim();
  if (!text) return;
  $("message").value = "";
  send({ message: text }, text);
});
$("approve").addEventListener("click", () => send({ action: "approve" }, "Approve"));
$("reject").addEventListener("click", () => send({ action: "reject" }, "Reject"));
$("new").addEventListener("click", () => { resetSession(); $("message").focus(); });
$("hero").addEventListener("click", (e) => {
  const b = e.target.closest(".example"); if (!b || busyNow) return;
  send({ message: b.dataset.q }, b.dataset.q);
});
function syncAvatar() { $("avatar").textContent = ($("user").value.trim()[0] || "?"); }
$("user").addEventListener("input", syncAvatar);
$("user").addEventListener("change", () => { resetSession(); loadSessions(); loadHistory(); });
$("refresh").addEventListener("click", loadHistory);
$("audit").addEventListener("click", loadAudit);
$("upload-file").addEventListener("change", () => {
  const f = $("upload-file").files[0];
  $("upload-name").textContent = f ? f.name : "Choose a document to add";
});
$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("upload-file").files[0];
  if (!file) { $("upload-status").textContent = "Choose a .md, .txt, .pdf, .xlsx or .csv file first."; return; }
  const form = new FormData();
  form.append("file", file);
  $("upload-status").textContent = `Uploading ${file.name}...`;
  busy(true);
  try {
    const r = await request("/api/documents", { method: "POST", body: form });
    const flags = (r.flags || []).length ? ` Flags: ${r.flags.join(", ")}.` : "";
    $("upload-status").textContent = `Added ${r.doc_id} (${r.chunks} chunks).${flags}`;
    $("upload-file").value = ""; $("upload-name").textContent = "Choose a document to add";
    toast("Document added to the library.");
    loadDocuments();
  } catch (err) {
    $("upload-status").textContent = `Upload rejected: ${err.message}`;
  } finally {
    busy(false);
  }
});

/* ---------- init ---------- */
syncAvatar();
const initial = new URLSearchParams(location.search).get("session");
loadSessions().then(() => { if (initial && /^[a-f0-9]{32}$/.test(initial)) openSession(initial); });
loadDocuments();
loadHistory();
