"use strict";
const $ = (id) => document.getElementById(id);
let threadId = null;

const PLACEHOLDERS = {
  awaiting_answers: "Answer the questions, e.g. Alzheimer's, service-line, none",
  awaiting_approval: "Approve or reject the draft above",
  done: "Ask a follow-up question about this briefing",
  rejected: "Ask a follow-up question about this briefing",
};

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// Minimal markdown renderer. Escapes everything first; only https links are turned into anchors.
function inline(s) {
  return escapeHtml(s)
    .replace(/\[([^\]]+)\]\((https:\/\/[^\s)]+)\)/g, (_, t, u) => `<a href="${u}" target="_blank" rel="noopener noreferrer">${t}</a>`)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/~~([^~]+)~~/g, "<del>$1</del>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/(^|\s)_([^_]+)_(?=\s|$|[.,;:])/g, "$1<em>$2</em>");
}

function renderMarkdown(md) {
  const lines = md.split("\n");
  let html = "", list = null, table = [];
  const flushList = () => { if (list) { html += `</${list}>`; list = null; } };
  const flushTable = () => {
    if (!table.length) return;
    const rows = table.filter((r) => !/^\|\s*-/.test(r)).map((r) => r.replace(/^\||\|$/g, "").split("|"));
    html += "<table>" + rows.map((cells, i) =>
      "<tr>" + cells.map((c) => (i === 0 ? `<th>${inline(c.trim())}</th>` : `<td>${inline(c.trim())}</td>`)).join("") + "</tr>").join("") + "</table>";
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

function kindClass(type) {
  if (type === "refusal" || type === "error") return "refusal";
  if (type === "followup") return "followup";
  return "";
}

function addMessage(text, who, extra = "", traceUrl = null) {
  const div = document.createElement("div");
  div.className = `msg ${who} ${extra}`;
  if (who === "user") div.textContent = text; else div.innerHTML = renderMarkdown(text);
  if (traceUrl && /^https:\/\//.test(traceUrl)) {
    const a = document.createElement("a");
    a.className = "trace";
    a.href = traceUrl;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = "View trace";
    div.appendChild(a);
  }
  $("chat").appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "start" });
  return div;
}

function identity() {
  return { "X-User": $("user").value.trim() };
}

async function parse(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.text || `HTTP ${res.status}`);
  return data;
}

async function api(path, body) {
  return parse(await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json", ...identity() },
    body: body ? JSON.stringify(body) : undefined,
  }));
}

function busy(on) {
  for (const id of ["send", "approve", "reject", "new", "upload"]) $(id).disabled = on;
}

function setSession(id) {
  threadId = id;
  const url = new URL(location.href);
  if (id) url.searchParams.set("session", id); else url.searchParams.delete("session");
  history.replaceState(null, "", url);
  for (const li of $("sessions").children) li.classList.toggle("active", li.dataset.id === id);
}

function setStatus(status, pendingType) {
  $("review").hidden = !(pendingType === "approval" || pendingType === "approval_pending");
  $("message").placeholder = PLACEHOLDERS[status] || "Enter a condition, e.g. dementia";
  const pill = $("session-status");
  pill.hidden = !status;
  pill.textContent = (status || "").replace("_", " ");
}

function resetSession() {
  setSession(null);
  setStatus(null, null);
  $("chat").replaceChildren();
}

async function openSession(id) {
  try {
    const v = await api(`/api/sessions/${encodeURIComponent(id)}`);
    $("chat").replaceChildren();
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
  if (label) addMessage(label, "user");
  const waiting = addMessage("_Working... a full briefing takes about a minute._", "bot");
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
  }
}

async function loadSessions() {
  try {
    const items = await api("/api/sessions");
    const ul = $("sessions");
    ul.replaceChildren();
    for (const s of items) {
      const li = document.createElement("li");
      li.dataset.id = s.thread_id;
      li.textContent = s.condition || s.title || "(untitled)";
      const small = document.createElement("small");
      small.textContent = `${(s.status || "").replace("_", " ")} | ${s.updated_at.slice(0, 16).replace("T", " ")}`;
      li.appendChild(small);
      li.classList.toggle("active", s.thread_id === threadId);
      li.addEventListener("click", () => openSession(s.thread_id));
      ul.appendChild(li);
    }
  } catch (e) { $("sessions").replaceChildren(); }
}

async function loadDocuments() {
  const ul = $("documents");
  try {
    const docs = await api("/api/documents");
    ul.replaceChildren();
    for (const d of docs) {
      const li = document.createElement("li");
      li.textContent = d.title || d.doc_id;
      const small = document.createElement("small");
      small.textContent = `${d.doc_type || "document"} | ${d.chunks} chunks${d.synthetic ? " | synthetic" : ""}`;
      li.appendChild(small);
      ul.appendChild(li);
    }
  } catch (e) {
    ul.replaceChildren();
    $("upload-status").textContent = e.message;
  }
}

async function loadHistory() {
  try {
    const items = await api("/api/briefings");
    const ul = $("history");
    ul.replaceChildren();
    for (const b of items) {
      const li = document.createElement("li");
      li.textContent = `${b.condition} (${b.subtype})`;
      const small = document.createElement("small");
      small.textContent = `${b.status} | ${b.created_by} | ${b.created_at.slice(0, 16).replace("T", " ")}`;
      li.appendChild(small);
      li.addEventListener("click", async () => {
        const rec = await api(`/api/briefings/${b.id}`);
        addMessage(rec.markdown, "bot");
      });
      ul.appendChild(li);
    }
  } catch (e) { /* identity missing: ignore until set */ }
}

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
$("user").addEventListener("change", () => { resetSession(); loadSessions(); loadHistory(); });
$("refresh").addEventListener("click", loadHistory);
$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("upload-file").files[0];
  if (!file) { $("upload-status").textContent = "Choose a .md, .txt or .pdf file first."; return; }
  const form = new FormData();
  form.append("file", file);
  $("upload-status").textContent = `Uploading ${file.name}...`;
  busy(true);
  try {
    const r = await parse(await fetch("/api/documents", { method: "POST", headers: identity(), body: form }));
    const flags = (r.flags || []).length ? ` Flags: ${r.flags.join(", ")}.` : "";
    $("upload-status").textContent = `Added ${r.doc_id} (${r.chunks} chunks).${flags}`;
    $("upload-file").value = "";
    loadDocuments();
  } catch (err) {
    $("upload-status").textContent = `Upload rejected: ${err.message}`;
  } finally {
    busy(false);
  }
});
$("audit").addEventListener("click", async () => {
  try {
    const rows = await api("/api/audit");
    $("audit-log").textContent = rows.slice(0, 15).map((r) => `${r.ts.slice(5, 16)} ${r.user} ${r.patient_ref} ${r.action}`).join("\n") || "(empty)";
  } catch (e) { $("audit-log").textContent = e.message; }
});

const initial = new URLSearchParams(location.search).get("session");
loadSessions().then(() => { if (initial && /^[a-f0-9]{32}$/.test(initial)) openSession(initial); });
loadDocuments();
loadHistory();
