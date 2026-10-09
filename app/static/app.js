"use strict";
const $ = (id) => document.getElementById(id);
let threadId = null;

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
    else if (line.trim() === "") { flushList(); }
    else { flushList(); html += `<p>${inline(line)}</p>`; }
  }
  flushList(); flushTable();
  return html;
}

function addMessage(text, who, extra = "") {
  const div = document.createElement("div");
  div.className = `msg ${who} ${extra}`;
  if (who === "user") div.textContent = text; else div.innerHTML = renderMarkdown(text);
  $("chat").appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "start" });
  return div;
}

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json", "X-User": $("user").value.trim() },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.text || `HTTP ${res.status}`);
  return data;
}

function busy(on) {
  for (const id of ["send", "approve", "reject", "new"]) $(id).disabled = on;
}

async function send(payload, label) {
  if (label) addMessage(label, "user");
  const waiting = addMessage("_Working... the briefing step takes about a minute._", "bot");
  busy(true);
  try {
    const r = await api("/api/chat", { thread_id: threadId, ...payload });
    waiting.remove();
    threadId = r.thread_id || threadId;
    addMessage(r.text || "(no content)", "bot", r.type === "refusal" ? "refusal" : "");
    $("review").hidden = !(r.type === "approval" || r.type === "approval_pending");
    if (r.type === "done" || r.type === "refusal") threadId = null;
    loadHistory();
  } catch (e) {
    waiting.remove();
    addMessage(`Error: ${e.message}`, "bot", "refusal");
  } finally {
    busy(false);
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
$("new").addEventListener("click", () => { threadId = null; $("review").hidden = true; $("chat").replaceChildren(); });
$("refresh").addEventListener("click", loadHistory);
$("audit").addEventListener("click", async () => {
  try {
    const rows = await api("/api/audit");
    $("audit-log").textContent = rows.slice(0, 15).map((r) => `${r.ts.slice(5, 16)} ${r.user} ${r.patient_ref} ${r.action}`).join("\n") || "(empty)";
  } catch (e) { $("audit-log").textContent = e.message; }
});
loadHistory();
