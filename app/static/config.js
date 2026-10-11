// Deployment config. When the UI is hosted separately (e.g. Netlify), set API_BASE to the API origin,
// e.g. "https://briefing-api.example.com" (no trailing slash). Empty string = same origin.
window.APP_CONFIG = { API_BASE: "" };
// Apply the saved theme before first paint to avoid a flash.
(function () {
  try {
    var t = localStorage.getItem("theme");
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  } catch (e) { /* storage unavailable */ }
})();
