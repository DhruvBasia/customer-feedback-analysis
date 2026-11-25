/* Customer Feedback Analysis - page behaviour.
   The page asks the Flask server (app.py) for data with fetch() and draws the answers. */

const COLORS = { negative: "#C0392B", neutral: "#8A8F98", positive: "#2E8B57", blue: "#2F6DB5" };
const SENTIMENTS = ["negative", "neutral", "positive"];
const $ = (id) => document.getElementById(id);
const pct = (v, digits = 0) => (v * 100).toFixed(digits) + "%";
const num = (v) => Number(v).toLocaleString("en-IN");

const PLOT_CONFIG = { displayModeBar: false, responsive: true };
const baseLayout = (extra = {}) => Object.assign({
  font: { family: '"Segoe UI", system-ui, sans-serif', size: 13, color: "#16202E" },
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  margin: { l: 10, r: 16, t: 8, b: 10 },
  xaxis: { automargin: true, gridcolor: "#E9EDF2", zeroline: false },
  yaxis: { automargin: true, gridcolor: "#E9EDF2", zeroline: false },
  hoverlabel: { font: { family: '"Segoe UI", sans-serif' } },
}, extra);

/* ---------- small helpers ---------- */
async function getJSON(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "Something went wrong. Please try again.");
  return data;
}

function el(tag, className, text) {            // create an element safely (no HTML injection)
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function drawBars(container, rows) {            // rows: [{name, value, kind}]
  container.replaceChildren();
  rows.forEach((row) => {
    const line = el("div", "bar-row");
    const track = el("div", "track");
    const fill = el("div", "fill " + (row.kind || ""));
    track.append(fill);
    line.append(el("div", "name", row.name), track, el("div", "num", pct(row.value)));
    container.append(line);
    requestAnimationFrame(() => { fill.style.width = pct(row.value, 1); });
  });
}

function showError(node, message) {
  node.textContent = message || "";
  node.hidden = !message;
}

/* ---------- tabs ---------- */
const loaded = { analyse: true, overview: false, themes: false, model: false };
const loaders = { overview: loadOverview, themes: loadThemes, model: loadModel };
let currentTab = "analyse";

function showTab(name) {
  currentTab = name;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  $("airline-filter").classList.toggle("off", !(name === "overview" || name === "themes"));
  if (!loaded[name]) { loaded[name] = true; loaders[name](); }
  window.scrollTo({ top: 0 });
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));

$("airline").addEventListener("change", () => {
  loaded.overview = loaded.themes = false;
  if (loaders[currentTab] && currentTab !== "model") { loaded[currentTab] = true; loaders[currentTab](); }
});
const airlineQuery = () => "airline=" + encodeURIComponent($("airline").value);

/* ---------- analyse one message ---------- */
async function analyse() {
  const text = $("message").value.trim();
  showError($("analyse-error"), "");
  if (!text) { showError($("analyse-error"), "Please type a message first."); return; }
  $("analyse-btn").disabled = true;
  try {
    const r = await getJSON("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
    });
    $("result-empty").hidden = true;
    $("result-body").hidden = false;

    $("r-sentiment").textContent = r.sentiment;
    $("r-sentiment").className = "badge " + r.sentiment;
    $("r-confidence").textContent = pct(r.confidence);
    $("r-theme").textContent = r.theme || "Not a complaint";
    $("r-warn").hidden = r.confidence >= 0.5;

    drawBars($("r-sent-bars"), SENTIMENTS.map((s) => ({ name: s, value: r.probabilities[s], kind: s })));
    $("r-theme-box").hidden = !r.theme;
    if (r.theme) drawBars($("r-theme-bars"), r.theme_probabilities.map((t) => ({ name: t.theme, value: t.probability })));

    $("r-words-box").hidden = !r.words.length;
    $("r-words").replaceChildren(...r.words.map((w) => el("span", "word " + r.sentiment, w)));
  } catch (err) {
    showError($("analyse-error"), err.message);
  } finally {
    $("analyse-btn").disabled = false;
  }
}
$("analyse-btn").addEventListener("click", analyse);
$("message").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyse(); });
document.querySelectorAll("[data-example]").forEach((chip) => chip.addEventListener("click", () => {
  $("message").value = chip.dataset.example;
  analyse();
}));

/* ---------- analyse a file ---------- */
let batchRows = [];
$("file").addEventListener("change", async () => {
  const file = $("file").files[0];
  if (!file) return;
  $("file-name").textContent = file.name;
  showError($("file-error"), "");
  $("batch").hidden = true;
  const form = new FormData();
  form.append("file", file);
  try {
    const r = await getJSON("/api/predict-file", { method: "POST", body: form });
    batchRows = r.rows;
    $("batch-kpis").replaceChildren(...SENTIMENTS.map((s) => {
      const tile = el("div", "kpi " + s.slice(0, 3));
      tile.append(el("div", "label", s), el("div", "value", num(r.summary[s])),
                  el("div", "sub", pct(r.summary[s] / r.total) + " of " + num(r.total) + " messages"));
      return tile;
    }));
    const body = $("batch-table").querySelector("tbody");
    body.replaceChildren(...r.rows.slice(0, 200).map((row) => {
      const tr = el("tr");
      const sentiment = el("td");
      sentiment.append(el("span", "pill " + row.sentiment, row.sentiment));
      tr.append(el("td", "", row.text), sentiment,
                el("td", "num", row.confidence == null ? "" : pct(row.confidence)), el("td", "", row.theme));
      return tr;
    }));
    $("batch-note").textContent = r.total > 200 ? "Showing the first 200 rows; the download has all " + num(r.total) + "." : "";
    $("batch").hidden = false;
  } catch (err) {
    showError($("file-error"), err.message);
  }
  $("file").value = "";
});

$("download").addEventListener("click", () => {
  const quote = (v) => '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"';
  const lines = ["text,sentiment,confidence,complaint_theme"]
    .concat(batchRows.map((r) => [r.text, r.sentiment, r.confidence, r.theme].map(quote).join(",")));
  const link = el("a");
  link.href = URL.createObjectURL(new Blob(["﻿" + lines.join("\r\n")], { type: "text/csv" }));
  link.download = "feedback_predictions.csv";
  link.click();
  URL.revokeObjectURL(link.href);
});

/* ---------- overview ---------- */
async function loadOverview() {
  const d = await getJSON("/api/overview?" + airlineQuery());
  $("k-messages").textContent = num(d.messages);
  $("k-period").textContent = d.period[0] + " to " + d.period[1];
  $("k-negative").textContent = pct(d.share.negative, 1);
  $("k-positive").textContent = pct(d.share.positive, 1);
  $("k-theme").textContent = d.top_theme;

  Plotly.react("chart-daily", SENTIMENTS.map((s) => ({
    x: d.daily.dates, y: d.daily[s], name: s, type: "scatter", mode: "lines+markers",
    line: { color: COLORS[s], width: 2.5 }, marker: { size: 7 },
    hovertemplate: "%{y:,} " + s + "<extra></extra>",
  })), baseLayout({
    hovermode: "x unified", legend: { orientation: "h", y: 1.12, x: 0 },
    xaxis: { tickformat: "%d %b", dtick: 86400000, automargin: true, showgrid: false },
    yaxis: { rangemode: "tozero", automargin: true, gridcolor: "#E9EDF2" }, margin: { l: 10, r: 16, t: 30, b: 10 },
  }), PLOT_CONFIG);

  Plotly.react("chart-airline", SENTIMENTS.map((s) => ({
    y: d.by_airline.airlines, x: d.by_airline[s], name: s, type: "bar", orientation: "h",
    marker: { color: COLORS[s] }, text: d.by_airline[s].map((v) => (v >= 0.06 ? pct(v) : "")),
    textposition: "inside", insidetextanchor: "middle", textfont: { color: "#fff" },
    hovertemplate: "%{y}: %{x:.1%} " + s + "<extra></extra>",
  })), baseLayout({
    barmode: "stack", bargap: 0.3, legend: { orientation: "h", y: 1.12, x: 0, traceorder: "normal" },
    xaxis: { tickformat: ".0%", range: [0, 1], automargin: true, showgrid: false },
    yaxis: { automargin: true, showgrid: false }, margin: { l: 10, r: 16, t: 30, b: 10 },
  }), PLOT_CONFIG);

  Plotly.react("chart-hour", [{
    x: d.hourly.map((_, h) => h), y: d.hourly, type: "bar", marker: { color: COLORS.blue },
    hovertemplate: "%{x}:00 to %{x}:59<br>%{y:,} messages<extra></extra>",
  }], baseLayout({
    bargap: 0.25, xaxis: { dtick: 2, automargin: true, showgrid: false, title: { text: "Hour", font: { size: 12 } } },
  }), PLOT_CONFIG);
}

/* ---------- complaint themes ---------- */
async function loadThemes() {
  const d = await getJSON("/api/themes?" + airlineQuery());
  $("themes-note").textContent = num(d.total) + " negative messages with a clear theme. " + num(d.unclear) +
    ' more were labelled "Can\'t Tell" by the human labellers and are left out.';

  const themes = d.themes.slice().reverse(), counts = d.counts.slice().reverse();
  Plotly.react("chart-themes", [{
    y: themes, x: counts, type: "bar", orientation: "h", marker: { color: COLORS.blue },
    text: counts.map((c) => pct(c / d.total)), textposition: "outside", cliponaxis: false,
    hovertemplate: "%{y}: %{x:,} messages<extra></extra>",
  }], baseLayout({ bargap: 0.3, yaxis: { automargin: true, showgrid: false }, margin: { l: 10, r: 44, t: 8, b: 10 } }), PLOT_CONFIG);

  const h = d.heatmap;
  Plotly.react("chart-heat", [{
    type: "heatmap", z: h.values, x: h.themes, y: h.airlines, xgap: 3, ygap: 3, showscale: false, zmin: 0,
    colorscale: [[0, "#F3F7FC"], [1, "#5B9BD5"]],
    text: h.values.map((row) => row.map((v) => pct(v))), texttemplate: "%{text}", textfont: { color: "#16202E" },
    hovertemplate: "%{y} · %{x}<br>%{z:.1%} of its complaints<extra></extra>",
  }], baseLayout({ xaxis: { tickangle: -35, automargin: true }, yaxis: { automargin: true, autorange: "reversed" } }), PLOT_CONFIG);

  const select = $("theme-select");
  const keep = select.value;
  select.replaceChildren(...d.themes.map((t) => { const o = el("option", "", t); o.value = t; return o; }));
  if (d.themes.includes(keep)) select.value = keep;
  loadMessages();
}

async function loadMessages() {
  const theme = $("theme-select").value;
  if (!theme) { $("messages").replaceChildren(el("p", "muted", "No complaints match this filter.")); return; }
  const rows = await getJSON("/api/messages?theme=" + encodeURIComponent(theme) + "&" + airlineQuery());
  $("messages").replaceChildren(...rows.map((m) => {
    const card = el("div", "msg");
    card.append(el("div", "meta", m.airline + " · " + m.when), el("p", "", m.text));
    return card;
  }));
}
$("theme-select").addEventListener("change", loadMessages);

/* ---------- model performance ---------- */
async function loadModel() {
  const d = await getJSON("/api/model");
  const tasks = [["sentiment", "Sentiment model"], ["theme", "Complaint-theme model"]];
  $("model-cards").replaceChildren();

  tasks.forEach(([key, title]) => {
    const m = d.metrics[key];
    const card = el("div", "card");
    card.append(el("h2", "", title + ": " + m.best_model),
      el("p", "muted small", "Trained on " + num(m.n_train) + " messages, tested on " + num(m.n_test) + " messages it had never seen."));

    const metrics = el("div", "metrics");
    [["Accuracy", pct(m.accuracy, 1), ""], ["Macro F1", m.macro_f1.toFixed(2), ""],
     ["Baseline", pct(m.baseline_accuracy, 1), ""],
     ["Gain", "+" + ((m.accuracy - m.baseline_accuracy) * 100).toFixed(1) + " pts", "good"]].forEach(([label, value, kind]) => {
      const box = el("div", "metric " + kind);
      box.append(el("div", "label", label), el("div", "v", value));
      metrics.append(box);
    });
    card.append(metrics);

    card.append(el("h3", "", "Score for each class"));
    const table = el("table");
    const head = el("tr");
    ["Class", "Precision", "Recall", "F1", "Test messages"].forEach((t, i) => head.append(el("th", i ? "num" : "", t)));
    const thead = el("thead"); thead.append(head);
    const tbody = el("tbody");
    m.labels.forEach((label) => {
      const c = m.per_class[label], tr = el("tr");
      tr.append(el("td", "", label), el("td", "num", c.precision.toFixed(2)), el("td", "num", c.recall.toFixed(2)),
                el("td", "num", c["f1-score"].toFixed(2)), el("td", "num", num(c.support)));
      tbody.append(tr);
    });
    table.append(thead, tbody);
    const scroll = el("div", "table-scroll"); scroll.append(table);
    card.append(scroll);

    card.append(el("h3", "", "Confusion matrix (each row adds up to 100%)"));
    const chart = el("div", "chart");
    chart.style.height = key === "theme" ? "430px" : "300px";
    card.append(chart);
    $("model-cards").append(card);

    const share = m.confusion_matrix.map((row) => { const total = row.reduce((a, b) => a + b, 0); return row.map((v) => v / total); });
    Plotly.newPlot(chart, [{
      type: "heatmap", z: share, x: m.labels, y: m.labels, xgap: 3, ygap: 3, showscale: false, zmin: 0, zmax: 1,
      colorscale: [[0, "#F3F7FC"], [1, "#3F7FC4"]],
      text: share.map((row) => row.map((v) => pct(v))), texttemplate: "%{text}", textfont: { color: "#16202E", size: key === "theme" ? 11 : 13 },
      customdata: m.confusion_matrix,
      hovertemplate: "Actual: %{y}<br>Predicted: %{x}<br>%{customdata} messages (%{z:.1%})<extra></extra>",
    }], baseLayout({
      xaxis: { title: { text: "Predicted by the model", font: { size: 12 } }, tickangle: key === "theme" ? -35 : 0, automargin: true },
      yaxis: { title: { text: "Actual (human label)", font: { size: 12 } }, autorange: "reversed", automargin: true },
    }), PLOT_CONFIG);
  });

  const body = $("compare-table").querySelector("tbody");
  body.replaceChildren(...d.comparison.map((row) => {
    const tr = el("tr");
    const chosen = d.metrics[row.task === "Sentiment" ? "sentiment" : "theme"].best_model === row.model;
    if (chosen) tr.className = "selected";
    tr.append(el("td", "", row.task), el("td", "", row.model + (chosen ? " (selected)" : "")),
              el("td", "num", row.cv_macro_f1 == null ? "–" : row.cv_macro_f1.toFixed(3)),
              el("td", "num", pct(row.test_accuracy, 1)), el("td", "num", row.test_macro_f1.toFixed(2)));
    return tr;
  }));

  $("top-words").replaceChildren(...SENTIMENTS.map((s) => {
    const col = el("div", "top-col");
    const words = el("div", "words");
    words.append(...d.top_words[s].map((w) => el("span", "word " + s, w)));
    col.append(el("h3", "", "Signals " + s), words);
    return col;
  }));
}

$("airline-filter").classList.add("off");
