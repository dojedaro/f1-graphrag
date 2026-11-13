"use strict";

// Static showcase: everything comes from data/bundle.json (built by `python -m f1rag.site`).

const SYSTEMS = ["vector", "vector-agent", "graph", "hybrid"];
const LABEL = { vector: "Vector RAG", "vector-agent": "Vector agent", graph: "Graph agent", hybrid: "Hybrid" };
const COLOR = { vector: "#5b9bff", "vector-agent": "#3fd0c9", graph: "#ff8a3d", hybrid: "#b48cff" };
let HOPS = ["1-hop", "2-hop", "3-hop", "aggregation"];
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (x) => `${Math.round(x * 100)}%`;

let B;
let current = null;

async function boot() {
  try {
    B = await (await fetch("data/bundle.json")).json();
  } catch (e) {
    $("#headline").textContent = "Could not load data/bundle.json.";
    return;
  }
  if (B.summary.graph["held-out"]) HOPS = [...HOPS, "held-out"];
  $("#model").textContent = B.model;
  renderHero();
  renderChart();
  renderCosts();
  $("#f-hops").innerHTML = ["all", ...HOPS].map((h) => `<option value="${h}">${h === "all" ? "all types" : h}</option>`).join("");
  $("#f-hops").addEventListener("change", renderList);
  $("#f-outcome").addEventListener("change", renderList);
  renderList();
}

// Totals over every question shown, main set plus held-out.
const total = (s) => {
  const c = s.all.correct + (s["held-out"]?.correct || 0), n = s.all.n + (s["held-out"]?.n || 0);
  return { correct: c, n, accuracy: n ? c / n : 0 };
};

function renderHero() {
  const S = B.summary;
  const g = S.graph, v = S.vector, h = S.hybrid;
  const multi = (s) => {
    const keys = ["2-hop", "3-hop", "aggregation"].filter((k) => s[k]);
    const c = keys.reduce((x, k) => x + s[k].correct, 0), n = keys.reduce((x, k) => x + s[k].n, 0);
    return n ? c / n : 0;
  };
  const one = (s) => (s["1-hop"] ? s["1-hop"].accuracy : 0);
  const va = S["vector-agent"];
  const all = (x) => (x ? total(x).accuracy : 0);
  $("#headline").innerHTML =
    `Plain vector search answers <b>${pct(all(v))}</b> of 160 Formula 1 questions. Letting it search repeatedly ` +
    `in an agent loop lifts that to <b>${pct(all(va))}</b>. A graph agent with typed tools answers <b>${pct(all(g))}</b>, ` +
    `at about 1/${Math.round(va ? va.cost_usd / S.graph.cost_usd : 1)} of the vector agent's cost` +
    (va ? ` ($${S.graph.cost_usd.toFixed(2)} vs $${va.cost_usd.toFixed(2)}).` : ".") +
    ` On structured questions, the graph's edge is reliability and efficiency more than capability.`;
  const stats = [
    [B.questions.length, "questions, answers computed from data"],
    [pct(total(g).accuracy), "graph agent"],
    [S["vector-agent"] ? pct(total(S["vector-agent"]).accuracy) : "-", "vector agent (search loop)"],
    [pct(total(v).accuracy), "vector RAG (one retrieval)"],
  ];
  $("#hero-stats").innerHTML = stats.map(([vv, k]) => `<div class="stat"><div class="v">${vv}</div><div class="k">${esc(k)}</div></div>`).join("");
}

function renderChart() {
  const S = B.summary;
  const W = 1000, H = 330, left = 50, top = 20, bottom = 50, groupW = (W - left - 20) / HOPS.length, barW = 36;
  const y = (a) => top + (1 - a) * (H - top - bottom);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy by question type and system">`;
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    svg += `<line class="axis" x1="${left}" x2="${W - 20}" y1="${y(t)}" y2="${y(t)}"/><text x="${left - 8}" y="${y(t) + 4}" text-anchor="end">${t * 100}%</text>`;
  }
  HOPS.forEach((hop, gi) => {
    const x0 = left + gi * groupW + (groupW - barW * SYSTEMS.length - 8 * (SYSTEMS.length - 1)) / 2;
    SYSTEMS.forEach((s, si) => {
      const cell = S[s]?.[hop];
      if (!cell) return;
      const x = x0 + si * (barW + 8);
      svg += `<rect x="${x}" y="${y(cell.accuracy)}" width="${barW}" height="${y(0) - y(cell.accuracy)}" rx="4" fill="${COLOR[s]}"><title>${LABEL[s]}, ${hop}: ${cell.correct}/${cell.n}</title></rect>`;
      if (cell.ci) {
        const cx = x + barW / 2;
        svg += `<line class="ci" x1="${cx}" x2="${cx}" y1="${y(cell.ci[0])}" y2="${y(cell.ci[1])}"/><line class="ci" x1="${cx - 6}" x2="${cx + 6}" y1="${y(cell.ci[1])}" y2="${y(cell.ci[1])}"/><line class="ci" x1="${cx - 6}" x2="${cx + 6}" y1="${y(cell.ci[0])}" y2="${y(cell.ci[0])}"/>`;
      }
      svg += `<text class="val" x="${x + barW / 2}" y="${Math.min(y(cell.accuracy), y(cell.ci ? cell.ci[1] : cell.accuracy)) - 8}" text-anchor="middle">${pct(cell.accuracy)}</text>`;
    });
    svg += `<text class="label" x="${left + gi * groupW + groupW / 2}" y="${H - 18}" text-anchor="middle">${hop}</text>`;
  });
  svg += "</svg>";
  const legend = `<div class="legend">${SYSTEMS.map((s) => `<span><i class="dot ${s}"></i>${LABEL[s]}</span>`).join("")}</div>`;
  $("#chart").innerHTML = legend + svg;
}

function renderCosts() {
  const S = B.summary;
  $("#cost-table").innerHTML = `<table><thead><tr><th>System</th><th class="num">Correct</th><th class="num">Tool calls per question</th>
    <th class="num">Seconds per question</th><th class="num">Total cost (${B.questions.length} questions)</th></tr></thead><tbody>${SYSTEMS.map(
      (s) => `<tr><td><i class="dot ${s}"></i> ${LABEL[s]}</td><td class="num">${total(S[s]).correct}/${total(S[s]).n}</td>
      <td class="num">${S[s].mean_tool_calls}</td><td class="num">${S[s].mean_seconds}</td><td class="num">$${S[s].cost_usd.toFixed(3)}</td></tr>`,
    ).join("")}</tbody></table>`;
}

function filtered() {
  const hop = $("#f-hops").value, outcome = $("#f-outcome").value;
  return B.questions.filter((q) => {
    if (hop !== "all" && q.hops !== hop) return false;
    const r = q.results, ok = (s) => r[s]?.correct;
    if (outcome === "graph-only") return ok("graph") && !ok("vector");
    if (outcome === "vector-only") return ok("vector") && !ok("graph");
    if (outcome === "all-wrong") return SYSTEMS.every((s) => !ok(s));
    return true;
  });
}

function renderList() {
  const qs = filtered();
  $("#count").textContent = `${qs.length} question${qs.length === 1 ? "" : "s"}`;
  $("#qlist").innerHTML = qs
    .map(
      (q) => `<li data-q="${esc(q.qid)}"><div>${esc(q.question)}</div><div class="marks"><span class="pill">${q.hops}</span>${SYSTEMS.map(
        (s) => `<span class="mark ${q.results[s]?.correct ? "ok" : "bad"}" title="${LABEL[s]}">${s === "vector-agent" ? "VA" : s[0].toUpperCase()}${q.results[s]?.correct ? "✓" : "✗"}</span>`,
      ).join("")}</div></li>`,
    )
    .join("");
  for (const li of document.querySelectorAll("#qlist li")) li.addEventListener("click", () => show(li.dataset.q));
  if (qs.length && !qs.some((q) => q.qid === current)) show(qs[0].qid);
  else if (!qs.length) $("#detail").innerHTML = `<p class="muted">No questions match.</p>`;
  else show(current);
}

function show(qid) {
  current = qid;
  for (const li of document.querySelectorAll("#qlist li")) li.classList.toggle("active", li.dataset.q === qid);
  const q = B.questions.find((x) => x.qid === qid);
  const col = (s) => {
    const r = q.results[s];
    if (!r) return `<div class="col ${s}"><h4>${LABEL[s]}</h4><p class="muted">not run</p></div>`;
    let body;
    if (s === "vector") {
      const hits = r.trace[0]?.retrieved || [];
      body = `<p class="muted" style="margin:0 0 4px">Retrieved (cosine similarity)</p><ol class="steps">${hits
        .map((h) => `<li>${esc(h.title)} <span class="muted">${h.score.toFixed(2)}</span></li>`)
        .join("")}</ol>`;
    } else {
      const steps = r.trace.filter((t) => t.tool);
      body = steps.length
        ? `<ol class="steps">${steps
            .map(
              (t) => `<li><code>${esc(t.tool)}(${esc(Object.values(t.input).join(", "))})</code><details><summary>result</summary><pre>${esc(prettify(t.output))}</pre></details></li>`,
            )
            .join("")}</ol>`
        : `<p class="muted">No tool calls.</p>`;
    }
    return `<div class="col ${s}"><h4>${LABEL[s]} <span class="mark ${r.correct ? "ok" : "bad"}">${r.correct ? "correct" : "wrong"}</span></h4>
      <div class="answer ${r.correct ? "ok" : "bad"}">${esc(r.answer || "(no answer)")}</div>${body}
      <div class="meta">${r.seconds}s · ${r.tokens.toLocaleString()} tokens · $${r.cost_usd.toFixed(4)}${r.error ? ` · ${esc(r.error)}` : ""}</div></div>`;
  };
  $("#detail").innerHTML = `<p class="q">${esc(q.question)}</p><p class="gold">Correct answer: <b>${esc(q.gold_text)}</b> · <span class="pill">${q.hops}</span> <span class="pill">${esc(q.template.replaceAll("_", " "))}</span></p>
    <div class="cols">${SYSTEMS.map(col).join("")}</div>`;
}

function prettify(text) {
  try {
    return JSON.stringify(JSON.parse(text), null, 1);
  } catch {
    return text;
  }
}

boot();
