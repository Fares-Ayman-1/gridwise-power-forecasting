/* GridWise dashboard. Plain JavaScript and SVG; reads the JSON files the daily job writes to ./live/. */
"use strict";

const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => v == null || Number.isNaN(v) ? "–" : v.toLocaleString("en-GB", { minimumFractionDigits: d, maximumFractionDigits: d });
const eur = (v, d = 0) => v == null ? "–" : (v < 0 ? "−" : "") + "€" + fmt(Math.abs(v), d);
const pct = (v, d = 0) => v == null ? "–" : (v < 0 ? "−" : "") + fmt(Math.abs(v) * 100, d) + "%";
const plural = (n, word) => `${fmt(n)} ${word}${n === 1 ? "" : "s"}`;
const dayLabel = (iso) => new Date(iso + "T12:00:00Z").toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
const hourLabel = (h) => String(h).padStart(2, "0") + ":00";

function el(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}
function html(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}
async function getJSON(path) {
  try {
    const r = await fetch(path, { cache: "no-store" });
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

/* ---------- scales and axes ---------- */
function niceTicks(lo, hi, count = 5) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const start = Math.floor(lo / step) * step, ticks = [];
  for (let v = start; v <= hi + step * 0.999; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
  return ticks;
}
const linear = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

const isNarrow = (host) => (host.clientWidth || 600) < 520;
function frame(host, height, margin) {
  host.replaceChildren();
  const width = Math.max(300, host.clientWidth || 600);
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  host.appendChild(svg);
  return { svg, width, height, m: margin, x0: margin.l, x1: width - margin.r, y0: height - margin.b, y1: margin.t };
}
function yAxis(f, ticks, y, format) {
  for (const t of ticks) {
    f.svg.appendChild(el("line", { class: t === 0 ? "baseline" : "gridline", x1: f.x0, x2: f.x1, y1: y(t), y2: y(t) }));
    f.svg.appendChild(el("text", { x: f.x0 - 8, y: y(t) + 4, "text-anchor": "end" }, format(t)));
  }
}
function emptyState(host, text) {
  host.replaceChildren(html("p", "empty", text));
}

/* ---------- tooltip ---------- */
const tip = $("tooltip");
function showTip(evt, head, rows) {
  tip.replaceChildren(html("div", "t-head", head));
  for (const r of rows) {
    const row = html("div", "t-row");
    const key = html("span", "t-key");
    if (r.color) { const i = html("i"); i.style.background = r.color; key.appendChild(i); }
    key.appendChild(document.createTextNode(r.label));
    row.append(key, html("b", null, r.value));
    tip.appendChild(row);
  }
  tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
  tip.style.left = Math.max(8, x) + "px";
  tip.style.top = Math.max(8, y) + "px";
}
const hideTip = () => { tip.hidden = true; };
const cssVar = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();

/* Crosshair that snaps to the nearest x position; also reachable with the arrow keys. */
function crosshair(f, n, xOf, describe) {
  const line = el("line", { class: "cross", y1: f.y1, y2: f.y0, visibility: "hidden" });
  const hit = el("rect", { class: "hit", x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1, tabindex: 0, "aria-label": "Chart values. Use the left and right arrow keys to move." });
  let cur = -1;
  const at = (i, evt) => {
    cur = Math.max(0, Math.min(n - 1, i));
    line.setAttribute("x1", xOf(cur)); line.setAttribute("x2", xOf(cur)); line.setAttribute("visibility", "visible");
    const d = describe(cur);
    const box = f.svg.getBoundingClientRect();
    showTip(evt || { clientX: box.left + (xOf(cur) / f.width) * box.width, clientY: box.top + 20 }, d.head, d.rows);
  };
  const off = () => { line.setAttribute("visibility", "hidden"); hideTip(); cur = -1; };
  hit.addEventListener("pointermove", (e) => {
    const box = f.svg.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * f.width;
    let best = 0;
    for (let i = 1; i < n; i++) if (Math.abs(xOf(i) - px) < Math.abs(xOf(best) - px)) best = i;
    at(best, e);
  });
  hit.addEventListener("pointerleave", off);
  hit.addEventListener("blur", off);
  hit.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") { at(cur + 1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { at(cur < 0 ? n - 1 : cur - 1); e.preventDefault(); }
    else if (e.key === "Escape") off();
  });
  f.svg.append(line, hit);
}

function legend(host, items) {
  host.replaceChildren();
  for (const it of items) {
    const s = html("span");
    const i = html("i", it.kind);
    i.style.background = it.color;
    s.append(i, document.createTextNode(it.label));
    host.appendChild(s);
  }
}
function table(host, head, rows, oursRow = -1) {
  const t = html("table"), thead = html("thead"), tr = html("tr"), tb = html("tbody");
  head.forEach((h) => tr.appendChild(html("th", null, h)));
  thead.appendChild(tr);
  rows.forEach((r, i) => {
    const row = html("tr", i === oursRow ? "ours" : null);
    r.forEach((c) => row.appendChild(html("td", null, c)));
    tb.appendChild(row);
  });
  t.append(thead, tb);
  host.replaceChildren(t);
}
/* Direct labels at the right end of lines; dropped (legend stays) if any two would collide. */
function endLabels(f, items) {
  const sorted = [...items].sort((a, b) => a.y - b.y);
  for (let i = 1; i < sorted.length; i++) if (sorted[i].y - sorted[i - 1].y < 14) return;
  for (const it of items) f.svg.appendChild(el("text", { class: "label", x: f.x1 + 8, y: it.y + 4 }, it.label));
}
const path = (pts) => pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join("");

/* ---------- the forecast fan ---------- */
function drawFan(doc, score) {
  const host = $("fan"), levels = doc.quantile_levels;
  const idx = (tau) => levels.findIndex((l) => Math.abs(l - tau) < 1e-9);
  const iLo95 = idx(0.025), iLo80 = idx(0.1), iMed = idx(0.5), iHi80 = idx(0.9), iHi95 = idx(0.975);
  const q = doc.hours.map((h) => h.quantiles), n = q.length;
  const actual = score ? score.actual : null;
  const all = q.flatMap((r) => [r[iLo95], r[iHi95]]).concat(actual || []);
  const ticks = niceTicks(Math.min(...all), Math.max(...all));
  const narrow = isNarrow(host), hourStep = narrow ? 6 : 3;
  const f = frame(host, narrow ? 260 : 300, { t: 10, r: narrow ? 12 : 108, b: 28, l: 48 });
  const x = linear(0, n - 1, f.x0 + 6, f.x1 - 6), y = linear(ticks[0], ticks[ticks.length - 1], f.y0, f.y1);
  f.svg.setAttribute("aria-label", `Price forecast for ${doc.delivery_day}: median and 80% and 95% ranges for each hour` + (actual ? ", with the real prices" : ""));
  yAxis(f, ticks, y, (t) => fmt(t));
  doc.hours.forEach((h, i) => {
    if (h.local_hour % hourStep === 0 && !(i > 0 && doc.hours[i - 1].local_hour === h.local_hour))
      f.svg.appendChild(el("text", { x: x(i), y: f.y0 + 18, "text-anchor": "middle" }, hourLabel(h.local_hour)));
  });
  const band = (lo, hi, fill) => {
    const up = q.map((r, i) => [x(i), y(r[hi])]), down = q.map((r, i) => [x(i), y(r[lo])]).reverse();
    f.svg.appendChild(el("path", { d: path(up.concat(down)) + "Z", fill }));
  };
  band(iLo95, iHi95, cssVar("--band-95"));
  band(iLo80, iHi80, cssVar("--band-80"));
  const line = (vals, color) => f.svg.appendChild(el("path", { d: path(vals.map((v, i) => [x(i), y(v)])), fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  const c1 = cssVar("--series-1"), c2 = cssVar("--series-2"), surface = cssVar("--surface-1");
  line(q.map((r) => r[iMed]), c1);
  const ends = [{ y: y(q[n - 1][iMed]), label: "Forecast median" }];
  if (actual) {
    line(actual, c2);
    f.svg.appendChild(el("circle", { cx: x(n - 1), cy: y(actual[n - 1]), r: 4, fill: c2, stroke: surface, "stroke-width": 2 }));
    ends.push({ y: y(actual[n - 1]), label: "Real price" });
  }
  if (!narrow) endLabels(f, ends);   // on a phone the legend above carries identity
  crosshair(f, n, x, (i) => ({
    head: `${hourLabel(doc.hours[i].local_hour)} to ${hourLabel((doc.hours[i].local_hour + 1) % 24)}`,
    rows: [
      ...(actual ? [{ label: "Real price", value: eur(actual[i], 2), color: c2 }] : []),
      { label: "Forecast median", value: eur(q[i][iMed], 2), color: c1 },
      { label: "80% range", value: `${fmt(q[i][iLo80])} to ${fmt(q[i][iHi80])}` },
      { label: "95% range", value: `${fmt(q[i][iLo95])} to ${fmt(q[i][iHi95])}` },
    ],
  }));
  legend($("fan-legend"), [
    { kind: "line", color: c1, label: "Forecast median" },
    ...(actual ? [{ kind: "line", color: c2, label: "Real price" }] : []),
    { kind: "rect", color: cssVar("--band-80"), label: "80% range" },
    { kind: "rect", color: cssVar("--band-95"), label: "95% range" },
  ]);
  table($("fan-table"), ["Hour", "2.5%", "10%", "Median", "90%", "97.5%", "Real price"],
    doc.hours.map((h, i) => [hourLabel(h.local_hour), fmt(q[i][iLo95], 2), fmt(q[i][iLo80], 2), fmt(q[i][iMed], 2), fmt(q[i][iHi80], 2), fmt(q[i][iHi95], 2), actual ? fmt(actual[i], 2) : "not yet published"]));
  let sub = `Issued ${doc.issued_at_utc.replace("T", " ").slice(0, 16)} UTC from prices up to ${doc.inputs.prices_until}.`;
  if (score) sub += ` Mean absolute error ${eur(score.mae, 2)} per MWh; ${score.inside_90} of ${score.hours} hours inside the 90% range.`;
  $("fan-sub").textContent = sub;
}

/* ---------- the battery plan ---------- */
function drawPlan(doc, score) {
  const host = $("plan"), s = doc.schedules.mean, n = s.charge_mw.length;
  const net = s.discharge_mw.map((d, i) => d - s.charge_mw[i]);
  const P = doc.battery.power_mw;
  const narrow = isNarrow(host), hourStep = narrow ? 6 : 3;
  const f = frame(host, 170, { t: 10, r: narrow ? 36 : 108, b: 28, l: 48 });
  const x = linear(0, n - 1, f.x0 + 6, f.x1 - 6), y = linear(-P, P, f.y0, f.y1);
  f.svg.setAttribute("aria-label", `Battery plan for ${doc.delivery_day}: megawatts sold (up) or bought (down) in each hour`);
  yAxis(f, [-P, 0, P], y, (t) => fmt(t, 1));
  f.svg.appendChild(el("text", { class: "label", x: f.x1 + 8, y: y(P) + 4 }, "sell"));
  f.svg.appendChild(el("text", { class: "label", x: f.x1 + 8, y: y(-P) + 4 }, "buy"));
  doc.hours.forEach((h, i) => {
    if (h.local_hour % hourStep === 0 && !(i > 0 && doc.hours[i - 1].local_hour === h.local_hour))
      f.svg.appendChild(el("text", { x: x(i), y: f.y0 + 18, "text-anchor": "middle" }, hourLabel(h.local_hour)));
  });
  const slot = (f.x1 - f.x0 - 12) / Math.max(1, n - 1), w = Math.min(24, Math.max(4, slot - 2)), color = cssVar("--series-3");
  const actual = score ? score.actual : null;
  net.forEach((v, i) => {
    const describe = (e) => showTip(e, `${hourLabel(doc.hours[i].local_hour)} to ${hourLabel((doc.hours[i].local_hour + 1) % 24)}`, [
      { label: v > 0 ? "Sell" : v < 0 ? "Buy" : "Idle", value: `${fmt(Math.abs(v), 2)} MW`, color: v ? color : null },
      { label: "Forecast mean price", value: eur(doc.mean[i], 2) },
      ...(actual ? [{ label: "Real price", value: eur(actual[i], 2) }] : []),
    ]);
    if (Math.abs(v) > 1e-6) {
      const top = y(Math.max(v, 0)), bottom = y(Math.min(v, 0)), h = bottom - top, r = Math.min(4, h, w / 2), xl = x(i) - w / 2;
      const d = v > 0   // rounded data-end, square at the baseline
        ? `M${xl} ${bottom}V${top + r}Q${xl} ${top} ${xl + r} ${top}H${xl + w - r}Q${xl + w} ${top} ${xl + w} ${top + r}V${bottom}Z`
        : `M${xl} ${top}V${bottom - r}Q${xl} ${bottom} ${xl + r} ${bottom}H${xl + w - r}Q${xl + w} ${bottom} ${xl + w} ${bottom - r}V${top}Z`;
      f.svg.appendChild(el("path", { class: "bar", d, fill: color, "data-i": i }));
    }
    const hit = el("rect", { class: "hit", x: x(i) - Math.max(slot, 24) / 2, y: f.y1, width: Math.max(slot, 24), height: f.y0 - f.y1, tabindex: 0, "aria-label": `${hourLabel(doc.hours[i].local_hour)}: ${v > 0 ? "sell" : v < 0 ? "buy" : "idle"} ${fmt(Math.abs(v), 2)} megawatts` });
    const bar = () => f.svg.querySelector(`.bar[data-i="${i}"]`);
    hit.addEventListener("pointermove", (e) => { describe(e); bar()?.classList.add("active"); });
    hit.addEventListener("pointerleave", () => { hideTip(); bar()?.classList.remove("active"); });
    hit.addEventListener("focus", () => { const b = hit.getBoundingClientRect(); describe({ clientX: b.left + b.width / 2, clientY: b.top }); });
    hit.addEventListener("blur", hideTip);
    f.svg.appendChild(hit);
  });
  let sub = `${doc.battery.power_mw} MW / ${doc.battery.capacity_mwh} MWh battery. Planned on the forecast: expected profit ${eur(doc.expected_profit_eur.mean)}.`;
  if (score) sub += ` Made ${eur(score.profit_eur.mean)} at the real prices; perfect foresight would have made ${eur(score.profit_eur.perfect)}.`;
  $("plan-sub").textContent = sub;
  table($("plan-table"), ["Hour", "Buy, MW", "Sell, MW", "Forecast mean price", "Real price"],
    doc.hours.map((h, i) => [hourLabel(h.local_hour), fmt(s.charge_mw[i], 2), fmt(s.discharge_mw[i], 2), fmt(doc.mean[i], 2), actual ? fmt(actual[i], 2) : "not yet published"]));
}

/* ---------- cumulative money ---------- */
const POLICIES = [
  { key: "mean", label: "GridWise plan", color: "--series-1" },
  { key: "profile", label: "No-model baseline", color: "--context-1" },
  { key: "perfect", label: "Perfect foresight", color: "--context-2" },
];
function drawMoney(board) {
  const host = $("money"), days = board && board.daily ? board.daily : [];
  if (!days.length) {
    $("money-legend").replaceChildren();
    $("money-table").replaceChildren();
    return emptyState(host, "No day has been scored yet. A day is scored once its real prices are published, the afternoon before delivery.");
  }
  const series = POLICIES.map((p) => { let c = 0; return { ...p, colorValue: cssVar(p.color), values: days.map((d) => (c += d.profit_eur[p.key])) }; });
  const all = series.flatMap((s) => s.values).concat([0]);
  const ticks = niceTicks(Math.min(...all), Math.max(...all)), n = days.length;
  const narrow = isNarrow(host);
  const f = frame(host, narrow ? 240 : 280, { t: 10, r: narrow ? 12 : 124, b: 28, l: 56 });
  const x = n === 1 ? () => (f.x0 + f.x1) / 2 : linear(0, n - 1, f.x0 + 6, f.x1 - 6);
  const y = linear(ticks[0], ticks[ticks.length - 1], f.y0, f.y1);
  f.svg.setAttribute("aria-label", "Cumulative profit by day for the GridWise plan, the no-model baseline and perfect foresight");
  yAxis(f, ticks, y, (t) => fmt(t));
  const step = Math.max(1, Math.ceil(n / (narrow ? 3 : 6)));
  days.forEach((d, i) => { if (i % step === 0) f.svg.appendChild(el("text", { x: x(i), y: f.y0 + 18, "text-anchor": "middle" }, d.day.slice(5))); });
  const surface = cssVar("--surface-1");
  for (const s of [...series].reverse()) {
    f.svg.appendChild(el("path", { d: path(s.values.map((v, i) => [x(i), y(v)])), fill: "none", stroke: s.colorValue, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    f.svg.appendChild(el("circle", { cx: x(n - 1), cy: y(s.values[n - 1]), r: 4, fill: s.colorValue, stroke: surface, "stroke-width": 2 }));
  }
  if (!narrow) endLabels(f, series.map((s) => ({ y: y(s.values[n - 1]), label: s.label })));
  crosshair(f, n, x, (i) => ({ head: dayLabel(days[i].day), rows: series.map((s) => ({ label: s.label, value: eur(s.values[i]), color: s.colorValue })) }));
  legend($("money-legend"), series.map((s) => ({ kind: "line", color: s.colorValue, label: s.label })));
  table($("money-table"), ["Day", ...series.map((s) => s.label + ", cumulative")], days.map((d, i) => [d.day, ...series.map((s) => eur(s.values[i]))]));
}

/* ---------- tiles ---------- */
function tile(label, value, context, good = false) {
  const t = html("div", "tile");
  t.append(html("div", "label", label), html("div", "value", value), html("div", "context" + (good ? " good" : ""), context));
  return t;
}
function drawTiles(board, back) {
  const host = $("tiles");
  host.replaceChildren();
  if (!board || !board.days_scored) {
    $("live-note").textContent = "The live record is just starting: no day has been scored yet. The backtest further down shows how the same system did on past data.";
    host.append(tile("Days scored", "0", "first score after the first forecast day"));
    return;
  }
  const b = board, share = b.share_of_perfect, per = b.profit_eur_per_day, gain = per.mean - per.profile;
  $("live-note").textContent = `${plural(b.days_scored, "delivery day")} scored, ${dayLabel(b.first_day)} to ${dayLabel(b.last_day)}. Small samples swing a lot; the backtest below covers far more days.`;
  host.append(
    tile("Forecast error", eur(b.mae, 1), "mean absolute error per MWh" + (back ? `; backtest ${eur(back.models.ensemble.mae, 1)}` : "")),
    tile("Real price inside the 90% range", pct(b.coverage_90), "target 90%"),
    tile("Profit captured", pct(share.mean), `of perfect foresight; no-model baseline ${pct(share.profile)}`),
    tile("Gain over the baseline", (gain >= 0 ? "+" : "") + eur(gain), "per day, for a 1 MW / 2 MWh battery", gain > 0),
  );
}

/* ---------- system checks: status is always icon + words, never colour alone ---------- */
function drawChecks(health) {
  const host = $("checks");
  host.replaceChildren();
  $("checks-title").hidden = !(health && health.checks && health.checks.length);
  if (!health || !health.checks) return;
  for (const c of health.checks) {
    const li = html("li"), warn = c.status !== "ok";
    const icon = html("span", "icon" + (warn ? " warning" : ""), warn ? "!" : "✓");
    icon.setAttribute("aria-hidden", "true");
    const body = html("span");
    body.append(html("span", "name", c.name + (warn ? " (needs a look): " : " (OK): ")), html("span", "detail", c.detail));
    li.append(icon, body);
    host.appendChild(li);
  }
}

/* ---------- backtest ---------- */
const MODEL_LABEL = { ensemble: "GridWise ensemble", qnn: "Quantile neural network", qgbm: "Quantile gradient boosting", lear: "LASSO baseline (LEAR style)", naive: "Naive (repeat a similar day)" };
const POLICY_LABEL = { perfect: "Perfect foresight (bound)", ensemble_mean: "GridWise plan", ensemble_cvar: "GridWise risk-averse plan", ensemble_median: "Plan on the median forecast", lear: "Plan on the LASSO baseline", naive: "Plan on the naive forecast", profile: "No-model baseline" };
function drawBacktest(back) {
  if (!back) { $("back-note").textContent = "Backtest results are not published yet."; return; }
  $("back-note").textContent = `Not live: every day from ${back.test_start} to ${back.test_end} (${fmt(back.days)} days) was forecast by models that had only seen earlier days, using only inputs available before the auction. Lower error is better; coverage should be close to its target.`;
  const order = ["ensemble", "qnn", "qgbm", "lear", "naive"].filter((m) => back.models[m]);
  table($("back-models"), ["Model", "Error, EUR/MWh", "vs naive", "CRPS", "80% range holds", "90% range holds"],
    order.map((m) => { const s = back.models[m]; return [MODEL_LABEL[m] || m, fmt(s.mae, 2), m === "naive" ? "–" : pct(s.rmae - 1), fmt(s.crps, 2), pct(s.coverage_80, 1), pct(s.coverage_90, 1)]; }), 0);
  const pol = Object.entries(back.policies).filter(([k]) => k !== "perfect").sort((a, b) => b[1].share_of_perfect - a[1].share_of_perfect);
  const host = $("back-bars"), narrow = isNarrow(host), rowH = narrow ? 46 : 34;
  const f = frame(host, pol.length * rowH + 30, { t: 6, r: 64, b: 24, l: narrow ? 4 : 190 });
  const x = linear(0, 1, f.x0, f.x1), color = cssVar("--series-1");
  f.svg.setAttribute("aria-label", "Share of perfect-foresight profit captured by each plan in the backtest");
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    f.svg.appendChild(el("line", { class: t === 0 ? "baseline" : "gridline", x1: x(t), x2: x(t), y1: f.y1, y2: f.y0 }));
    f.svg.appendChild(el("text", { x: x(t), y: f.y0 + 16, "text-anchor": "middle" }, pct(t)));
  }
  pol.forEach(([k, p], i) => {
    const cy = f.y1 + i * rowH + (narrow ? rowH - 12 : rowH / 2), w = Math.max(0, x(p.share_of_perfect) - f.x0), h = 14, r = Math.min(4, w);
    f.svg.appendChild(narrow
      ? el("text", { class: "label", x: f.x0, y: cy - 13 }, POLICY_LABEL[k] || k)
      : el("text", { class: "label", x: f.x0 - 10, y: cy + 4, "text-anchor": "end" }, POLICY_LABEL[k] || k));
    const bar = el("path", { class: "bar", fill: color, d: `M${f.x0} ${cy - h / 2}H${f.x0 + w - r}Q${f.x0 + w} ${cy - h / 2} ${f.x0 + w} ${cy - h / 2 + r}V${cy + h / 2 - r}Q${f.x0 + w} ${cy + h / 2} ${f.x0 + w - r} ${cy + h / 2}H${f.x0}Z` });
    f.svg.appendChild(bar);
    f.svg.appendChild(el("text", { class: "value", x: f.x0 + w + 8, y: cy + 4 }, pct(p.share_of_perfect, 1)));
    const hit = el("rect", { class: "hit", x: f.x0, y: f.y1 + i * rowH, width: f.x1 - f.x0, height: rowH, tabindex: 0, "aria-label": `${POLICY_LABEL[k] || k}: ${pct(p.share_of_perfect, 1)} of perfect foresight, ${eur(p.eur_per_day)} per day` });
    const describe = (e) => showTip(e, POLICY_LABEL[k] || k, [
      { label: "Share of perfect foresight", value: pct(p.share_of_perfect, 1), color },
      { label: "Profit per day", value: eur(p.eur_per_day) },
      { label: "Days with a loss", value: pct(p.loss_day_share, 1) },
      { label: "Mean of the worst 5% of days", value: eur(p.worst_5pct_mean) },
    ]);
    hit.addEventListener("pointermove", (e) => { describe(e); bar.classList.add("active"); });
    hit.addEventListener("pointerleave", () => { hideTip(); bar.classList.remove("active"); });
    hit.addEventListener("focus", () => { const b = hit.getBoundingClientRect(); describe({ clientX: b.left + b.width / 2, clientY: b.top }); });
    hit.addEventListener("blur", hideTip);
    f.svg.appendChild(hit);
  });
  $("back-bars-sub").textContent = `${fmt(back.policy_days)} backtest days. Perfect foresight made ${eur(back.policies.perfect.eur_per_day)} per day with a ${back.battery.power_mw} MW / ${back.battery.capacity_mwh} MWh battery.`;
  if (back.leak_note) $("back-leak").textContent = back.leak_note;
}

/* ---------- wiring ---------- */
const state = { days: [], day: null, scores: {}, board: null, back: null, health: null, cache: {} };
async function showDay(day) {
  state.day = day;
  $("day").value = day;
  const i = state.days.indexOf(day);
  $("prev").disabled = i <= 0;
  $("next").disabled = i >= state.days.length - 1;
  const doc = state.cache[day] || (state.cache[day] = await getJSON(`live/forecasts/${day}.json`));
  if (!doc) { emptyState($("fan"), "This forecast file could not be loaded."); emptyState($("plan"), ""); return; }
  const score = state.scores[day] || null;
  $("day-note").textContent = doc.issued_after_prices
    ? "Issued after the prices were already public: shown for illustration, never scored."
    : score ? "Scored against the real prices." : "Waiting for the real prices.";
  drawFan(doc, doc.issued_after_prices ? null : score);
  drawPlan(doc, doc.issued_after_prices ? null : score);
}
function redraw() {
  drawTiles(state.board, state.back);
  drawMoney(state.board);
  drawBacktest(state.back);
  if (state.day) showDay(state.day);
}
async function init() {
  const [index, board, scores, back, health] = await Promise.all(["index", "scoreboard", "scores", "backtest", "monitoring"].map((n) => getJSON(`live/${n}.json`)));
  Object.assign(state, { days: index ? index.forecast_days : [], board, scores: scores || {}, back, health });
  $("status").textContent = board
    ? `${plural(board.days_scored, "day")} scored${board.first_day ? " since " + dayLabel(board.first_day) : ""} · ${plural(state.days.length, "forecast")} published · updated ${board.updated_utc.replace("T", " ").slice(0, 16)} UTC`
    : "The live record has not been published yet.";
  const sel = $("day");
  for (const d of [...state.days].reverse()) { const o = html("option", null, dayLabel(d)); o.value = d; sel.appendChild(o); }
  sel.addEventListener("change", () => showDay(sel.value));
  $("prev").addEventListener("click", () => showDay(state.days[state.days.indexOf(state.day) - 1]));
  $("next").addEventListener("click", () => showDay(state.days[state.days.indexOf(state.day) + 1]));
  $("latest").addEventListener("click", () => showDay(state.days[state.days.length - 1]));
  for (const b of document.querySelectorAll(".toggle")) {
    b.addEventListener("click", () => {
      const t = $(b.dataset.table), open = t.hidden;
      t.hidden = !open;
      b.setAttribute("aria-expanded", String(open));
      b.textContent = open ? "Hide table" : "Show as table";
    });
  }
  drawTiles(board, back);
  drawChecks(health);
  drawMoney(board);
  drawBacktest(back);
  if (state.days.length) await showDay(state.days[state.days.length - 1]);
  else { emptyState($("fan"), "No forecast has been published yet."); emptyState($("plan"), ""); }
  let timer;
  window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(redraw, 150); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
}
init();
