"""Self-contained HTML explorer for the gathered participant table."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.config import COHORT_LABELS


def write_explorer_html(
    *,
    participants: pl.DataFrame,
    runs: pl.DataFrame,
    output_path: Path,
    source_csv: Path,
) -> Path:
    """Write a filterable Chart.js dashboard next to the markdown report."""
    with start_action(action_type="output.write_explorer_html") as action:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _explorer_payload(participants, runs, source_csv)
        html = _HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, default=_json_default))
        output_path.write_text(html, encoding="utf-8")
        action.log(message_type="info", path=str(output_path), n_players=payload["n_players"])
        return output_path


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def _formats_label(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return ",".join(str(item) for item in raw)
    return str(raw)


def _explorer_payload(participants: pl.DataFrame, runs: pl.DataFrame, source_csv: Path) -> dict[str, Any]:
    players: list[dict[str, Any]] = []
    for row in participants.iter_rows(named=True):
        category = str(row.get("cohort_category") or "unknown")
        players.append(
            {
                "study_id": str(row["study_id"])[:12],
                "cohort": category,
                "cohort_label": COHORT_LABELS.get(category, category),
                "diabetic": row.get("diabetic"),
                "uses_cgm": row.get("uses_cgm"),
                "age": row.get("age"),
                "n_runs": int(row.get("n_runs") or 0),
                "n_formats": int(row.get("n_formats_played") or 0),
                "formats": _formats_label(row.get("formats_played")),
                "repeat": bool(row.get("is_repeat_player")),
                "all_formats": bool(row.get("played_all_formats")),
                "mae_primary": row.get("mae_primary"),
                "mae_generic": row.get("mae_generic"),
                "mae_own": row.get("mae_own"),
                "mae_a": row.get("mae_format_a"),
                "mae_b": row.get("mae_format_b"),
                "mae_c": row.get("mae_format_c"),
                "n_generic": int(row.get("n_rounds_generic") or 0),
                "n_own": int(row.get("n_rounds_own") or 0),
            }
        )
    format_counts = (
        runs.group_by("format").len().sort("format").to_dicts()
        if runs.height
        else []
    )
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_csv": str(source_csv),
        "n_players": participants.height,
        "n_runs": runs.height,
        "format_counts": format_counts,
        "players": players,
        "cohort_labels": COHORT_LABELS,
    }


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sugar Sugar study explorer</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --ink: #0f172a;
      --muted: #475569;
      --paper: #f8fafc;
      --card: #ffffff;
      --line: #e2e8f0;
      --accent: #4C78A8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      color: var(--ink);
      background: var(--paper);
    }
    header {
      padding: 28px 32px 16px;
      background: linear-gradient(120deg, #0f172a, #1e3a5f);
      color: white;
    }
    header h1 { margin: 0 0 8px; font-size: 1.6rem; }
    header p { margin: 0; color: #cbd5e1; max-width: 72ch; }
    main { padding: 20px 24px 48px; }
    .stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0 8px; }
    .stat {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 16px;
      min-width: 140px;
    }
    .stat b { display: block; font-size: 1.35rem; }
    .stat span { color: var(--muted); font-size: 0.85rem; }
    .filters {
      display: flex; flex-wrap: wrap; gap: 10px 18px;
      align-items: center;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 16px;
      margin: 12px 0 20px;
    }
    .filters label { font-size: 0.92rem; color: var(--muted); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px 8px;
    }
    .card h2 { margin: 0 0 8px; font-size: 1.05rem; }
    canvas { max-height: 320px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
    th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
    th { position: sticky; top: 0; background: #f1f5f9; cursor: pointer; }
    .table-wrap { max-height: 420px; overflow: auto; margin-top: 8px; }
    .wide { grid-column: 1 / -1; }
    footer { color: var(--muted); font-size: 0.82rem; margin-top: 18px; }
    select, input[type="search"] {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 8px;
      font: inherit;
    }
  </style>
</head>
<body>
  <header>
    <h1>Sugar Sugar study explorer</h1>
    <p>Filter the cohort, compare tasks, and see whether people who played every variant did better on their own data or on generic traces. Lower MAE is better.</p>
  </header>
  <main>
    <div class="stats" id="stats"></div>
    <div class="filters">
      <label>Cohort
        <select id="cohortFilter"><option value="all">All categories</option></select>
      </label>
      <label>Players
        <select id="repeatFilter">
          <option value="all">Everyone</option>
          <option value="single">Single-run only</option>
          <option value="repeat">Repeat players</option>
          <option value="all_formats">Played A + B + C</option>
        </select>
      </label>
      <label>Search id
        <input id="search" type="search" placeholder="study_id prefix" />
      </label>
    </div>
    <div class="grid">
      <div class="card"><h2>How the categories split</h2><canvas id="pie"></canvas></div>
      <div class="card"><h2>Accuracy on each task</h2><canvas id="formats"></canvas></div>
      <div class="card"><h2>Individuals vs repeats</h2><canvas id="repeats"></canvas></div>
      <div class="card"><h2>All-variant players: own vs generic</h2><canvas id="paired"></canvas></div>
      <div class="card wide">
        <h2>People in the current filter</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th data-key="study_id">id</th>
                <th data-key="cohort_label">category</th>
                <th data-key="n_runs">runs</th>
                <th data-key="formats">formats</th>
                <th data-key="mae_primary">MAE</th>
                <th data-key="mae_generic">generic</th>
                <th data-key="mae_own">own</th>
                <th data-key="mae_a">A</th>
                <th data-key="mae_b">B</th>
                <th data-key="mae_c">C</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
    </div>
    <footer id="meta"></footer>
  </main>
  <script>
    const DATA = __PAYLOAD__;
    const COLORS = {
      diabetic_cgm: "#E45756",
      diabetic_non_cgm: "#F58518",
      nondiabetic_cgm: "#4C78A8",
      nondiabetic_non_cgm: "#72B7B2",
      unknown: "#94a3b8",
    };
    const cohortSelect = document.getElementById("cohortFilter");
    Object.entries(DATA.cohort_labels).forEach(([key, label]) => {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = label;
      cohortSelect.appendChild(opt);
    });

    function num(value) {
      return value === null || value === undefined || Number.isNaN(value) ? null : Number(value);
    }
    function fmt(value) {
      const n = num(value);
      return n === null ? "—" : n.toFixed(1);
    }
    function filtered() {
      const cohort = cohortSelect.value;
      const repeat = document.getElementById("repeatFilter").value;
      const q = document.getElementById("search").value.trim().toLowerCase();
      return DATA.players.filter((p) => {
        if (cohort !== "all" && p.cohort !== cohort) return false;
        if (repeat === "single" && p.repeat) return false;
        if (repeat === "repeat" && !p.repeat) return false;
        if (repeat === "all_formats" && !p.all_formats) return false;
        if (q && !p.study_id.toLowerCase().includes(q)) return false;
        return true;
      });
    }

    const charts = {};
    function mean(values) {
      const xs = values.filter((v) => v !== null);
      if (!xs.length) return 0;
      return xs.reduce((a, b) => a + b, 0) / xs.length;
    }

    function render() {
      const rows = filtered();
      const stats = document.getElementById("stats");
      const allFormats = rows.filter((p) => p.all_formats && num(p.mae_generic) !== null && num(p.mae_own) !== null);
      const betterOwn = allFormats.filter((p) => p.mae_own < p.mae_generic).length;
      stats.innerHTML = `
        <div class="stat"><b>${rows.length}</b><span>people in filter</span></div>
        <div class="stat"><b>${rows.filter((p) => p.repeat).length}</b><span>repeat players</span></div>
        <div class="stat"><b>${rows.filter((p) => p.all_formats).length}</b><span>played every variant</span></div>
        <div class="stat"><b>${fmt(mean(rows.map((p) => num(p.mae_primary))))}</b><span>mean person MAE</span></div>
        <div class="stat"><b>${betterOwn}/${allFormats.length || 0}</b><span>all-variant better on own</span></div>
      `;

      const cohortKeys = Object.keys(DATA.cohort_labels);
      const pieValues = cohortKeys.map((k) => rows.filter((p) => p.cohort === k).length);
      upsertChart("pie", "pie", {
        labels: cohortKeys.map((k) => DATA.cohort_labels[k]),
        datasets: [{ data: pieValues, backgroundColor: cohortKeys.map((k) => COLORS[k] || "#94a3b8") }],
      }, { plugins: { legend: { position: "bottom" } } });

      upsertChart("formats", "bar", {
        labels: ["Format A generic", "Format B own", "Format C mixed"],
        datasets: [{
          label: "Mean MAE",
          data: [
            mean(rows.map((p) => num(p.mae_a))),
            mean(rows.map((p) => num(p.mae_b))),
            mean(rows.map((p) => num(p.mae_c))),
          ],
          backgroundColor: ["#4C78A8", "#54A24B", "#F58518"],
        }],
      }, { scales: { y: { title: { display: true, text: "mg/dL (lower is better)" } } } });

      upsertChart("repeats", "bar", {
        labels: ["Unique people", "Single-run", "Repeat players", "Saved runs"],
        datasets: [{
          data: [
            rows.length,
            rows.filter((p) => !p.repeat).length,
            rows.filter((p) => p.repeat).length,
            rows.reduce((s, p) => s + (p.n_runs || 0), 0),
          ],
          backgroundColor: ["#4C78A8", "#72B7B2", "#F58518", "#9D755D"],
        }],
      }, { plugins: { legend: { display: false } } });

      upsertChart("paired", "scatter", {
        datasets: [{
          label: "Played A+B+C",
          data: allFormats.map((p) => ({ x: p.mae_generic, y: p.mae_own })),
          backgroundColor: "#54A24B",
        }],
      }, {
        scales: {
          x: { title: { display: true, text: "Generic MAE" } },
          y: { title: { display: true, text: "Own MAE" } },
        },
        plugins: {
          tooltip: {
            callbacks: { label: (ctx) => `generic ${ctx.parsed.x.toFixed(1)} / own ${ctx.parsed.y.toFixed(1)}` },
          },
        },
      });

      const body = document.getElementById("rows");
      body.innerHTML = rows.map((p) => `
        <tr>
          <td>${p.study_id}</td>
          <td>${p.cohort_label}</td>
          <td>${p.n_runs}</td>
          <td>${p.formats}</td>
          <td>${fmt(p.mae_primary)}</td>
          <td>${fmt(p.mae_generic)}</td>
          <td>${fmt(p.mae_own)}</td>
          <td>${fmt(p.mae_a)}</td>
          <td>${fmt(p.mae_b)}</td>
          <td>${fmt(p.mae_c)}</td>
        </tr>
      `).join("");
    }

    function upsertChart(id, type, data, options) {
      if (charts[id]) {
        charts[id].data = data;
        charts[id].update();
        return;
      }
      charts[id] = new Chart(document.getElementById(id), { type, data, options });
    }

    document.getElementById("cohortFilter").addEventListener("change", render);
    document.getElementById("repeatFilter").addEventListener("change", render);
    document.getElementById("search").addEventListener("input", render);
    document.getElementById("meta").textContent =
      `Generated ${DATA.generated_at} | source ${DATA.source_csv} | ${DATA.n_runs} runs / ${DATA.n_players} people`;
    render();
  </script>
</body>
</html>
"""
