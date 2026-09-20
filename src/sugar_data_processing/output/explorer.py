"""Self-contained HTML explorer that also carries the markdown report results."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.config import (
    COHORT_LABELS,
    EVALUATION_MODE_LABELS,
    MAX_PLAUSIBLE_CGM_YEARS,
    MAX_PLAUSIBLE_DIABETES_YEARS,
)
from sugar_data_processing.gathering.participants import opposite_trait_summary
from sugar_data_processing.output.narration import (
    explain_edition_and_ai_path,
    explain_hypothesis_result,
    explain_opposite_trait,
    how_to_read_report,
)
from sugar_data_processing.statistics.catalog import HYPOTHESES


def write_explorer_html(
    *,
    participants: pl.DataFrame,
    runs: pl.DataFrame,
    output_path: Path,
    source_csv: Path,
    suite: Any | None = None,
    benchmarks: Any | None = None,
    verification: Any | None = None,
    edition: str = "human",
    comparison: dict[str, Any] | None = None,
    sequences_dir: Path | None = None,
) -> Path:
    """Write a filterable dashboard that also embeds the analysis-report results."""
    with start_action(action_type="output.write_explorer_html") as action:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _explorer_payload(
            participants,
            runs,
            source_csv,
            suite=suite,
            benchmarks=benchmarks,
            verification=verification,
            edition=edition,
            comparison=comparison,
            sequences_dir=sequences_dir,
        )
        html_text = _HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, default=_json_default))
        output_path.write_text(html_text, encoding="utf-8")
        action.log(message_type="info", path=str(output_path), n_players=payload["n_players"])
        return output_path


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _formats_label(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return ",".join(str(item) for item in raw)
    return str(raw)


def _as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return result
    return None


def markdown_fragment_to_html(md: str) -> str:
    """Small markdown subset → HTML for embedding report prose in the explorer."""
    escaped = html.escape(md)
    lines = escaped.splitlines()
    out: list[str] = []
    in_list = False
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            if in_list:
                out.append("</ul>")
                in_list = False
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= {"-", ":"} and c for c in cells):
                continue
            if not in_table:
                out.append("<table>")
                out.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</tbody></table>")
            in_table = False
        if stripped.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(stripped[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        heading = re.match(r"^(#{2,4})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            continue
        if not stripped:
            continue
        out.append(f"<p>{_inline(stripped)}</p>")
    if in_list:
        out.append("</ul>")
    if in_table:
        out.append("</tbody></table>")
    return "\n".join(out)


def _inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def _explorer_payload(
    participants: pl.DataFrame,
    runs: pl.DataFrame,
    source_csv: Path,
    *,
    suite: Any | None,
    benchmarks: Any | None,
    verification: Any | None,
    edition: str,
    comparison: dict[str, Any] | None,
    sequences_dir: Path | None,
) -> dict[str, Any]:
    opposite = opposite_trait_summary(participants)
    suite_dict = _as_dict(suite) or {}
    bench_dict = _as_dict(benchmarks) or {}
    ver_dict = _as_dict(verification) or {}

    hyp_blocks: dict[str, str] = {}
    for key in ("h1", "h2", "h3", "h4", "h5", "h6"):
        hyp_blocks[key] = markdown_fragment_to_html(
            explain_hypothesis_result(key, suite_dict.get(key))
        )

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
                "player_trait": row.get("player_trait"),
                "age": row.get("age"),
                "diabetes_duration": row.get("diabetes_duration"),
                "diabetes_duration_months": row.get("diabetes_duration_months"),
                "cgm_duration_years": row.get("cgm_duration_years"),
                "cgm_duration_months": row.get("cgm_duration_months"),
                "n_runs": int(row.get("n_runs") or 0),
                "n_formats": int(row.get("n_formats_played") or 0),
                "formats": _formats_label(row.get("formats_played")),
                "repeat": bool(row.get("is_repeat_player")),
                "all_formats": bool(row.get("played_all_formats")),
                "challenge_unknown": bool(row.get("played_challenge_unknown")),
                "opposite_trait": bool(row.get("played_opposite_trait")),
                "mae_primary": row.get("mae_primary"),
                "mae_generic": row.get("mae_generic"),
                "mae_own": row.get("mae_own"),
                "mae_same": row.get("mae_same_trait"),
                "mae_opposite": row.get("mae_opposite_trait"),
                "mae_a": row.get("mae_format_a"),
                "mae_b": row.get("mae_format_b"),
                "mae_c": row.get("mae_format_c"),
                "n_generic": int(row.get("n_rounds_generic") or 0),
                "n_own": int(row.get("n_rounds_own") or 0),
            }
        )
    format_counts = (
        runs.group_by("format").len().sort("format").to_dicts() if runs.height else []
    )
    title = (
        "Sugar Sugar AI study explorer"
        if edition == "ai"
        else "Sugar Sugar human study explorer"
    )
    return {
        "edition": edition,
        "title": title,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_csv": str(source_csv),
        "n_players": participants.height,
        "n_runs": runs.height,
        "format_counts": format_counts,
        "players": players,
        "cohort_labels": COHORT_LABELS,
        "reading_guide_html": markdown_fragment_to_html(how_to_read_report()),
        "edition_html": markdown_fragment_to_html(explain_edition_and_ai_path()),
        "opposite_html": markdown_fragment_to_html(explain_opposite_trait(opposite)),
        "opposite": opposite,
        "hypotheses_html": hyp_blocks,
        "hypothesis_catalog": HYPOTHESES,
        "benchmarks": bench_dict,
        "verification": ver_dict,
        "comparison": comparison,
        "evaluation_mode_labels": EVALUATION_MODE_LABELS,
        "sequences_dir": str(sequences_dir) if sequences_dir is not None else "",
        "duration_caps": {
            "diabetes_duration_months": MAX_PLAUSIBLE_DIABETES_YEARS * 12.0,
            "cgm_duration_years": MAX_PLAUSIBLE_CGM_YEARS,
            "cgm_duration_months": MAX_PLAUSIBLE_CGM_YEARS * 12.0,
        },
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
    header p { margin: 0; color: #cbd5e1; max-width: 80ch; }
    .badge {
      display: inline-block;
      margin-bottom: 10px;
      padding: 3px 10px;
      border-radius: 999px;
      background: #38bdf8;
      color: #0f172a;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .badge.ai { background: #f59e0b; }
    main { padding: 20px 24px 48px; }
    section { margin: 28px 0; }
    section > h2 { margin: 0 0 10px; font-size: 1.25rem; }
    .prose { max-width: 90ch; color: var(--muted); }
    .prose table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 8px 0 16px; }
    .prose th, .prose td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
    .prose code { background: #e2e8f0; padding: 1px 5px; border-radius: 4px; }
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
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 20px; }
    .pair {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) minmax(380px, 1.25fr);
      gap: 24px;
      align-items: start;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px 18px 12px;
      margin: 16px 0;
    }
    @media (max-width: 900px) {
      .pair { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px 8px;
    }
    .card h2, .card h3 { margin: 0 0 8px; font-size: 1.05rem; }
    .pair h3 { margin: 0 0 8px; font-size: 1.05rem; }
    canvas { width: 100% !important; height: 420px !important; max-height: none; }
    .pie-box {
      width: min(360px, 100%);
      aspect-ratio: 1 / 1;
      margin: 0 auto;
    }
    .pie-box canvas {
      width: 100% !important;
      height: 100% !important;
    }
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
    .hyp { margin-bottom: 18px; }
  </style>
</head>
<body>
  <header>
    <div class="badge" id="editionBadge">Human edition</div>
    <h1 id="pageTitle">Sugar Sugar human study explorer</h1>
    <p>The same results as the markdown analysis report, with every person drawn as a point. Filter the cohort, inspect Challenge the unknown / opposite-trait play, and (in the AI edition) compare ingested models. Lower MAE is better.</p>
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
          <option value="challenge">Challenge the unknown</option>
          <option value="opposite">Played opposite trait</option>
        </select>
      </label>
      <label>Search id
        <input id="search" type="search" placeholder="study_id prefix" />
      </label>
    </div>

    <section>
      <h2>How to read this</h2>
      <div class="prose" id="reading"></div>
      <div class="prose" id="editionNote"></div>
    </section>

    <section>
      <h2>1. Cohort</h2>
      <div class="pair">
        <div>
          <h3>Who is in the cohort</h3>
          <p class="prose">Each slice is a unique person in one diabetes × CGM bucket. This is head-count, not accuracy.</p>
        </div>
        <div class="pie-box"><canvas id="pie"></canvas></div>
      </div>
      <div class="pair">
        <div>
          <h3>How accurate people were</h3>
          <p class="prose">Each tick is one person's MAE (mg/dL). Lower is better. Colour is the diabetes × CGM bucket.</p>
        </div>
        <div><canvas id="swarm"></canvas></div>
      </div>
      <div class="pair">
        <div>
          <h3>Accuracy on each task</h3>
          <p class="prose">A is generic, B is own data, C is mixed. One point per person who played that format. Lower MAE is better.</p>
        </div>
        <div><canvas id="formats"></canvas></div>
      </div>
    </section>

    <section>
      <h2>3. Primary hypotheses</h2>
      <div class="pair">
        <div class="prose" id="h1Text"></div>
        <div><canvas id="h1"></canvas></div>
      </div>
      <div class="pair">
        <div class="prose" id="h2Text"></div>
        <div><canvas id="h2"></canvas></div>
      </div>
    </section>

    <section>
      <h2>4. Secondary hypotheses</h2>
      <div class="pair">
        <div class="prose" id="h3Text"></div>
        <div><canvas id="h3"></canvas></div>
      </div>
      <div class="pair">
        <div class="prose" id="h4Text"></div>
        <div><canvas id="h4"></canvas></div>
      </div>
      <div class="pair">
        <div class="prose" id="h5Text"></div>
        <div><canvas id="clusters"></canvas></div>
      </div>
      <div class="prose" id="h6Text"></div>
    </section>

    <section>
      <h2>Challenge the unknown / opposite trait</h2>
      <div class="pair">
        <div class="prose" id="oppositeText"></div>
        <div><canvas id="opposite"></canvas></div>
      </div>
    </section>

    <section>
      <h2>Literature / GlucoBench context</h2>
      <div class="prose" id="bench"></div>
    </section>

    <section>
      <h2>Data verification</h2>
      <div class="prose" id="verify"></div>
    </section>

    <section id="aiSection">
      <h2>AI comparison</h2>
      <div class="prose" id="aiText"></div>
    </section>

    <section>
      <h2>People in the current filter</h2>
      <div class="card wide">
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th data-key="study_id">id</th>
                <th data-key="cohort_label">category</th>
                <th data-key="player_trait">trait</th>
                <th data-key="n_runs">runs</th>
                <th data-key="formats">formats</th>
                <th data-key="challenge_unknown">challenge</th>
                <th data-key="mae_primary">MAE</th>
                <th data-key="mae_generic">generic</th>
                <th data-key="mae_own">own</th>
                <th data-key="mae_same">same</th>
                <th data-key="mae_opposite">opposite</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
    </section>

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
    document.getElementById("pageTitle").textContent = DATA.title;
    const badge = document.getElementById("editionBadge");
    badge.textContent = DATA.edition === "ai" ? "AI edition" : "Human edition";
    badge.classList.toggle("ai", DATA.edition === "ai");
    document.getElementById("reading").innerHTML = DATA.reading_guide_html;
    document.getElementById("editionNote").innerHTML = DATA.edition_html;
    document.getElementById("oppositeText").innerHTML = DATA.opposite_html;

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
    function jitter(id, spread) {
      let h = 0;
      for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
      return ((h % 1000) / 1000 - 0.5) * spread;
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
        if (repeat === "challenge" && !p.challenge_unknown) return false;
        if (repeat === "opposite" && !p.opposite_trait) return false;
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

    function renderHypotheses() {
      Object.entries(DATA.hypotheses_html || {}).forEach(([key, htmlBlock]) => {
        const el = document.getElementById(key + "Text");
        if (el) el.innerHTML = htmlBlock;
      });
    }

    function renderBench() {
      const b = DATA.benchmarks || {};
      if (!b.human_mean_mae && b.human_mean_mae !== 0) {
        document.getElementById("bench").innerHTML = "<p>Benchmark numbers were not passed into this explorer.</p>";
        return;
      }
      document.getElementById("bench").innerHTML = `
        <p>${b.narrative || ""}</p>
        <table>
          <thead><tr><th>Band</th><th>% of humans inside</th></tr></thead>
          <tbody>
            <tr><td>Simple / ARIMA ${b.simple_baseline_mae_range ? b.simple_baseline_mae_range[0] : ""}–${b.simple_baseline_mae_range ? b.simple_baseline_mae_range[1] : ""}</td><td>${fmt(b.pct_inside_simple_baseline_band)}%</td></tr>
            <tr><td>Deep learning</td><td>${fmt(b.pct_inside_deep_learning_band)}%</td></tr>
            <tr><td>Personalized</td><td>${fmt(b.pct_inside_personalized_band)}%</td></tr>
            <tr><td>Below simple-band low</td><td>${fmt(b.pct_below_simple_baseline_low)}%</td></tr>
          </tbody>
        </table>`;
    }

    function renderVerify() {
      const v = DATA.verification || {};
      if (!Object.keys(v).length) {
        document.getElementById("verify").innerHTML = "<p>Verification payload was not passed into this explorer.</p>";
        return;
      }
      const issues = [...(v.schema_issues || []), ...(v.quality_flags || [])].slice(0, 40);
      const rows = issues.map((i) => `<tr><td>${i.study_id}</td><td>${i.category}</td><td>${i.severity}</td><td>${i.detail}</td></tr>`).join("");
      document.getElementById("verify").innerHTML = `
        <p>Schema: <strong>${v.schema_ok ? "PASSED" : "FAILED"}</strong> —
        ${v.n_schema_issues || 0} schema issues, ${v.n_quality_flags || 0} quality flags
        (${v.n_high || 0} high).</p>
        <table><thead><tr><th>id</th><th>category</th><th>severity</th><th>detail</th></tr></thead>
        <tbody>${rows || "<tr><td colspan=4>none</td></tr>"}</tbody></table>`;
    }

    function renderAi() {
      const el = document.getElementById("aiText");
      const c = DATA.comparison;
      if (DATA.edition !== "ai" && !c) {
        el.innerHTML = `<p>This is the human edition. Sequences for post-factum scoring land in <code>${DATA.sequences_dir || "data/processed/ai"}</code>. After models write predictions, run <code>sdp ingest-ai</code> to fill the AI edition. In-place scoring is not collected yet.</p>`;
        return;
      }
      if (!c || !c.n_points) {
        el.innerHTML = `<p>No ingested model scores yet. Export is ready at <code>${DATA.sequences_dir || "data/processed/ai"}</code>. Evaluation mode available now: <code>post_factum</code>. <code>in_place</code> will appear when live-game model scores exist.</p>`;
        return;
      }
      const modelRows = Object.entries(c.model_mean_mae || {}).map(([name, mae]) => `<tr><td>${name}</td><td>${fmt(mae)}</td></tr>`).join("");
      const modeRows = Object.entries(c.by_mode || {}).map(([mode, info]) => {
        const models = Object.entries(info.model_mean_mae || {}).map(([k, v]) => `${k}=${fmt(v)}`).join(", ");
        return `<tr><td>${mode}</td><td>${info.n_points}</td><td>${info.n_people}</td><td>${fmt(info.human_mean_mae)}</td><td>${models || "—"}</td></tr>`;
      }).join("");
      el.innerHTML = `
        <p>Scored <strong>${c.n_points}</strong> points across <strong>${c.n_people}</strong> people.
        Human point MAE on that subset: <strong>${fmt(c.human_mean_mae)}</strong> mg/dL.</p>
        <table><thead><tr><th>Model</th><th>MAE</th></tr></thead><tbody>${modelRows}</tbody></table>
        <table><thead><tr><th>Mode</th><th>Points</th><th>People</th><th>Human MAE</th><th>Models</th></tr></thead>
        <tbody>${modeRows}</tbody></table>`;
    }

    function render() {
      const rows = filtered();
      const stats = document.getElementById("stats");
      const allFormats = rows.filter((p) => p.all_formats && num(p.mae_generic) !== null && num(p.mae_own) !== null);
      const betterOwn = allFormats.filter((p) => p.mae_own < p.mae_generic).length;
      stats.innerHTML = `
        <div class="stat"><b>${rows.length}</b><span>people in filter</span></div>
        <div class="stat"><b>${rows.filter((p) => p.repeat).length}</b><span>repeat players</span></div>
        <div class="stat"><b>${rows.filter((p) => p.challenge_unknown).length}</b><span>challenge the unknown</span></div>
        <div class="stat"><b>${rows.filter((p) => p.opposite_trait).length}</b><span>played opposite trait</span></div>
        <div class="stat"><b>${fmt(mean(rows.map((p) => num(p.mae_primary))))}</b><span>mean person MAE</span></div>
        <div class="stat"><b>${betterOwn}/${allFormats.length || 0}</b><span>all-variant better on own</span></div>
      `;

      const cohortKeys = Object.keys(DATA.cohort_labels);
      upsertChart("pie", "pie", {
        labels: cohortKeys.map((k) => DATA.cohort_labels[k]),
        datasets: [{ data: cohortKeys.map((k) => rows.filter((p) => p.cohort === k).length), backgroundColor: cohortKeys.map((k) => COLORS[k] || "#94a3b8") }],
      }, { maintainAspectRatio: true, aspectRatio: 1, plugins: { legend: { position: "bottom" } } });

      upsertChart("clusters", "scatter", {
        datasets: cohortKeys.map((k) => ({
          label: DATA.cohort_labels[k],
          data: rows.filter((p) => p.cohort === k && num(p.mae_generic) !== null).map((p) => ({
            x: num(p.mae_generic),
            y: num(p.mae_own) === null ? num(p.mae_primary) : num(p.mae_own),
          })),
          backgroundColor: COLORS[k] || "#94a3b8",
          pointRadius: rows.filter((p) => p.cohort === k).map((p) => p.challenge_unknown ? 7 : 5),
        })),
      }, {
        scales: {
          x: { title: { display: true, text: "Generic MAE" } },
          y: { title: { display: true, text: "Own MAE (or person MAE)" } },
        },
      });

      upsertChart("swarm", "scatter", {
        datasets: cohortKeys.map((k, idx) => ({
          label: DATA.cohort_labels[k],
          data: rows.filter((p) => p.cohort === k && num(p.mae_primary) !== null).map((p) => ({
            x: idx + jitter(p.study_id, 0.35),
            y: num(p.mae_primary),
          })),
          backgroundColor: COLORS[k] || "#94a3b8",
        })),
      }, {
        scales: {
          x: {
            min: -0.6,
            max: cohortKeys.length - 0.4,
            ticks: {
              stepSize: 1,
              callback: (v) => Number.isInteger(v) ? (DATA.cohort_labels[cohortKeys[v]] || "") : "",
            },
          },
          y: { title: { display: true, text: "Person MAE (mg/dL)" } },
        },
      });

      upsertChart("h1", "scatter", {
        datasets: layeredCategorySwarm(rows, (p) => p.diabetic === true, (p) => p.diabetic === false),
      }, layeredCategoryScales("PwD", "non-PwD"));

      upsertChart("h2", "scatter", {
        datasets: layeredCategorySwarm(rows, (p) => p.uses_cgm === true, (p) => p.uses_cgm === false),
      }, layeredCategoryScales("CGM user", "no CGM"));

      upsertChart("h3", "scatter", {
        datasets: durationSourceSeries(rows, (p) => p.diabetic === true, (p) => {
          const months = num(p.diabetes_duration_months);
          const years = num(p.diabetes_duration);
          const value = months !== null ? months : (years === null ? null : years * 12);
          const cap = (DATA.duration_caps || {}).diabetes_duration_months;
          return value === null || (cap && value > cap) ? null : value;
        }),
      }, {
        scales: {
          x: { title: { display: true, text: "Months with diabetes" } },
          y: { title: { display: true, text: "Person MAE (mg/dL)" } },
        },
      });

      upsertChart("h4", "scatter", {
        datasets: durationSourceSeries(rows, (p) => p.uses_cgm === true, (p) => {
          const months = num(p.cgm_duration_months);
          const years = num(p.cgm_duration_years);
          const value = months !== null ? months : (years === null ? null : years * 12);
          const cap = (DATA.duration_caps || {}).cgm_duration_months;
          return value === null || (cap && value > cap) ? null : value;
        }),
      }, {
        scales: {
          x: { title: { display: true, text: "Months using CGM" } },
          y: { title: { display: true, text: "Person MAE (mg/dL)" } },
        },
      });

      const formatCols = [
        { key: "mae_a", label: "A generic", color: "#4C78A8" },
        { key: "mae_b", label: "B own", color: "#54A24B" },
        { key: "mae_c", label: "C mixed", color: "#F58518" },
      ];
      upsertChart("formats", "scatter", {
        datasets: formatCols.map((f, idx) => ({
          label: f.label,
          data: rows.filter((p) => num(p[f.key]) !== null).map((p) => ({
            x: idx + jitter(p.study_id + f.key, 0.28),
            y: num(p[f.key]),
          })),
          backgroundColor: f.color,
        })),
      }, {
        scales: {
          x: {
            min: -0.5,
            max: 2.5,
            ticks: {
              stepSize: 1,
              callback: (v) => Number.isInteger(v) ? ((formatCols[v] || {}).label || "") : "",
            },
          },
          y: { title: { display: true, text: "mg/dL (lower is better)" } },
        },
      });

      upsertChart("opposite", "scatter", {
        datasets: [{
          label: "Has both sides",
          data: rows.filter((p) => num(p.mae_same) !== null && num(p.mae_opposite) !== null)
            .map((p) => ({ x: num(p.mae_same), y: num(p.mae_opposite) })),
          backgroundColor: "#E45756",
        }],
      }, {
        scales: {
          x: { title: { display: true, text: "Same-trait MAE" } },
          y: { title: { display: true, text: "Opposite-trait MAE" } },
        },
      });

      const body = document.getElementById("rows");
      body.innerHTML = rows.map((p) => `
        <tr>
          <td>${p.study_id}</td>
          <td>${p.cohort_label}</td>
          <td>${p.player_trait || "—"}</td>
          <td>${p.n_runs}</td>
          <td>${p.formats}</td>
          <td>${p.challenge_unknown ? "yes" : ""}</td>
          <td>${fmt(p.mae_primary)}</td>
          <td>${fmt(p.mae_generic)}</td>
          <td>${fmt(p.mae_own)}</td>
          <td>${fmt(p.mae_same)}</td>
          <td>${fmt(p.mae_opposite)}</td>
        </tr>
      `).join("");
    }

    function layeredCategorySwarm(rows, yesFn, noFn) {
      return [
        {
          label: "generic",
          data: [
            ...rows.filter((p) => yesFn(p) && num(p.mae_generic) !== null).map((p) => ({
              x: 0 + jitter(p.study_id + "generic", 0.22),
              y: num(p.mae_generic),
            })),
            ...rows.filter((p) => noFn(p) && num(p.mae_generic) !== null).map((p) => ({
              x: 1 + jitter(p.study_id + "generic", 0.22),
              y: num(p.mae_generic),
            })),
          ],
          backgroundColor: "#4C78A8",
        },
        {
          label: "own",
          data: [
            ...rows.filter((p) => yesFn(p) && num(p.mae_own) !== null).map((p) => ({
              x: 0 + jitter(p.study_id + "own", 0.22),
              y: num(p.mae_own),
            })),
            ...rows.filter((p) => noFn(p) && num(p.mae_own) !== null).map((p) => ({
              x: 1 + jitter(p.study_id + "own", 0.22),
              y: num(p.mae_own),
            })),
          ],
          backgroundColor: "#54A24B",
        },
      ];
    }
    function layeredCategoryScales(yesLabel, noLabel) {
      return {
        plugins: { legend: { position: "top" } },
        scales: {
          x: {
            min: -0.5,
            max: 1.5,
            title: { display: true, text: "Category" },
            afterBuildTicks(scale) {
              scale.ticks = [{ value: 0 }, { value: 1 }];
            },
            ticks: {
              callback: (v) => (Number(v) === 0 ? yesLabel : Number(v) === 1 ? noLabel : ""),
            },
          },
          y: { title: { display: true, text: "Person MAE (mg/dL)" } },
        },
      };
    }
    function durationSourceSeries(rows, matchFn, xFn) {
      return [
        {
          label: "generic",
          data: rows.filter((p) => matchFn(p) && xFn(p) !== null && num(p.mae_generic) !== null)
            .map((p) => ({ x: xFn(p), y: num(p.mae_generic) })),
          backgroundColor: "#4C78A8",
        },
        {
          label: "own",
          data: rows.filter((p) => matchFn(p) && xFn(p) !== null && num(p.mae_own) !== null)
            .map((p) => ({ x: xFn(p), y: num(p.mae_own) })),
          backgroundColor: "#54A24B",
        },
      ];
    }
    function upsertChart(id, type, data, options) {
      const canvas = document.getElementById(id);
      if (!canvas) return;
      if (charts[id]) {
        charts[id].data = data;
        charts[id].update();
        return;
      }
      charts[id] = new Chart(canvas, { type, data, options });
    }

    document.getElementById("cohortFilter").addEventListener("change", render);
    document.getElementById("repeatFilter").addEventListener("change", render);
    document.getElementById("search").addEventListener("input", render);
    document.getElementById("meta").textContent =
      `Generated ${DATA.generated_at} | source ${DATA.source_csv} | ${DATA.n_runs} runs / ${DATA.n_players} people | edition ${DATA.edition}`;
    renderHypotheses();
    renderBench();
    renderVerify();
    renderAi();
    render();
  </script>
</body>
</html>
"""
