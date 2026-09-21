"""Self-contained HTML explorer: Overview, Human, AI, and per-user traces."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from eliot import start_action

from sugar_data_processing.ai.traces import LINE_STYLES
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
    explain_milestone_page,
    explain_milestone_reading,
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
    edition: str = "merged",
    comparison: dict[str, Any] | None = None,
    sequences_dir: Path | None = None,
    ai_participants: pl.DataFrame | None = None,
    ai_suite: Any | None = None,
    traces: list[dict[str, Any]] | None = None,
    primary_model: str = "",
    h6: dict[str, Any] | None = None,
    milestone: bool = False,
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
            ai_participants=ai_participants,
            ai_suite=ai_suite,
            traces=traces or [],
            primary_model=primary_model,
            h6=h6,
            milestone=milestone,
        )
        html_text = _HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, default=_json_default))
        if milestone:
            html_text = _FULL_ONLY.sub("", html_text)
        output_path.write_text(html_text, encoding="utf-8")
        action.log(message_type="info", path=str(output_path), n_players=payload["n_players"], milestone=milestone)
        return output_path


_FULL_ONLY = re.compile(r"<!-- BEGIN_FULL_ONLY -->.*?<!-- END_FULL_ONLY -->", re.S)


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


def _player_rows(participants: pl.DataFrame) -> list[dict[str, Any]]:
    players: list[dict[str, Any]] = []
    for row in participants.iter_rows(named=True):
        category = str(row.get("cohort_category") or "unknown")
        study_id = str(row["study_id"])
        players.append(
            {
                "study_id": study_id[:12],
                "study_id_full": study_id,
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
    return players


def _hypothesis_blocks(suite_dict: dict[str, Any], h6: dict[str, Any] | None) -> dict[str, str]:
    blocks: dict[str, str] = {}
    for key in ("h1", "h2", "h3", "h4", "h5", "h6"):
        result = h6 if key == "h6" and h6 is not None else suite_dict.get(key)
        blocks[key] = markdown_fragment_to_html(explain_hypothesis_result(key, result))
    return blocks


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
    ai_participants: pl.DataFrame | None,
    ai_suite: Any | None,
    traces: list[dict[str, Any]],
    primary_model: str,
    h6: dict[str, Any] | None,
    milestone: bool = False,
) -> dict[str, Any]:
    opposite = opposite_trait_summary(participants)
    suite_dict = _as_dict(suite) or {}
    ai_suite_dict = _as_dict(ai_suite) or {}
    bench_dict = _as_dict(benchmarks) or {}
    ver_dict = _as_dict(verification) or {}
    format_counts = (
        runs.group_by("format").len().sort("format").to_dicts() if runs.height else []
    )
    compact_traces = [] if milestone else [_compact_trace(item) for item in traces]
    return {
        "edition": "milestone" if milestone else edition,
        "milestone": milestone,
        "title": "Sugar Sugar — human participants" if milestone else "Sugar Sugar study explorer",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_csv": str(source_csv),
        "n_players": participants.height,
        "n_runs": runs.height,
        "format_counts": format_counts,
        "players": _player_rows(participants),
        "ai_players": [] if milestone else (_player_rows(ai_participants) if ai_participants is not None else []),
        "cohort_labels": COHORT_LABELS,
        "reading_guide_html": markdown_fragment_to_html(
            explain_milestone_reading() if milestone else how_to_read_report()
        ),
        "edition_html": markdown_fragment_to_html(
            explain_milestone_page() if milestone else explain_edition_and_ai_path()
        ),
        "opposite_html": markdown_fragment_to_html(explain_opposite_trait(opposite)),
        "opposite": opposite,
        "hypotheses_html": _hypothesis_blocks(suite_dict, h6),
        "ai_hypotheses_html": _hypothesis_blocks(ai_suite_dict, h6),
        "hypothesis_catalog": HYPOTHESES,
        "benchmarks": bench_dict,
        "verification": ver_dict,
        "comparison": None if milestone else comparison,
        "evaluation_mode_labels": EVALUATION_MODE_LABELS,
        "sequences_dir": "" if milestone else (str(sequences_dir) if sequences_dir is not None else ""),
        "primary_model": "" if milestone else primary_model,
        "h6": {} if milestone else (h6 or {}),
        "traces": compact_traces,
        "line_styles": LINE_STYLES,
        "duration_caps": {
            "diabetes_duration_months": MAX_PLAUSIBLE_DIABETES_YEARS * 12.0,
            "cgm_duration_years": MAX_PLAUSIBLE_CGM_YEARS,
            "cgm_duration_months": MAX_PLAUSIBLE_CGM_YEARS * 12.0,
        },
    }


def _compact_trace(item: dict[str, Any]) -> dict[str, Any]:
    study_id = str(item.get("study_id") or "")
    return {
        "study_id": study_id,
        "study_id_short": study_id[:12],
        "run_id": str(item.get("run_id") or ""),
        "round_number": int(item.get("round_number") or 0),
        "format": str(item.get("format") or ""),
        "source": str(item.get("source") or ""),
        "data_source_name": str(item.get("data_source_name") or ""),
        "times": list(item.get("times") or []),
        "actual": list(item.get("actual") or []),
        "human": list(item.get("human") or []),
        "models": item.get("models") or {},
        "visible_end": int(item.get("visible_end") or 0),
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
    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 24px 0;
      background: #0f172a;
    }
    .tabs button {
      border: 0;
      background: transparent;
      color: #cbd5e1;
      font: inherit;
      font-weight: 650;
      padding: 10px 14px;
      border-radius: 10px 10px 0 0;
      cursor: pointer;
    }
    .tabs button.active {
      background: var(--paper);
      color: var(--ink);
    }
    main { padding: 20px 24px 48px; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
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
    .trace-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .trace-card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px 10px 4px;
    }
    .trace-card h4 {
      margin: 0 0 6px;
      font-size: 0.82rem;
      color: var(--muted);
      font-weight: 650;
    }
    .trace-card canvas { height: 200px !important; }
    .dev-row {
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }
    @media (max-width: 1000px) {
      .dev-row { grid-template-columns: 1fr; }
    }
    .dev-panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px 10px 4px;
    }
    .dev-panel h4 { margin: 0 0 6px; font-size: 0.9rem; }
    .dev-panel canvas { height: 280px !important; }
    .dev-model { margin: 18px 0; }
    .zoomable, .trace-card canvas, .dev-panel canvas { cursor: zoom-in; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 10px 0; }
    .chip {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 6px 12px;
      font: inherit;
      font-size: 0.88rem;
      cursor: pointer;
    }
    .chip.on { background: #0f172a; color: #fff; border-color: #0f172a; }
    .mix-wrap { display: grid; grid-template-columns: auto 1fr auto auto; gap: 10px; align-items: center; max-width: 640px; margin: 8px 0 14px; }
    .mix-wrap input[type="range"] { width: 100%; }
    .expand-chart {
      display: inline-block;
      margin: 0 0 6px;
      border: 1px solid var(--line);
      background: #0f172a;
      color: #fff;
      border-radius: 8px;
      padding: 4px 10px;
      font: inherit;
      font-size: 0.8rem;
      font-weight: 650;
      cursor: pointer;
    }
    #lightboxStage {
      flex: 1;
      min-height: 70vh;
      display: flex;
      align-items: stretch;
    }
    #lightboxImage {
      width: 100%;
      height: 70vh;
      object-fit: contain;
      background: #fff;
      border-radius: 12px;
      pointer-events: none;
    }
    #lightboxFocusBar {
      flex: 0 0 auto;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px 0 0;
    }
    #lightboxFocusBar .chip { color: #0f172a; background: #fff; }
    #lightboxFocusBar .chip.on { background: #38bdf8; border-color: #38bdf8; color: #0f172a; }
    #chartLightbox {
      display: none;
      position: fixed;
      inset: 0;
      z-index: 50;
      background: rgba(15, 23, 42, 0.78);
      padding: 16px 20px 20px;
      flex-direction: column;
    }
    #chartLightbox.open { display: flex; }
    #chartLightbox header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      color: #fff;
      margin-bottom: 10px;
    }
    #lightboxHint {
      flex: 1;
      font-size: 0.85rem;
      font-weight: 400;
      color: #cbd5e1;
    }
    #chartLightbox header button {
      border: 0;
      background: #fff;
      color: #0f172a;
      border-radius: 8px;
      padding: 8px 12px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    #chartLightbox canvas {
      flex: 1;
      width: 100% !important;
      min-height: 70vh !important;
      height: 70vh !important;
      background: #fff;
      border-radius: 12px;
    }
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
    .legend-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin: 10px 0 16px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.9rem;
      color: var(--muted);
    }
    .swatch {
      width: 42px;
      height: 0;
      border-top-width: 3px;
      border-top-style: solid;
    }
    .swatch.solid { border-top-style: solid; }
    .swatch.dashed { border-top-style: dashed; }
    .swatch.dotted { border-top-style: dotted; }
    .swatch.dashdot { border-top-style: dashed; }
    .swatch.longdash { border-top-style: dashed; border-top-width: 4px; }
    .people-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      align-items: center;
      margin: 12px 0 8px;
    }
    .people-toolbar button {
      border: 1px solid var(--line);
      background: #0f172a;
      color: white;
      border-radius: 8px;
      padding: 8px 14px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    .people-toolbar button:disabled { opacity: 0.4; cursor: default; }
    .people-caption { font-weight: 650; min-width: 16rem; }
    .missing-model {
      background: #f3e8ff;
      color: #6b21a8;
      border: 1px solid #d8b4fe;
      border-radius: 10px;
      padding: 10px 12px;
      margin: 8px 0 12px;
    }
    .empty { color: var(--muted); padding: 18px; background: var(--card); border: 1px dashed var(--line); border-radius: 12px; }
    #rows tr { cursor: pointer; }
    #rows tr.active { background: #ede9fe; }
  </style>
</head>
<body>
  <header>
    <div class="badge" id="editionBadge">Merged</div>
    <h1 id="pageTitle">Sugar Sugar study explorer</h1>
    <p id="headerLead">Human results, the same drawings on AI MAE, and a per-person chart: actual CGM, the human forecast, and each AI line. Colours and dashes are fixed in the legend.</p>
  </header>
  <nav class="tabs" id="tabs">
    <button type="button" data-tab="overview" class="active">Overview</button>
    <button type="button" data-tab="human">Human</button>
    <!-- BEGIN_FULL_ONLY -->
    <button type="button" data-tab="ai">AI</button>
    <button type="button" data-tab="people">People</button>
    <!-- END_FULL_ONLY -->
  </nav>
  <main>
    <div class="stats" id="stats"></div>
    <div class="filters" id="sharedFilters">
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

    <div id="tab-overview" class="tab-panel active">
      <section>
        <h2>How to read this</h2>
        <div class="prose" id="reading"></div>
        <div class="prose" id="editionNote"></div>
      </section>
      <!-- BEGIN_FULL_ONLY -->
      <section>
        <h2>H6 — humans vs models</h2>
        <div class="prose" id="h6Overview"></div>
        <div class="prose" id="aiSummary"></div>
      </section>
      <!-- END_FULL_ONLY -->
      <section>
        <h2>Literature / GlucoBench context</h2>
        <div class="prose" id="bench"></div>
      </section>
      <section>
        <h2>Data verification</h2>
        <div class="prose" id="verify"></div>
      </section>
    </div>

    <div id="tab-human" class="tab-panel">
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
        <!-- BEGIN_FULL_ONLY -->
        <div class="prose" id="h6Text"></div>
        <!-- END_FULL_ONLY -->
      </section>
      <section>
        <h2>Challenge the unknown / opposite trait</h2>
        <div class="pair">
          <div class="prose" id="oppositeText"></div>
          <div><canvas id="opposite"></canvas></div>
        </div>
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
    </div>

    <!-- BEGIN_FULL_ONLY -->
    <div id="tab-ai" class="tab-panel">
      <section>
        <h2>Human vs AI on the same task</h2>
        <p class="prose" id="aiPrimaryNote"></p>
        <p class="prose">Age and diabetes type do not move the model, so this tab does not repeat H1–H4. Each tick is one person, and only if that person has both a human score and an AI score on that task. Order is Generic (A), Own (B), Mixed (C). Mixed is format C, not a blend of A and B.</p>
        <div class="stats" id="aiTaskStats"></div>
        <div class="chip-row" id="aiCohortChips"></div>
        <div class="mix-wrap">
          <span>Human</span>
          <input type="range" id="aiMix" min="0" max="100" value="50" step="1" list="aiMixSnaps" />
          <span>AI</span>
          <button type="button" id="aiMixMiddle">50%</button>
        </div>
        <datalist id="aiMixSnaps">
          <option value="0" label="Human"></option>
          <option value="50" label="Both"></option>
          <option value="100" label="AI"></option>
        </datalist>
        <p class="prose">The slider snaps to the middle (and to each end). Left is only the human, right is only the model, middle is both at 50%. The four category buttons act on the cluster and the deviation charts below. Use Expand on a chart to open it full screen.</p>
        <div class="pair">
          <div>
            <h3>Per-task MAE</h3>
            <p class="prose">Same person, same task. Blue is the human, purple is the model. We expect the AI group to sit tighter; a person may be better on generic or on own.</p>
          </div>
          <div><canvas id="aiTaskMae" class="zoomable"></canvas></div>
        </div>
        <div class="pair">
          <div>
            <h3>Same user: generic vs own</h3>
            <p class="prose">Each person who has a generic score and an own score is one point. Colour is the diabetes × CGM category. A line joins that person's human point to their AI point when both exist. Hollow marks are people with only one of those tasks, sat on the equal-MAE line so they stay visible. Below the diagonal = better on own data.</p>
          </div>
          <div><canvas id="aiCluster" class="zoomable"></canvas></div>
        </div>
        <div class="prose" id="aiH6Text"></div>
      </section>
      <section>
        <h2>All-round deviations</h2>
        <p class="prose">Same 0% line as the sugar-sugar ending page. Rounds are folded into at most eight areas (four diabetes × CGM groups, human and AI). Human bands keep the category colours; AI bands use a second palette so they are not the same hue with a different dash. The edge is min–max; the thicker inner line is the median. Persistence, linear, and GluMind each get their own three panels.</p>
        <div class="legend-grid" id="aiDevLegend"></div>
        <div id="aiDevHost"></div>
      </section>
    </div>

    <div id="tab-people" class="tab-panel">
      <section>
        <h2>Per-person traces</h2>
        <p class="prose">One person at a time, every reconstructed round on one screen (up to 12). Slate is the original CGM, blue is the human forecast, orange / green / purple are the AI models. Under the traces, this person's hidden-hour error is the same kind of overlapping area as the AI tab: one band per series (human and each model), min–max edge and a median inside.</p>
        <div class="legend-grid" id="lineLegend"></div>
        <div id="missingModel" class="missing-model" hidden></div>
        <div class="people-toolbar">
          <button type="button" id="prevPerson">Previous</button>
          <div class="people-caption" id="personCaption">No person selected</div>
          <button type="button" id="nextPerson">Next</button>
        </div>
        <div id="traceEmpty" class="empty" hidden>No reconstructed 3-hour windows for this filter. Own-data rounds need the saved upload; generic rounds need the corpus file.</div>
        <div class="trace-grid" id="traceGrid"></div>
        <h3>This person's deviations</h3>
        <p class="prose">Horizontal 0% line, then percent error across the hidden hour. Generic (A), Own (B), Mixed (C). Each colour is one predictor — blue human, then a different colour per model — as a transparent area, not one line per round.</p>
        <div class="dev-row" id="personDevRow">
          <div class="dev-panel"><h4>Generic (A)</h4><canvas id="personDev-A" class="zoomable"></canvas></div>
          <div class="dev-panel"><h4>Own (B)</h4><canvas id="personDev-B" class="zoomable"></canvas></div>
          <div class="dev-panel"><h4>Mixed (C)</h4><canvas id="personDev-C" class="zoomable"></canvas></div>
        </div>
      </section>
    </div>
    <!-- END_FULL_ONLY -->

    <footer id="meta"></footer>
  </main>
  <div id="chartLightbox" aria-modal="true">
    <header>
      <strong id="lightboxTitle">Chart</strong>
      <span id="lightboxHint">Use the buttons under the chart to keep one category; the others fade to 25%. Click the same button again, or Show all.</span>
      <button type="button" id="closeLightbox">Close</button>
    </header>
    <div id="lightboxStage">
      <canvas id="lightboxCanvas"></canvas>
      <img id="lightboxImage" alt="Enlarged chart" hidden />
    </div>
    <div id="lightboxFocusBar" class="chip-row"></div>
  </div>
  <script>
    const DATA = __PAYLOAD__;
    const COLORS = {
      diabetic_cgm: "#E45756",
      diabetic_non_cgm: "#F58518",
      nondiabetic_cgm: "#4C78A8",
      nondiabetic_non_cgm: "#72B7B2",
      unknown: "#94a3b8",
    };
    const AI_COLORS = {
      diabetic_cgm: "#6d28d9",
      diabetic_non_cgm: "#0f766e",
      nondiabetic_cgm: "#db2777",
      nondiabetic_non_cgm: "#3f6212",
      unknown: "#1e293b",
    };
    const DASH = {
      solid: [],
      dashed: [10, 5],
      dotted: [2, 3],
      dashdot: [10, 4, 2, 4],
      longdash: [18, 6],
    };
    const POINT = {
      actual: "circle",
      human: "triangle",
      persistence: "rect",
      linear: "rectRot",
      glumind: "star",
      sugar_one: "star",
    };
    const TRACE_ORDER = ["actual", "human", "persistence", "linear", "glumind", "sugar_one"];
    const MAX_ROUNDS = 12;
    let personIndex = 0;
    let aiMix = 0.5;
    let lightboxFocus = null;
    const aiCohortOn = new Set(Object.keys(DATA.cohort_labels || {}));

    document.getElementById("pageTitle").textContent = DATA.title;
    const badge = document.getElementById("editionBadge");
    badge.textContent = DATA.edition === "merged" ? "Merged" : DATA.edition;
    document.getElementById("reading").innerHTML = DATA.reading_guide_html;
    document.getElementById("editionNote").innerHTML = DATA.edition_html;
    document.getElementById("oppositeText").innerHTML = DATA.opposite_html;
    if (DATA.milestone) {
      const lead = document.getElementById("headerLead");
      if (lead) lead.textContent = "Milestone view: human participants only. Overview explains the numbers; Human has the cohort, hypotheses, and the participant table.";
    }
    const aiPrimaryNote = document.getElementById("aiPrimaryNote");
    if (aiPrimaryNote) {
      aiPrimaryNote.textContent =
        DATA.primary_model
          ? ("Primary model: " + DATA.primary_model + ". Each point is one person on one task.")
          : "No primary model yet — reconstructed windows were empty.";
    }

    const cohortSelect = document.getElementById("cohortFilter");
    Object.entries(DATA.cohort_labels).forEach(([key, label]) => {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = label;
      cohortSelect.appendChild(opt);
    });

    document.querySelectorAll("#tabs button").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("#tabs button").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
      });
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
    function filtered(list) {
      const cohort = cohortSelect.value;
      const repeat = document.getElementById("repeatFilter").value;
      const q = document.getElementById("search").value.trim().toLowerCase();
      return (list || []).filter((p) => {
        if (cohort !== "all" && p.cohort !== cohort) return false;
        if (repeat === "single" && p.repeat) return false;
        if (repeat === "repeat" && !p.repeat) return false;
        if (repeat === "all_formats" && !p.all_formats) return false;
        if (repeat === "challenge" && !p.challenge_unknown) return false;
        if (repeat === "opposite" && !p.opposite_trait) return false;
        if (q && !p.study_id.toLowerCase().includes(q) && !(p.study_id_full || "").toLowerCase().includes(q)) return false;
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
      const overview = document.getElementById("h6Overview");
      if (overview) overview.innerHTML = (DATA.hypotheses_html || {}).h6 || "";
      const aiH6 = document.getElementById("aiH6Text");
      if (aiH6) aiH6.innerHTML = (DATA.hypotheses_html || {}).h6 || "";
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

    function renderAiSummary() {
      const el = document.getElementById("aiSummary");
      if (!el) return;
      const c = DATA.comparison;
      if (!c || !c.n_points) {
        el.innerHTML = `<p>No model scores on reconstructed windows yet. Sequences live in <code>${DATA.sequences_dir || "data/processed/ai"}</code>. Models only see the 3-hour game slice (24 visible points, left-padded to 128).</p>`;
        return;
      }
      const modelRows = Object.entries(c.model_mean_mae || {}).map(([name, mae]) => `<tr><td>${name}</td><td>${fmt(mae)}</td></tr>`).join("");
      el.innerHTML = `
        <p>Scored <strong>${c.n_points}</strong> points across <strong>${c.n_people}</strong> people.
        Human point MAE on that subset: <strong>${fmt(c.human_mean_mae)}</strong> mg/dL.
        Primary model: <code>${DATA.primary_model || "—"}</code>.</p>
        <table><thead><tr><th>Model</th><th>MAE</th></tr></thead><tbody>${modelRows}</tbody></table>`;
    }

    function renderLegend() {
      const box = document.getElementById("lineLegend");
      if (!box) return;
      const styles = DATA.line_styles || {};
      box.innerHTML = Object.entries(styles).map(([_key, style]) => `
        <div class="legend-item">
          <span class="swatch solid" style="border-top-color:${style.color}"></span>
          <span><strong>${style.label}</strong></span>
        </div>
      `).join("");
    }

    function playerMatches(traceId, player) {
      const full = player.study_id_full || player.study_id;
      return traceId === full || traceId === player.study_id || (full && full.startsWith(traceId)) || traceId.startsWith(player.study_id);
    }

    function personIds() {
      const rows = filtered(DATA.players);
      const fromTraces = [...new Set((DATA.traces || []).map((t) => t.study_id))];
      const ids = fromTraces.filter((id) => rows.some((p) => playerMatches(id, p)));
      ids.sort();
      return ids;
    }

    function tracesForPerson(studyId) {
      return (DATA.traces || [])
        .filter((t) => t.study_id === studyId)
        .sort((a, b) => {
          const fa = formatKey(a.format);
          const fb = formatKey(b.format);
          if (fa !== fb) return fa.localeCompare(fb);
          return (Number(a.round_number) - Number(b.round_number)) || String(a.run_id).localeCompare(String(b.run_id));
        });
    }

    function destroyChartsByPrefix(prefix) {
      Object.keys(charts).forEach((id) => {
        if (id.startsWith(prefix)) {
          charts[id].destroy();
          delete charts[id];
        }
      });
    }

    function destroyTraceCharts() {
      destroyChartsByPrefix("trace-");
      destroyChartsByPrefix("personDev-");
    }

    function modelKeys(trace) {
      const names = Object.keys(trace.models || {});
      const ordered = TRACE_ORDER.filter((k) => k !== "actual" && k !== "human" && names.includes(k));
      names.forEach((name) => {
        if (!ordered.includes(name)) ordered.push(name);
      });
      return ordered;
    }

    function showMissingModel(ids) {
      const note = document.getElementById("missingModel");
      const haveDeep = (DATA.traces || []).some((t) => t.models && (t.models.glumind || t.models.sugar_one));
      if (haveDeep || !ids.length) {
        note.hidden = true;
        note.textContent = "";
        return;
      }
      note.hidden = false;
      note.textContent = "Purple GluMind / SugarOne is missing because it was not scored. The sibling repo has GluMind weights in test_model, but this environment does not have torch loaded. Persistence (orange) and linear (green) are the AI lines present now.";
    }

    function lineSeries(label, values, key) {
      const style = (DATA.line_styles || {})[key] || { color: "#64748b", dash: "solid", label };
      return {
        label: style.label || label,
        data: values,
        borderColor: style.color,
        backgroundColor: style.color,
        borderWidth: key === "actual" ? 3.2 : 1.8,
        borderDash: [],
        pointStyle: POINT[key] || "circle",
        pointRadius: 1,
        pointHoverRadius: 3,
        pointHitRadius: 6,
        order: key === "actual" ? 10 : 1,
        spanGaps: false,
        tension: 0.15,
      };
    }

    function renderPersonStrip() {
      const ids = personIds();
      const empty = document.getElementById("traceEmpty");
      const grid = document.getElementById("traceGrid");
      if (!grid || !empty) return;
      const caption = document.getElementById("personCaption");
      const prev = document.getElementById("prevPerson");
      const next = document.getElementById("nextPerson");
      showMissingModel(ids);
      destroyTraceCharts();
      grid.innerHTML = "";
      if (!ids.length) {
        empty.hidden = false;
        caption.textContent = "No reconstructed windows";
        prev.disabled = true;
        next.disabled = true;
        renderPersonDeviations([]);
        return;
      }
      empty.hidden = true;
      if (personIndex >= ids.length) personIndex = 0;
      if (personIndex < 0) personIndex = ids.length - 1;
      const studyId = ids[personIndex];
      const player = (DATA.players || []).find((p) => playerMatches(studyId, p));
      const rounds = tracesForPerson(studyId).slice(0, MAX_ROUNDS);
      caption.textContent = `${personIndex + 1} / ${ids.length} · ${(player && player.study_id) || studyId.slice(0, 12)} · ${(player && player.cohort_label) || ""} · ${rounds.length} round${rounds.length === 1 ? "" : "s"}`;
      prev.disabled = ids.length < 2;
      next.disabled = ids.length < 2;
      document.querySelectorAll("#rows tr").forEach((tr) => {
        tr.classList.toggle("active", tr.dataset.study === ((player && player.study_id) || studyId.slice(0, 12)));
      });
      rounds.forEach((trace, i) => {
        const card = document.createElement("div");
        card.className = "trace-card";
        const title = document.createElement("h4");
        title.textContent = `r${trace.round_number} · ${trace.format || "?"} · ${trace.source || trace.data_source_name || ""}`;
        const canvas = document.createElement("canvas");
        canvas.id = "trace-" + i;
        canvas.className = "zoomable";
        card.appendChild(title);
        card.appendChild(canvas);
        grid.appendChild(card);
        const labels = (trace.times || []).map((t, idx) => {
          const bit = String(t).slice(11, 16);
          return bit || String(idx);
        });
        const datasets = [
          lineSeries("Human", trace.human, "human"),
        ];
        modelKeys(trace).forEach((name) => {
          datasets.push(lineSeries(name, trace.models[name], name));
        });
        datasets.push(lineSeries("Actual CGM", trace.actual, "actual"));
        upsertChart(canvas.id, "line", { labels, datasets }, {
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: { display: i === 0, position: "bottom", labels: { usePointStyle: true, boxWidth: 10 } },
            forecastGate: { index: Number(trace.visible_end || 0) },
          },
          scales: {
            x: { ticks: { maxTicksLimit: 6, font: { size: 9 } } },
            y: { title: { display: true, text: "mg/dL", font: { size: 10 } } },
          },
        });
      });
      renderPersonDeviations(rounds);
    }

    function stepPerson(delta) {
      const ids = personIds();
      if (!ids.length) return;
      personIndex = (personIndex + delta + ids.length) % ids.length;
      renderPersonStrip();
    }

    const DEV_FORMATS = [
      { key: "A", label: "Generic (A)" },
      { key: "B", label: "Own (B)" },
      { key: "C", label: "Mixed (C)" },
    ];

    function formatKey(fmt) {
      const upper = String(fmt || "").trim().toUpperCase();
      if (upper === "GENERIC") return "A";
      if (upper === "OWN") return "B";
      if (upper === "MIXED") return "C";
      return upper;
    }

    function playerByStudy(studyId) {
      return (DATA.players || []).find((p) => playerMatches(studyId, p));
    }

    function hiddenHour(values, visibleEnd) {
      return (values || []).slice(Number(visibleEnd || 0) + 1);
    }

    function percentErrors(actual, predicted) {
      const out = [];
      for (let i = 0; i < actual.length; i++) {
        const a = num(actual[i]);
        let p = num(predicted[i]);
        if (i === 0 && p === null && a !== null && a !== 0) p = a;
        if (a === null || p === null || a === 0) out.push(null);
        else out.push(((p - a) / a) * 100);
      }
      return out;
    }

    function minutePoints(pct) {
      return pct.map((y, i) => ({ x: (i + 1) * 5, y }));
    }

    function replaceChart(id, type, data, options) {
      const canvas = document.getElementById(id);
      if (!canvas) return;
      if (charts[id]) {
        charts[id].destroy();
        delete charts[id];
      }
      charts[id] = new Chart(canvas, {
        type,
        data,
        options: Object.assign({ animation: false, responsive: true, maintainAspectRatio: false }, options || {}),
      });
      bindEnlarge(canvas);
    }

    const zeroPctLine = {
      id: "zeroPctLine",
      afterDraw(chart, _args, opts) {
        if (!opts || !opts.enabled) return;
        const yScale = chart.scales.y;
        const xScale = chart.scales.x;
        if (!yScale || !xScale) return;
        const y = yScale.getPixelForValue(0);
        if (!Number.isFinite(y)) return;
        const ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = 2.6;
        ctx.setLineDash([]);
        ctx.beginPath();
        const left = chart.chartArea.left;
        const right = chart.chartArea.right;
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        ctx.fillStyle = "#0f172a";
        ctx.font = "12px sans-serif";
        ctx.fillText("0%", left + 4, y - 6);
        ctx.restore();
      },
    };
    Chart.register(zeroPctLine);
    if (Chart.Filler) Chart.register(Chart.Filler);

    function deviationScales() {
      return {
        x: {
          type: "linear",
          min: 5,
          max: 60,
          title: { display: true, text: "Minutes into the hidden hour" },
          ticks: { stepSize: 10, font: { size: 9 } },
        },
        y: {
          title: { display: true, text: "% error vs actual" },
          ticks: { callback: (v) => v + "%" },
        },
      };
    }

    function legendFilter(item) {
      const text = item && item.text;
      return !!text && text !== "undefined";
    }

    function renderPersonDeviations(rounds) {
      destroyChartsByPrefix("personDev-");
      DEV_FORMATS.forEach((fmt) => {
        const subset = rounds.filter((trace) => formatKey(trace.format) === fmt.key);
        const byName = { human: [] };
        subset.forEach((trace) => {
          const actual = hiddenHour(trace.actual, trace.visible_end);
          byName.human.push(percentErrors(actual, hiddenHour(trace.human, trace.visible_end)));
          modelKeys(trace).forEach((name) => {
            (byName[name] || (byName[name] = [])).push(
              percentErrors(actual, hiddenHour((trace.models || {})[name], trace.visible_end))
            );
          });
        });
        const datasets = [];
        const names = ["human"].concat(availableModels().filter((name) => (byName[name] || []).length));
        names.forEach((name) => {
          const style = (DATA.line_styles || {})[name] || { color: "#64748b", label: name };
          pushErrorBand(
            datasets,
            style.label || name,
            style.color,
            [],
            name === "human" ? "human" : "ai",
            byName[name]
          );
        });
        replaceChart("personDev-" + fmt.key, "scatter", { datasets }, {
          plugins: {
            legend: { position: "bottom", labels: { filter: legendFilter, boxWidth: 10 } },
            zeroPctLine: { enabled: true },
          },
          scales: deviationScales(),
        });
      });
    }

    function availableModels() {
      const names = new Set();
      (DATA.traces || []).forEach((trace) => {
        Object.keys(trace.models || {}).forEach((name) => names.add(name));
      });
      const ordered = TRACE_ORDER.filter((key) => key !== "actual" && key !== "human" && names.has(key));
      names.forEach((name) => {
        if (!ordered.includes(name)) ordered.push(name);
      });
      return ordered;
    }

    function hexAlpha(hex, alpha) {
      const raw = String(hex || "#64748b").replace("#", "");
      if (raw.length !== 6) return hex;
      const r = parseInt(raw.slice(0, 2), 16);
      const g = parseInt(raw.slice(2, 4), 16);
      const b = parseInt(raw.slice(4, 6), 16);
      return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
    }

    function renderAiDeviations(humanRows) {
      const host = document.getElementById("aiDevHost");
      const legend = document.getElementById("aiDevLegend");
      if (!host) return;
      const allowed = new Set(humanRows.map((p) => p.study_id_full || p.study_id));
      const traces = (DATA.traces || []).filter((trace) => {
        const player = playerByStudy(trace.study_id);
        if (!player) return false;
        if (!aiCohortOn.has(player.cohort || "unknown")) return false;
        return allowed.has(player.study_id_full || player.study_id) || allowed.has(player.study_id);
      });
      const models = availableModels();
      const cohortKeys = Object.keys(DATA.cohort_labels || {});
      if (legend) {
        legend.innerHTML = cohortKeys.map((key) => `
          <div class="legend-item">
            <span class="swatch solid" style="border-top-color:${COLORS[key] || "#94a3b8"}"></span>
            <span>Human · ${DATA.cohort_labels[key]}</span>
          </div>
        `).join("") + cohortKeys.map((key) => `
          <div class="legend-item">
            <span class="swatch solid" style="border-top-color:${AI_COLORS[key] || "#1e293b"}"></span>
            <span>AI · ${DATA.cohort_labels[key]}</span>
          </div>
        `).join("") + `
          <div class="legend-item"><span class="swatch solid" style="border-top-color:#0f172a"></span><span>0% (actual)</span></div>
        `;
      }
      destroyChartsByPrefix("aiDev-");
      const modelBlocks = models.length ? models : ["human_only"];
      host.innerHTML = modelBlocks.map((model) => {
        const title = model === "human_only"
          ? "Human only"
          : (((DATA.line_styles || {})[model] || {}).label || model);
        return `
          <div class="dev-model">
            <h3>${title}</h3>
            <p class="prose">${traces.filter((t) => (t.models || {})[model] || model === "human_only").length} reconstructed rounds in the current filter.</p>
            <div class="dev-row">
              ${DEV_FORMATS.map((fmt) => `
                <div class="dev-panel">
                  <h4>${fmt.label}</h4>
                  <canvas id="aiDev-${model}-${fmt.key}" class="zoomable"></canvas>
                </div>
              `).join("")}
            </div>
          </div>
        `;
      }).join("");
      modelBlocks.forEach((model) => {
        DEV_FORMATS.forEach((fmt) => {
          const subset = traces.filter((trace) => formatKey(trace.format) === fmt.key);
          const humanByCohort = {};
          const aiByCohort = {};
          subset.forEach((trace) => {
            const player = playerByStudy(trace.study_id);
            const cohort = (player && player.cohort) || "unknown";
            const actual = hiddenHour(trace.actual, trace.visible_end);
            const humanPct = percentErrors(actual, hiddenHour(trace.human, trace.visible_end));
            (humanByCohort[cohort] || (humanByCohort[cohort] = [])).push(humanPct);
            if (model === "human_only") return;
            const pred = (trace.models || {})[model];
            if (!pred) return;
            (aiByCohort[cohort] || (aiByCohort[cohort] = [])).push(
              percentErrors(actual, hiddenHour(pred, trace.visible_end))
            );
          });
          const datasets = [];
          const cohortKeysOn = Object.keys(DATA.cohort_labels || {});
          cohortKeysOn.forEach((cohort) => {
            const color = COLORS[cohort] || "#334155";
            const cohortLabel = DATA.cohort_labels[cohort] || cohort;
            pushErrorBand(datasets, "Human · " + cohortLabel, color, [], "human", humanByCohort[cohort]);
            pushErrorBand(datasets, "AI · " + cohortLabel, AI_COLORS[cohort] || "#1e293b", [], "ai", aiByCohort[cohort]);
          });
          replaceChart("aiDev-" + model + "-" + fmt.key, "scatter", { datasets }, {
            plugins: {
              legend: { position: "bottom", labels: { filter: legendFilter, boxWidth: 10 } },
              zeroPctLine: { enabled: true },
            },
            scales: deviationScales(),
          });
        });
      });
      applyWhoOpacity();
    }

    function seriesEnvelope(seriesList) {
      const n = (seriesList || []).reduce((max, series) => Math.max(max, series.length), 0);
      const upper = [];
      const lower = [];
      const mid = [];
      for (let i = 0; i < n; i++) {
        const vals = [];
        (seriesList || []).forEach((series) => {
          const value = series[i];
          if (value !== null && value !== undefined && Number.isFinite(value)) vals.push(value);
        });
        if (!vals.length) {
          upper.push(null);
          lower.push(null);
          mid.push(null);
          continue;
        }
        vals.sort((a, b) => a - b);
        lower.push(vals[0]);
        upper.push(vals[vals.length - 1]);
        mid.push(vals[Math.floor((vals.length - 1) / 2)]);
      }
      return { upper, lower, mid };
    }

    function pushErrorBand(datasets, label, color, dash, who, seriesList) {
      if (!seriesList || !seriesList.length) return;
      const env = seriesEnvelope(seriesList);
      if (!env.upper.some((value) => value !== null)) return;
      datasets.push({
        label,
        data: minutePoints(env.upper),
        borderColor: color,
        backgroundColor: hexAlpha(color, 0.2),
        borderWidth: 1.6,
        borderDash: dash || [],
        pointRadius: 0,
        fill: "+1",
        showLine: true,
        tension: 0.15,
        spanGaps: false,
        _who: who,
        _baseColor: color,
        _band: "upper",
        _group: label,
      });
      datasets.push({
        label: "",
        data: minutePoints(env.lower),
        borderColor: color,
        backgroundColor: "transparent",
        borderWidth: 1.6,
        borderDash: dash || [],
        pointRadius: 0,
        fill: false,
        showLine: true,
        tension: 0.15,
        spanGaps: false,
        _who: who,
        _baseColor: color,
        _band: "lower",
        _group: label,
      });
      datasets.push({
        label: "",
        data: minutePoints(env.mid),
        borderColor: color,
        backgroundColor: "transparent",
        borderWidth: 2.2,
        borderDash: dash || [],
        pointRadius: 0,
        fill: false,
        showLine: true,
        tension: 0.15,
        spanGaps: false,
        _who: who,
        _baseColor: color,
        _band: "mid",
        _group: label,
      });
    }

    const forecastGate = {
      id: "forecastGate",
      afterDraw(chart, _args, opts) {
        const idx = opts && opts.index;
        if (idx === undefined || idx === null) return;
        const x = chart.scales.x.getPixelForValue(idx);
        const { top, bottom } = chart.chartArea;
        const ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle = "#94a3b8";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, bottom);
        ctx.stroke();
        ctx.fillStyle = "#64748b";
        ctx.font = "12px sans-serif";
        ctx.fillText("forecast starts", x + 6, top + 14);
        ctx.restore();
      },
    };
    Chart.register(forecastGate);

    function renderPeopleTable(rows) {
      document.getElementById("rows").innerHTML = rows.map((p) => `
        <tr data-study="${p.study_id}">
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

    function renderStats(rows) {
      const allFormats = rows.filter((p) => p.all_formats && num(p.mae_generic) !== null && num(p.mae_own) !== null);
      const betterOwn = allFormats.filter((p) => p.mae_own < p.mae_generic).length;
      document.getElementById("stats").innerHTML = `
        <div class="stat"><b>${rows.length}</b><span>people in filter</span></div>
        <div class="stat"><b>${rows.filter((p) => p.repeat).length}</b><span>repeat players</span></div>
        <div class="stat"><b>${rows.filter((p) => p.challenge_unknown).length}</b><span>challenge the unknown</span></div>
        <div class="stat"><b>${rows.filter((p) => p.opposite_trait).length}</b><span>played opposite trait</span></div>
        <div class="stat"><b>${fmt(mean(rows.map((p) => num(p.mae_primary))))}</b><span>mean person MAE</span></div>
        <div class="stat"><b>${betterOwn}/${allFormats.length || 0}</b><span>all-variant better on own</span></div>
        <div class="stat"><b>${(DATA.traces || []).length}</b><span>reconstructed windows</span></div>
      `;
    }

    function renderSwarmBundle(prefix, rows) {
      const cohortKeys = Object.keys(DATA.cohort_labels);
      const pieId = prefix ? prefix + "Pie" : "pie";
      const swarmId = prefix ? prefix + "Swarm" : "swarm";
      const formatsId = prefix ? prefix + "Formats" : "formats";
      const h1Id = prefix ? prefix + "H1" : "h1";
      const h2Id = prefix ? prefix + "H2" : "h2";
      const h3Id = prefix ? prefix + "H3" : "h3";
      const h4Id = prefix ? prefix + "H4" : "h4";
      const clustersId = prefix ? prefix + "Clusters" : "clusters";
      const oppositeId = prefix ? prefix + "Opposite" : "opposite";

      upsertChart(pieId, "pie", {
        labels: cohortKeys.map((k) => DATA.cohort_labels[k]),
        datasets: [{ data: cohortKeys.map((k) => rows.filter((p) => p.cohort === k).length), backgroundColor: cohortKeys.map((k) => COLORS[k] || "#94a3b8") }],
      }, { maintainAspectRatio: true, aspectRatio: 1, plugins: { legend: { position: "bottom" } } });

      upsertChart(clustersId, "scatter", {
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

      upsertChart(swarmId, "scatter", {
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

      upsertChart(h1Id, "scatter", {
        datasets: layeredCategorySwarm(rows, (p) => p.diabetic === true, (p) => p.diabetic === false),
      }, layeredCategoryScales("PwD", "non-PwD"));

      upsertChart(h2Id, "scatter", {
        datasets: layeredCategorySwarm(rows, (p) => p.uses_cgm === true, (p) => p.uses_cgm === false),
      }, layeredCategoryScales("CGM user", "no CGM"));

      upsertChart(h3Id, "scatter", {
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

      upsertChart(h4Id, "scatter", {
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
      upsertChart(formatsId, "scatter", {
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

      if (document.getElementById(oppositeId)) {
        upsertChart(oppositeId, "scatter", {
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
      }
    }

    function render() {
      const rows = filtered(DATA.players);
      renderStats(rows);
      renderSwarmBundle("", rows);
      if (!DATA.milestone) {
        renderAiCompare(rows);
        renderAiDeviations(rows);
        renderPersonStrip();
      }
      renderPeopleTable(rows);
    }

    function firstNum() {
      for (let i = 0; i < arguments.length; i++) {
        const value = num(arguments[i]);
        if (value !== null) return value;
      }
      return null;
    }

    function taskMae(person, letter) {
      if (!person) return null;
      if (letter === "a") return firstNum(person.mae_a, person.mae_generic);
      if (letter === "b") return firstNum(person.mae_b, person.mae_own);
      return firstNum(person.mae_c);
    }

    function clusterPoint(person) {
      const x = taskMae(person, "a");
      const y = taskMae(person, "b");
      if (x !== null && y !== null) return { x, y, kind: "both" };
      if (x !== null) return { x, y: x, kind: "generic_only" };
      if (y !== null) return { x: y, y, kind: "own_only" };
      return null;
    }

    function pairedPlayers(humanRows) {
      const aiById = new Map((DATA.ai_players || []).map((p) => [p.study_id_full || p.study_id, p]));
      return humanRows.map((human) => ({
        human,
        ai: aiById.get(human.study_id_full) || aiById.get(human.study_id) || null,
      }));
    }

    function renderAiCompare(humanRows) {
      const pairs = pairedPlayers(humanRows).filter((p) => aiCohortOn.has(p.human.cohort || "unknown"));
      const tasks = [
        { key: "a", label: "Generic (A)" },
        { key: "b", label: "Own (B)" },
        { key: "c", label: "Mixed (C)" },
      ];
      const taskPairs = tasks.map((task) => pairs.filter((p) => {
        const hv = taskMae(p.human, task.key);
        const av = taskMae(p.ai, task.key);
        return hv !== null && av !== null;
      }));
      const bothSides = pairs.filter((p) => clusterPoint(p.human) && clusterPoint(p.human).kind === "both");
      const statsEl = document.getElementById("aiTaskStats");
      if (statsEl) {
        const linked = pairs.filter((p) => {
          const h = clusterPoint(p.human);
          const a = clusterPoint(p.ai);
          return h && a && h.kind === "both" && a.kind === "both";
        });
        const humanOwn = linked.filter((p) => clusterPoint(p.human).y < clusterPoint(p.human).x).length;
        statsEl.innerHTML = `
          <div class="stat"><b>${pairs.length}</b><span>people in these categories</span></div>
          <div class="stat"><b>${taskPairs[0].length}</b><span>paired on generic</span></div>
          <div class="stat"><b>${taskPairs[1].length}</b><span>paired on own</span></div>
          <div class="stat"><b>${taskPairs[2].length}</b><span>paired on mixed (C)</span></div>
          <div class="stat"><b>${bothSides.length}</b><span>have generic and own</span></div>
          <div class="stat"><b>${linked.length}</b><span>human+AI both tasks</span></div>
          <div class="stat"><b>${humanOwn}/${linked.length || 0}</b><span>human better on own</span></div>
        `;
      }

      replaceChart("aiTaskMae", "scatter", {
        datasets: tasks.flatMap((task, idx) => {
          const ready = taskPairs[idx];
          return [
            {
              label: "Human · " + task.label,
              data: ready.map((p) => ({
                x: idx - 0.16 + jitter(p.human.study_id + task.key + "h", 0.12),
                y: taskMae(p.human, task.key),
              })),
              backgroundColor: "#2563eb",
              pointRadius: 5,
              _who: "human",
              _baseColor: "#2563eb",
            },
            {
              label: "AI · " + task.label,
              data: ready.map((p) => ({
                x: idx + 0.16 + jitter(p.human.study_id + task.key + "a", 0.12),
                y: taskMae(p.ai, task.key),
              })),
              backgroundColor: "#9333ea",
              pointRadius: 5,
              _who: "ai",
              _baseColor: "#9333ea",
            },
          ];
        }),
      }, {
        plugins: { legend: { position: "bottom" } },
        scales: {
          x: {
            min: -0.5,
            max: 2.5,
            ticks: {
              stepSize: 1,
              callback: (v) => Number.isInteger(v) ? ((tasks[v] || {}).label || "") : "",
            },
          },
          y: { title: { display: true, text: "Person MAE on that task (mg/dL)" } },
        },
      });

      const cohortKeys = Object.keys(DATA.cohort_labels || {});
      const datasets = [];
      pairs.forEach((p) => {
        const humanPt = clusterPoint(p.human);
        const aiPt = clusterPoint(p.ai);
        if (humanPt && aiPt && humanPt.kind === "both" && aiPt.kind === "both") {
          datasets.push({
            data: [humanPt, aiPt],
            showLine: true,
            borderColor: "#475569",
            backgroundColor: "transparent",
            pointRadius: 0,
            borderWidth: 1.2,
            _who: "link",
          });
        }
      });
      cohortKeys.forEach((key) => {
        const color = COLORS[key] || "#334155";
        const label = DATA.cohort_labels[key];
        const humans = pairs.filter((p) => (p.human.cohort || "unknown") === key).map((p) => clusterPoint(p.human)).filter(Boolean);
        const ais = pairs.filter((p) => (p.human.cohort || "unknown") === key).map((p) => clusterPoint(p.ai)).filter(Boolean);
        datasets.push({
          label: "Human · " + label,
          data: humans,
          backgroundColor: color,
          borderColor: color,
          pointRadius: humans.map((pt) => pt.kind === "both" ? 6 : 4),
          pointStyle: humans.map((pt) => pt.kind === "both" ? "circle" : "rectRot"),
          _who: "human",
          _baseColor: color,
        });
        datasets.push({
          label: "AI · " + label,
          data: ais,
          backgroundColor: color,
          borderColor: "#0f172a",
          pointRadius: ais.map((pt) => pt.kind === "both" ? 6 : 4),
          pointStyle: ais.map((pt) => pt.kind === "both" ? "triangle" : "rectRot"),
          _who: "ai",
          _baseColor: color,
        });
      });
      replaceChart("aiCluster", "scatter", { datasets }, {
        plugins: { legend: { position: "bottom" } },
        scales: {
          x: { title: { display: true, text: "Generic (A) MAE" } },
          y: { title: { display: true, text: "Own (B) MAE" } },
        },
      });
      applyWhoOpacity();
    }

    function applyWhoOpacity() {
      const humanA = 1 - aiMix;
      const aiA = aiMix;
      Object.keys(charts).forEach((id) => {
        const chart = charts[id];
        if (!chart || !chart.data || !chart.data.datasets) return;
        let touched = false;
        if (!id.startsWith("ai")) return;
        chart.data.datasets.forEach((ds) => {
          if (!ds._who) return;
          touched = true;
          if (ds._who === "link") {
            ds.hidden = humanA < 0.02 || aiA < 0.02;
            ds.borderColor = hexAlpha("#475569", Math.min(humanA, aiA) * 0.9);
            return;
          }
          const alpha = ds._who === "human" ? humanA : aiA;
          ds.hidden = alpha < 0.02;
          const base = ds._baseColor || "#334155";
          if (ds._band === "upper") {
            ds.borderColor = hexAlpha(base, 0.4 + 0.55 * alpha);
            ds.backgroundColor = hexAlpha(base, 0.08 + 0.18 * alpha);
            return;
          }
          if (ds._band === "lower" || ds._band === "mid") {
            ds.borderColor = hexAlpha(base, 0.4 + 0.55 * alpha);
            ds.backgroundColor = "transparent";
            return;
          }
          const painted = hexAlpha(base, 0.25 + 0.7 * alpha);
          if (ds.borderColor && !Array.isArray(ds.borderColor)) ds.borderColor = painted;
          if (ds.backgroundColor && !Array.isArray(ds.backgroundColor)) ds.backgroundColor = painted;
        });
        if (touched) chart.update("none");
      });
    }

    function fillAiCohortChips() {
      const host = document.getElementById("aiCohortChips");
      if (!host) return;
      const keys = Object.keys(DATA.cohort_labels || {});
      if (!aiCohortOn.size) keys.forEach((key) => aiCohortOn.add(key));
      host.innerHTML = keys.map((key) => `
        <button type="button" class="chip ${aiCohortOn.has(key) ? "on" : ""}" data-cohort="${key}">${DATA.cohort_labels[key]}</button>
      `).join("");
    }

    function bindEnlarge(canvas) {
      if (!canvas || !canvas.id || canvas.id === "lightboxCanvas" || canvas.dataset.enlargeBound) return;
      canvas.dataset.enlargeBound = "1";
      canvas.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        openLightbox(canvas.id);
      }, true);
      const parent = canvas.parentElement;
      if (parent && !parent.querySelector(':scope > .expand-chart[data-for="' + canvas.id + '"]')) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "expand-chart";
        btn.dataset.for = canvas.id;
        btn.textContent = "Expand";
        parent.insertBefore(btn, canvas);
        btn.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          openLightbox(canvas.id);
        });
      }
    }

    function chartTypeOf(chart) {
      if (!chart || !chart.config) return "scatter";
      return chart.config.type || (chart.config._config && chart.config._config.type) || "scatter";
    }

    function cloneDataset(ds) {
      const data = (ds.data || []).map((point) => {
        if (point === null || point === undefined) return null;
        if (typeof point === "number") return point;
        if (typeof point === "object") {
          const row = {};
          if ("x" in point) row.x = point.x;
          if ("y" in point) row.y = point.y;
          return row;
        }
        return point;
      });
      const label = ds.label && ds.label !== "undefined" ? ds.label : "";
      const defaultRadius = ds.showLine ? 0 : 5;
      return {
        label,
        data,
        borderColor: ds.borderColor,
        backgroundColor: ds.backgroundColor,
        borderWidth: ds.borderWidth || 1.2,
        borderDash: ds.borderDash || [],
        pointRadius: Array.isArray(ds.pointRadius) ? ds.pointRadius.slice() : (ds.pointRadius == null ? defaultRadius : ds.pointRadius),
        pointHoverRadius: ds.pointHoverRadius,
        pointStyle: Array.isArray(ds.pointStyle) ? ds.pointStyle.slice() : ds.pointStyle,
        showLine: ds.showLine,
        spanGaps: ds.spanGaps,
        tension: ds.tension,
        hidden: ds.hidden,
        fill: ds.fill === undefined ? false : ds.fill,
        _who: ds._who,
        _baseColor: ds._baseColor,
        _band: ds._band,
        _group: ds._group || label,
        _fullBorder: ds.borderColor,
        _fullBackground: ds.backgroundColor,
      };
    }

    function scaleColorAlpha(color, factor) {
      if (!color || color === "transparent") return color;
      if (Array.isArray(color)) return color.map((item) => scaleColorAlpha(item, factor));
      const text = String(color);
      const rgba = text.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([0-9.]+))?\s*\)$/);
      if (rgba) {
        const alpha = rgba[4] === undefined ? 1 : Number(rgba[4]);
        return "rgba(" + rgba[1] + "," + rgba[2] + "," + rgba[3] + "," + (alpha * factor) + ")";
      }
      if (text.charAt(0) === "#") return hexAlpha(text, factor);
      return color;
    }

    function datasetGroup(ds) {
      return (ds && (ds._group || ds.label)) || "";
    }

    function applyLightboxFocus(chart) {
      if (!chart || !chart.data || !chart.data.datasets) return;
      chart.data.datasets.forEach((ds) => {
        const group = datasetGroup(ds);
        const factor = (!lightboxFocus || group === lightboxFocus) ? 1 : 0.25;
        const border = ds._fullBorder !== undefined ? ds._fullBorder : ds.borderColor;
        const background = ds._fullBackground !== undefined ? ds._fullBackground : ds.backgroundColor;
        ds.borderColor = factor === 1 ? border : scaleColorAlpha(border, factor);
        ds.backgroundColor = factor === 1 ? background : scaleColorAlpha(background, factor);
      });
      chart.update("none");
      paintLightboxFocusBar(chart);
    }

    function lightboxGroups(chart) {
      const groups = [];
      const colors = {};
      (chart.data.datasets || []).forEach((ds) => {
        const group = datasetGroup(ds);
        if (!group || groups.includes(group)) return;
        groups.push(group);
        colors[group] = ds._baseColor || ds._fullBorder || ds.borderColor || "#334155";
      });
      return { groups, colors };
    }

    function paintLightboxFocusBar(chart) {
      const bar = document.getElementById("lightboxFocusBar");
      if (!bar) return;
      const packed = lightboxGroups(chart);
      bar._groups = packed.groups;
      bar.innerHTML = packed.groups.map((group, index) => {
        const focused = lightboxFocus === group;
        const dim = lightboxFocus && !focused;
        const color = packed.colors[group];
        return `<button type="button" class="chip${focused ? " on" : ""}" data-focus-i="${index}" style="border-color:${color};${dim ? "opacity:0.4" : ""}">${group}</button>`;
      }).join("") + (lightboxFocus ? `<button type="button" class="chip" data-focus-i="-1">Show all</button>` : "");
    }

    function setLightboxFocus(group, chart) {
      lightboxFocus = group || null;
      applyLightboxFocus(chart);
    }

    function onLightboxChartClick(_event, elements, chart) {
      if (!elements.length) {
        if (lightboxFocus) setLightboxFocus(null, chart);
        return;
      }
      const group = datasetGroup(chart.data.datasets[elements[0].datasetIndex]);
      if (!group) return;
      setLightboxFocus(lightboxFocus === group ? null : group, chart);
    }

    function lightboxScales(src) {
      const out = {};
      if (!src.scales) return out;
      Object.keys(src.scales).forEach((key) => {
        const scale = src.scales[key];
        const opt = scale.options || {};
        out[key] = {
          type: scale.type || opt.type,
          min: opt.min,
          max: opt.max,
          title: opt.title ? { display: !!opt.title.display, text: opt.title.text } : undefined,
          ticks: opt.ticks && opt.ticks.stepSize != null ? { stepSize: opt.ticks.stepSize } : undefined,
        };
      });
      return out;
    }

    function openLightbox(sourceId) {
      lightboxFocus = null;
      const src = charts[sourceId] || (typeof Chart.getChart === "function" ? Chart.getChart(sourceId) : null);
      const box = document.getElementById("chartLightbox");
      const stageCanvas = document.getElementById("lightboxCanvas");
      const stageImage = document.getElementById("lightboxImage");
      if (!src || !box || !stageCanvas || !stageImage) return;
      document.getElementById("lightboxTitle").textContent = sourceId;
      if (charts.lightboxCanvas) {
        charts.lightboxCanvas.destroy();
        delete charts.lightboxCanvas;
      }
      const existing = typeof Chart.getChart === "function" ? Chart.getChart(stageCanvas) : null;
      if (existing) existing.destroy();
      stageCanvas.hidden = true;
      stageImage.hidden = false;
      stageImage.src = src.canvas.toDataURL("image/png");
      box.classList.add("open");
      requestAnimationFrame(() => {
        const width = Math.max(640, Math.floor(box.clientWidth - 48));
        const height = Math.max(400, Math.floor(box.clientHeight - 88));
        try {
          stageCanvas.hidden = false;
          stageCanvas.style.width = width + "px";
          stageCanvas.style.height = height + "px";
          charts.lightboxCanvas = new Chart(stageCanvas, {
            type: chartTypeOf(src),
            data: {
              labels: src.data.labels ? src.data.labels.slice() : undefined,
              datasets: src.data.datasets.map(cloneDataset),
            },
            options: {
              animation: false,
              responsive: true,
              maintainAspectRatio: false,
              onClick: onLightboxChartClick,
              plugins: {
                legend: { display: false },
                zeroPctLine: {
                  enabled: !!(src.options && src.options.plugins && src.options.plugins.zeroPctLine && src.options.plugins.zeroPctLine.enabled),
                },
                forecastGate: src.options && src.options.plugins ? src.options.plugins.forecastGate : undefined,
              },
              scales: lightboxScales(src),
            },
          });
          stageImage.hidden = true;
          paintLightboxFocusBar(charts.lightboxCanvas);
        } catch (err) {
          stageCanvas.hidden = true;
          stageImage.hidden = false;
        }
      });
    }

    function closeLightbox() {
      lightboxFocus = null;
      const bar = document.getElementById("lightboxFocusBar");
      if (bar) bar.innerHTML = "";
      const box = document.getElementById("chartLightbox");
      box.classList.remove("open");
      const stageImage = document.getElementById("lightboxImage");
      if (stageImage) {
        stageImage.hidden = true;
        stageImage.removeAttribute("src");
      }
      const stageCanvas = document.getElementById("lightboxCanvas");
      if (stageCanvas) stageCanvas.hidden = false;
      if (charts.lightboxCanvas) {
        charts.lightboxCanvas.destroy();
        delete charts.lightboxCanvas;
      }
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
        charts[id].data.labels = data.labels;
        charts[id].data.datasets = data.datasets;
        if (options) Object.assign(charts[id].options, options);
        charts[id].update();
        return;
      }
      charts[id] = new Chart(canvas, { type, data, options });
      bindEnlarge(canvas);
    }

    document.getElementById("cohortFilter").addEventListener("change", render);
    document.getElementById("repeatFilter").addEventListener("change", render);
    document.getElementById("search").addEventListener("input", render);
    fillAiCohortChips();
    const aiCohortHost = document.getElementById("aiCohortChips");
    if (aiCohortHost) {
      aiCohortHost.addEventListener("click", (event) => {
        const btn = event.target.closest("[data-cohort]");
        if (!btn) return;
        const key = btn.dataset.cohort;
        if (aiCohortOn.has(key)) aiCohortOn.delete(key);
        else aiCohortOn.add(key);
        if (!aiCohortOn.size) {
          Object.keys(DATA.cohort_labels || {}).forEach((item) => aiCohortOn.add(item));
        }
        fillAiCohortChips();
        const rows = filtered(DATA.players);
        renderAiCompare(rows);
        renderAiDeviations(rows);
      });
    }
    function snapMix(raw) {
      const n = Number(raw);
      if (Math.abs(n - 50) <= 8) return 50;
      if (n <= 4) return 0;
      if (n >= 96) return 100;
      return n;
    }
    function setMix(raw) {
      const snapped = snapMix(raw);
      const slider = document.getElementById("aiMix");
      if (slider && Number(slider.value) !== snapped) slider.value = String(snapped);
      aiMix = snapped / 100;
      applyWhoOpacity();
    }
    const aiMixEl = document.getElementById("aiMix");
    if (aiMixEl) aiMixEl.addEventListener("input", (event) => setMix(event.target.value));
    const aiMixMiddle = document.getElementById("aiMixMiddle");
    if (aiMixMiddle) aiMixMiddle.addEventListener("click", () => setMix(50));
    document.getElementById("closeLightbox").addEventListener("click", closeLightbox);
    document.getElementById("chartLightbox").addEventListener("click", (event) => {
      if (event.target.id === "chartLightbox" || event.target.id === "lightboxStage") closeLightbox();
    });
    document.getElementById("lightboxFocusBar").addEventListener("click", (event) => {
      event.stopPropagation();
      const btn = event.target.closest("[data-focus-i]");
      if (!btn || !charts.lightboxCanvas) return;
      const index = Number(btn.getAttribute("data-focus-i"));
      const groups = document.getElementById("lightboxFocusBar")._groups || [];
      const group = index < 0 ? null : groups[index];
      setLightboxFocus(group && lightboxFocus === group ? null : group, charts.lightboxCanvas);
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeLightbox();
    });
    const prevPerson = document.getElementById("prevPerson");
    const nextPerson = document.getElementById("nextPerson");
    if (prevPerson) prevPerson.addEventListener("click", () => stepPerson(-1));
    if (nextPerson) nextPerson.addEventListener("click", () => stepPerson(1));
    document.getElementById("rows").addEventListener("click", (event) => {
      const row = event.target.closest("tr");
      if (!row || !row.dataset.study) return;
      const ids = personIds();
      const idx = ids.findIndex((id) => id === row.dataset.study || id.startsWith(row.dataset.study) || row.dataset.study.startsWith(id.slice(0, 12)));
      if (idx < 0) return;
      personIndex = idx;
      renderPersonStrip();
    });
    document.addEventListener("keydown", (event) => {
      const peopleTab = document.getElementById("tab-people");
      if (!peopleTab || !peopleTab.classList.contains("active")) return;
      if (event.key === "ArrowRight") stepPerson(1);
      if (event.key === "ArrowLeft") stepPerson(-1);
    });
    document.getElementById("meta").textContent =
      `Generated ${DATA.generated_at} | source ${DATA.source_csv} | ${DATA.n_runs} runs / ${DATA.n_players} people | ${DATA.edition}`;
    renderHypotheses();
    renderBench();
    renderVerify();
    renderAiSummary();
    renderLegend();
    render();
  </script>
</body>
</html>
"""
