"""Self-contained HTML explorer: Overview, Human, AI, and per-user traces."""

from __future__ import annotations

import base64
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
from sugar_data_processing.gathering.sources import classify_data_class
from sugar_data_processing.output.narration import (
    explain_edition_and_ai_path,
    explain_hypothesis_result,
    explain_milestone_page,
    explain_milestone_reading,
    explain_opposite_trait,
    how_to_read_report,
)
from sugar_data_processing.statistics.catalog import HYPOTHESES
from sugar_data_processing.statistics.progress import study_progress

LOGO_PATH: Path = Path(__file__).parent / "assets" / "glucosedao_logo.jpeg"
ICON_PATH: Path = Path(__file__).parent / "assets" / "sugar_sugar_icon.svg"


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
        html_text = (
            _HTML_TEMPLATE.replace("__LOGO_URI__", _logo_data_uri())
            .replace("__ICON_URI__", _icon_data_uri())
            .replace("__PAYLOAD__", json.dumps(payload, default=_json_default))
        )
        if milestone:
            html_text = _FULL_ONLY.sub("", html_text)
        output_path.write_text(html_text, encoding="utf-8")
        action.log(message_type="info", path=str(output_path), n_players=payload["n_players"], milestone=milestone)
        return output_path


_FULL_ONLY = re.compile(r"<!-- BEGIN_FULL_ONLY -->.*?<!-- END_FULL_ONLY -->", re.S)


def _logo_data_uri() -> str:
    """Inline the GlucoseDAO logo so the published page stays a single file."""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _icon_data_uri() -> str:
    """Inline the Sugar-Sugar mark used for the brand and the favicon."""
    encoded = base64.b64encode(ICON_PATH.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _without_leading_heading(fragment: str) -> str:
    """The dashboard puts the guide under its own summary line."""
    return re.sub(r"^\s*<h[2-4]>.*?</h[2-4]>\s*", "", fragment, count=1)


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
                out.append("<thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
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
                "mae_dia_data": row.get("mae_generic_diabetic_data"),
                "n_dia_data": int(row.get("n_rounds_generic_diabetic_data") or 0),
                "mae_nondia_data": row.get("mae_generic_nondiabetic_data"),
                "n_nondia_data": int(row.get("n_rounds_generic_nondiabetic_data") or 0),
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
    diabetic_by_id = {
        str(row["study_id"]): row.get("diabetic") for row in participants.iter_rows(named=True)
    }
    compact_traces = [] if milestone else [_compact_trace(item, diabetic_by_id) for item in traces]
    return {
        "edition": "milestone" if milestone else edition,
        "milestone": milestone,
        "title": "Sugar-Sugar: human participants" if milestone else "Sugar-Sugar study",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_csv": str(source_csv),
        "n_players": participants.height,
        "n_runs": runs.height,
        "format_counts": format_counts,
        "players": _player_rows(participants),
        "ai_players": [] if milestone else (_player_rows(ai_participants) if ai_participants is not None else []),
        "cohort_labels": COHORT_LABELS,
        "reading_guide_html": _without_leading_heading(
            markdown_fragment_to_html(explain_milestone_reading() if milestone else how_to_read_report())
        ),
        "edition_html": markdown_fragment_to_html(
            explain_milestone_page() if milestone else explain_edition_and_ai_path()
        ),
        "opposite_html": markdown_fragment_to_html(explain_opposite_trait(opposite)),
        "opposite": opposite,
        "hypotheses_html": _hypothesis_blocks(suite_dict, h6),
        "ai_hypotheses_html": _hypothesis_blocks(ai_suite_dict, h6),
        "hypothesis_catalog": HYPOTHESES,
        "progress": study_progress(
            participants,
            suite_dict,
            h6=None if milestone else h6,
            benchmarks=bench_dict,
            linked_identities="original_study_id" in runs.columns,
        ),
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


def _compact_trace(item: dict[str, Any], diabetic_by_id: dict[str, Any]) -> dict[str, Any]:
    study_id = str(item.get("study_id") or "")
    source = str(item.get("source") or "")
    data_class = classify_data_class(
        str(item.get("data_source_name") or ""),
        is_example=(source == "generic") if source else None,
        player_diabetic=diabetic_by_id.get(study_id),
    )
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
        "data_class": data_class or "",
    }


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sugar-Sugar study dashboard</title>
  <link rel="icon" type="image/svg+xml" href="__ICON_URI__" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700;900&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --ink: #0f172a;
      --text: #334155;
      --muted: #64748b;
      --card: #ffffff;
      --line: rgba(15, 23, 42, 0.1);
      --soft: #f1f5f9;
      --blue: #2185d0;
      --blue-deep: #1565c0;
      --accent: #2185d0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Lato, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      font-size: 15px;
      line-height: 1.5;
      color: var(--text);
      background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 35%, #fff7ed 100%) fixed;
      min-height: 100vh;
    }
    a { color: var(--blue-deep); }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      align-items: stretch;
      gap: 8px;
      padding: 0 16px;
      background: var(--blue);
      box-shadow: 0 3px 10px rgba(0, 0, 0, 0.18);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 14px 8px 0;
      color: #fff;
      font-weight: 900;
      font-size: 1.05rem;
      text-decoration: none;
      white-space: nowrap;
    }
    .brand img {
      width: 38px;
      height: 38px;
      border-radius: 10px;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.18);
    }
    .brand span { font-size: 1.2rem; letter-spacing: 0.01em; }
    .tabs { display: flex; flex-wrap: wrap; align-items: stretch; flex: 1; }
    .tabs button {
      border: 0;
      background: transparent;
      color: rgba(255, 255, 255, 0.92);
      font: inherit;
      font-size: 1rem;
      font-weight: 700;
      padding: 14px 16px;
      cursor: pointer;
    }
    .tabs button:hover { background: rgba(0, 0, 0, 0.06); }
    .tabs button.active { background: rgba(0, 0, 0, 0.14); color: #fff; }
    .play, .cta {
      color: #fff;
      font-weight: 900;
      text-decoration: none;
      white-space: nowrap;
      background: #f97316;
      border-radius: 10px;
      box-shadow: 0 2px 8px rgba(249, 115, 22, 0.35);
    }
    .play { align-self: center; padding: 8px 16px; }
    .play:hover, .cta:hover { background: #ea580c; }
    .cta-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; margin: 16px 0 4px; }
    .cta { display: inline-block; padding: 12px 22px; font-size: 1.1rem; }
    .cta-row span { color: var(--muted); max-width: 46ch; }
    .byline { margin: 10px 0 0 !important; color: var(--muted); font-size: 0.85rem !important; }
    .byline a { color: var(--muted); font-weight: 700; text-decoration: none; }
    .byline img { width: 16px; height: 16px; border-radius: 4px; margin: 0 4px 0 2px; vertical-align: -3px; }
    .wrap { max-width: 1240px; margin: 0 auto; padding: 0 20px; }
    .hero { padding: 28px 0 6px; }
    .eyebrow {
      display: inline-block;
      margin-bottom: 6px;
      color: var(--blue-deep);
      font-size: 0.8rem;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .hero h1 { margin: 0 0 6px; color: var(--ink); font-size: 2.6rem; font-weight: 900; line-height: 1.1; }
    .hero p { margin: 0; max-width: 72ch; font-size: 1.08rem; }
    main { padding: 8px 0 48px; }
    [hidden] { display: none !important; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    section { margin: 24px 0; }
    h2 { margin: 0 0 6px; color: var(--ink); font-size: 1.35rem; font-weight: 900; }
    h3 { color: var(--ink); }
    section > h2 { margin-bottom: 6px; }
    .lead { margin: 0 0 14px; max-width: 80ch; color: var(--text); }
    .note { margin: 10px 0 0; color: var(--muted); font-size: 0.88rem; }
    .prose { max-width: 90ch; color: var(--text); }
    .prose table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin: 8px 0 16px; }
    .prose th, .prose td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
    .prose code { background: var(--soft); padding: 1px 5px; border-radius: 4px; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px 20px;
    }
    .card h2, .card h3 { margin: 0 0 8px; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
    .kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }
    @media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    .kpi b { display: block; color: var(--blue-deep); font-size: 2.6rem; font-weight: 900; line-height: 1.05; }
    .kpi span { display: block; color: var(--ink); font-weight: 700; }
    .kpi small { display: block; color: var(--muted); font-size: 0.85rem; }
    .target-track { margin: 34px 0 4px; }
    .track {
      position: relative;
      height: 18px;
      border-radius: 999px;
      background: var(--soft);
      border: 1px solid var(--line);
      overflow: visible;
    }
    .track .fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #2185d0, #1565c0); }
    .track .mark { position: absolute; top: -6px; bottom: -6px; width: 2px; background: var(--ink); }
    .track .mark span {
      position: absolute;
      top: 26px;
      left: 50%;
      transform: translateX(-50%);
      white-space: nowrap;
      color: var(--muted);
      font-size: 0.8rem;
    }
    .track .mark.up span { top: auto; bottom: 26px; }
    .track .mark.end span { transform: translateX(-100%); }
    .track-labels { display: flex; justify-content: space-between; margin-top: 30px; font-size: 0.88rem; }
    .findings { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
    .finding {
      display: flex;
      flex-direction: column;
      gap: 6px;
      background: var(--card);
      border: 1px solid var(--line);
      border-left: 5px solid #94a3b8;
      border-radius: 14px;
      padding: 14px 16px;
    }
    .finding .q { color: var(--muted); font-size: 0.85rem; font-weight: 700; }
    .finding .q b { color: var(--blue-deep); margin-right: 6px; }
    .finding .headline { color: var(--ink); font-size: 1.05rem; font-weight: 700; }
    .finding .detail { color: var(--text); font-size: 0.9rem; }
    .verdict {
      align-self: flex-start;
      border-radius: 999px;
      padding: 2px 10px;
      font-size: 0.78rem;
      font-weight: 900;
      letter-spacing: 0.02em;
    }
    .t-good { border-left-color: #16a34a; background: #f0fdf4; }
    .t-good .verdict { background: #dcfce7; color: #166534; }
    .t-warn { border-left-color: #eab308; background: #fefce8; }
    .t-warn .verdict { background: #fef9c3; color: #854d0e; }
    .t-bad { border-left-color: #dc2626; background: #fef2f2; }
    .t-bad .verdict { background: #fee2e2; color: #991b1b; }
    .tone-legend { display: flex; flex-wrap: wrap; gap: 6px 20px; margin: 0 0 14px; font-size: 0.9rem; color: var(--text); }
    .tone-legend span::before {
      content: "";
      display: inline-block;
      width: 12px;
      height: 12px;
      margin-right: 7px;
      border-radius: 3px;
      vertical-align: -1px;
      background: var(--tone);
    }
    .cell { margin: 12px 0 16px; }
    .cell-head { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 6px; font-size: 0.95rem; }
    .cell-head strong { color: var(--ink); }
    .dot { display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 8px; vertical-align: 0; }
    .cell .track { height: 12px; }
    .why { margin: 0; padding-left: 20px; }
    .why li { margin: 0 0 8px; }
    .why b { color: var(--ink); }
    .callout {
      margin: 16px 0;
      padding: 12px 16px;
      border-radius: 12px;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      color: var(--ink);
    }
    .callout small { display: block; margin-top: 4px; color: var(--muted); }
    .table-scroll { overflow-x: auto; }
    table.clean { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    table.clean th {
      position: static;
      background: transparent;
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      cursor: default;
      border-bottom: 2px solid var(--line);
    }
    table.clean td { padding: 10px 8px; vertical-align: top; }
    table.clean td small { display: block; color: var(--muted); }
    .pill { display: inline-block; border-radius: 999px; padding: 1px 8px; font-size: 0.8rem; font-weight: 700; background: var(--soft); }
    .pill.good { background: #dcfce7; color: #166534; }
    .pill.warn { background: #fef3c7; color: #92400e; }
    .pill.low { background: #fee2e2; color: #991b1b; }
    table.design { border-collapse: separate; border-spacing: 8px; width: 100%; }
    table.design th { position: static; background: transparent; cursor: default; border: 0; color: var(--muted); font-size: 0.85rem; text-align: center; }
    table.design th.row { text-align: left; color: var(--ink); font-size: 0.95rem; }
    table.design td { border: 1px solid var(--line); border-radius: 12px; padding: 12px; text-align: center; }
    table.design td b { display: block; color: var(--ink); font-size: 1.9rem; font-weight: 900; line-height: 1.1; }
    table.design td small { display: block; color: var(--muted); font-size: 0.8rem; }
    table.design td.default { background: #eff6ff; border-color: #bfdbfe; }
    table.design td.sparse { background: #fefce8; border-color: #fde68a; }
    table.design .tag { display: inline-block; margin-top: 4px; color: var(--blue-deep); font-size: 0.72rem; font-weight: 900; }
    .details-stack details {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 0 18px;
      margin: 10px 0;
    }
    .details-stack summary {
      cursor: pointer;
      padding: 14px 0;
      color: var(--ink);
      font-weight: 700;
      font-size: 1.02rem;
    }
    .details-stack details[open] { padding-bottom: 14px; }
    .stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0 8px; }
    .stat {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 16px;
      min-width: 130px;
    }
    .stat b { display: block; color: var(--blue-deep); font-size: 1.5rem; font-weight: 900; }
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
    .filters label { font-size: 0.92rem; color: var(--muted); font-weight: 700; }
    .pair {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(0, 1.35fr);
      gap: 24px;
      align-items: start;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px 20px;
      margin: 16px 0;
    }
    .pair-flat { display: grid; grid-template-columns: minmax(240px, 1fr) minmax(0, 1.6fr); gap: 24px; align-items: start; }
    .pair-flat.even { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
    @media (max-width: 900px) {
      .pair, .pair-flat, .pair-flat.even { grid-template-columns: 1fr; }
      .hero h1 { font-size: 2rem; }
      .topbar { position: static; flex-wrap: wrap; padding: 0 12px; }
      .play { order: 2; margin-left: auto; padding: 6px 12px; }
      .tabs { order: 3; flex: 1 0 100%; flex-wrap: nowrap; overflow-x: auto; }
      .tabs button { padding: 10px 12px; font-size: 0.95rem; white-space: nowrap; }
    }
    .pair h3 { margin: 0 0 8px; font-size: 1.1rem; }
    .pair > .chart-box, .pair > .pie-box { position: sticky; top: 72px; }
    .pair .finding { margin: 10px 0 12px; }
    .stat-details summary { cursor: pointer; margin: 8px 0; color: var(--blue-deep); font-weight: 700; }
    .howto { margin: 8px 0 0; padding-left: 18px; color: var(--text); font-size: 0.92rem; }
    .howto li { margin-bottom: 4px; }
    .chart-box { position: relative; width: 100%; height: 380px; }
    .chart-box.short { height: 300px; }
    .chart-box.tall { height: 460px; }
    canvas { display: block; }
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
      padding: 8px 10px 6px;
    }
    .trace-card h4 {
      margin: 0 0 6px;
      font-size: 0.85rem;
      color: var(--muted);
      font-weight: 700;
    }
    .trace-card .chart-box { height: 230px; }
    .dev-row {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
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
      padding: 8px 10px 6px;
    }
    .dev-panel h4 { margin: 0 0 6px; font-size: 0.92rem; color: var(--ink); }
    .dev-panel .chart-box { height: 300px; }
    .dev-model { margin: 18px 0; }
    .zoomable { cursor: zoom-in; }
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
    .chip.on { background: var(--blue); color: #fff; border-color: var(--blue); }
    .mix-wrap { display: grid; grid-template-columns: auto 1fr auto auto; gap: 10px; align-items: center; max-width: 640px; margin: 8px 0 14px; }
    .mix-wrap input[type="range"] { width: 100%; }
    .mix-wrap button, .people-toolbar button {
      border: 0;
      background: var(--blue);
      color: #fff;
      border-radius: 8px;
      padding: 8px 14px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    .expand-chart {
      position: absolute;
      top: 4px;
      right: 4px;
      z-index: 2;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.9);
      color: var(--blue-deep);
      border-radius: 999px;
      padding: 2px 10px;
      font: inherit;
      font-size: 0.78rem;
      font-weight: 700;
      cursor: pointer;
    }
    .expand-chart:hover { background: var(--blue); color: #fff; }
    #lightboxStage {
      position: relative;
      flex: 1;
      min-height: 60vh;
      background: #fff;
      border-radius: 12px;
    }
    #lightboxImage {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
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
    #lightboxFocusBar .chip { color: var(--ink); background: #fff; }
    #lightboxFocusBar .chip.on { background: #38bdf8; border-color: #38bdf8; color: var(--ink); }
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
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    .pie-box {
      position: relative;
      width: min(380px, 100%);
      aspect-ratio: 1 / 1;
      margin: 0 auto;
    }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
    th { position: sticky; top: 0; background: #f1f5f9; cursor: pointer; }
    .table-wrap { max-height: 420px; overflow: auto; margin-top: 8px; }
    .wide { grid-column: 1 / -1; }
    footer {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 18px;
      margin-top: 28px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 0.85rem;
    }
    footer img { width: 28px; height: 28px; border-radius: 6px; }
    select, input[type="search"] {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 8px;
      font: inherit;
      background: #fff;
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
      color: var(--text);
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
    .people-toolbar button:disabled { opacity: 0.4; cursor: default; }
    .people-caption { font-weight: 700; min-width: 16rem; color: var(--ink); }
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
    #rows tr.active { background: #dbeafe; }
  </style>
</head>
<body>
  <nav class="topbar">
    <a class="brand" href="https://sugar-sugar.study" target="_blank" rel="noopener noreferrer" title="Play Sugar-Sugar">
      <img src="__ICON_URI__" alt="" />
      <span>Sugar-Sugar</span>
    </a>
    <div class="tabs" id="tabs">
      <button type="button" data-tab="overview" class="active">Overview</button>
      <button type="button" data-tab="human">Results</button>
      <!-- BEGIN_FULL_ONLY -->
      <button type="button" data-tab="ai">People vs AI</button>
      <button type="button" data-tab="people">Individual players</button>
      <!-- END_FULL_ONLY -->
    </div>
    <a class="play" href="https://sugar-sugar.study" target="_blank" rel="noopener noreferrer">▶ Play the game</a>
  </nav>
  <div class="wrap">
  <header class="hero">
    <div class="eyebrow" id="editionBadge">Live study dashboard</div>
    <h1 id="pageTitle">Sugar-Sugar study</h1>
    <p id="headerLead">Can people predict where glucose goes in the next hour? Players see three hours of real CGM data and draw their forecast. This page shows how the study is going, what we have learned and how many more players we need.</p>
    <div class="cta-row">
      <a class="cta" href="https://sugar-sugar.study" target="_blank" rel="noopener noreferrer">▶ Play Sugar-Sugar</a>
      <span>Predict the next hour in 15–20 minutes. Every new player brings the study closer to an answer.</span>
    </div>
    <p class="byline">A research project by <a href="https://glucosedao.org" target="_blank" rel="noopener noreferrer"><img src="__LOGO_URI__" alt="" />GlucoseDAO</a></p>
  </header>
  <main>
    <div class="stats" id="stats" hidden></div>
    <div class="filters" id="sharedFilters" hidden>
      <label>Group
        <select id="cohortFilter"><option value="all">All groups</option></select>
      </label>
      <label>Players
        <select id="repeatFilter">
          <option value="all">Everyone</option>
          <option value="single">Played once</option>
          <option value="repeat">Played more than once</option>
          <option value="all_formats">Played tasks A, B and C</option>
          <option value="challenge">Chose "challenge the unknown"</option>
          <option value="opposite">Predicted the opposite group's data</option>
        </select>
      </label>
      <label>Player id
        <input id="search" type="search" placeholder="start of the id" />
      </label>
    </div>

    <div id="tab-overview" class="tab-panel active">
      <section class="card">
        <div class="kpis" id="kpis"></div>
        <div class="target-track" id="targetTrack"></div>
        <p class="note" id="identityNote"></p>
      </section>

      <section>
        <h2>What we have learned so far</h2>
        <p class="lead">One card per study question. The error is how far a player's drawn forecast was from the real CGM reading, averaged over the next hour, in mg/dL. <b>Lower is better.</b></p>
        <div class="tone-legend"><span style="--tone:#16a34a">Supported or useful finding</span><span style="--tone:#eab308">Not enough data yet</span><span style="--tone:#dc2626">Against our assumptions</span></div>
        <div class="findings" id="findings"></div>
      </section>

      <section class="card" id="designCard" hidden>
        <h2>Design check: who saw which glucose traces?</h2>
        <p class="lead">By default the game shows players with diabetes diabetic traces and everyone else non-diabetic traces. Only players who choose "challenge the unknown" also see the other kind. A fair comparison needs both groups on the <b>same</b> kind of trace, so the off-default cells (yellow) matter most.</p>
        <div class="pair-flat even">
          <div class="table-scroll"><table class="design" id="designTable"></table></div>
          <div id="designVerdict"></div>
        </div>
      </section>

      <section class="grid-2">
        <div class="card">
          <h2>Recruitment by group</h2>
          <p class="lead">Players who count for the main analysis (at least 6 rounds on public glucose traces), against the quota for each group.</p>
          <div id="cellBars"></div>
          <p class="note" id="cellNote"></p>
        </div>
        <div class="card">
          <h2>Why the target changed</h2>
          <ul class="why" id="whyTarget"></ul>
        </div>
      </section>

      <section class="card">
        <h2>How many players do we need?</h2>
        <div class="pair-flat">
          <div>
            <p class="lead"><b>Statistical power</b> is the chance that the study detects a real difference of <span id="effectLabel">3.5</span> mg/dL between two groups. Studies usually aim for 80%.</p>
            <ul class="howto">
              <li>Each line shows power rising as both groups grow.</li>
              <li>Coloured lines use the spread of errors we actually observe; the grey line is the cautious planning assumption.</li>
              <li>Large dots mark where each question stands today. Unequal groups count like smaller equal groups.</li>
              <li>The dashed line is the 80% goal.</li>
            </ul>
          </div>
          <div class="chart-box"><canvas id="powerCurve" data-title="Power vs players per group"></canvas></div>
        </div>
        <div class="callout" id="reachCallout" hidden></div>
        <div class="table-scroll"><table class="clean" id="powerTable"></table></div>
        <p class="note" id="powerNote"></p>
      </section>

      <!-- BEGIN_FULL_ONLY -->
      <section class="card" id="aiGlanceCard" hidden>
        <h2>People vs algorithms on the same windows</h2>
        <div class="pair-flat">
          <div>
            <p class="lead">Algorithms were given exactly the three hours of data the player saw, then scored on the same hidden hour. Bars show average error, split into <b style="color:#E45756">diabetic</b> and <b style="color:#4C78A8">non-diabetic</b> traces because diabetic traces are harder for everyone; <b>shorter is better</b>.</p>
            <ul class="howto">
              <li><b>Persistence</b>: assumes glucose stays at its last value.</li>
              <li><b>Linear trend</b>: extends the recent slope.</li>
              <li><b>CITRAS-FM / KronLSTM / GluMind</b>: GlucoseDAO's trained forecasting models, when scored.</li>
            </ul>
            <p class="note" id="aiGlanceNote"></p>
          </div>
          <div class="chart-box tall"><canvas id="aiGlance" data-title="People vs algorithms by trace type"></canvas></div>
        </div>
      </section>
      <!-- END_FULL_ONLY -->

      <section class="details-stack">
        <details>
          <summary>How to read this dashboard</summary>
          <div class="prose" id="reading"></div>
        </details>
        <details>
          <summary>About this page and the AI comparison</summary>
          <div class="prose" id="editionNote"></div>
          <!-- BEGIN_FULL_ONLY -->
          <div class="prose" id="h6Overview"></div>
          <div class="prose" id="aiSummary"></div>
          <!-- END_FULL_ONLY -->
        </details>
        <details>
          <summary>Published model benchmarks (GlucoBench)</summary>
          <div class="prose" id="bench"></div>
        </details>
        <details>
          <summary>Data quality checks</summary>
          <div class="prose" id="verify"></div>
        </details>
      </section>
    </div>

    <div id="tab-human" class="tab-panel">
      <section>
        <h2>1. Who took part and how accurate they were</h2>
        <p class="lead">Use the filters above to narrow every chart on this tab. One dot is one player; their error is the average over all their rounds.</p>
        <div class="pair">
          <div>
            <h3>Who is in the study</h3>
            <p class="prose">Each slice is the number of players in one group, by diabetes status and whether they use a continuous glucose monitor (CGM). This is a head-count, not accuracy.</p>
          </div>
          <div class="pie-box"><canvas id="pie" data-title="Players by group"></canvas></div>
        </div>
        <div class="pair">
          <div>
            <h3>How accurate players were</h3>
            <p class="prose">Each dot is one player's average error in mg/dL. <b>Lower is better.</b> The black dash in each column is that group's median (the middle player).</p>
            <p class="note">Error is the average distance between the drawn forecast and the real CGM value over the next hour.</p>
          </div>
          <div class="chart-box"><canvas id="swarm" data-title="Error per player by group"></canvas></div>
        </div>
        <div class="pair">
          <div>
            <h3>Accuracy on each task</h3>
            <p class="prose"><b>A</b>: public glucose traces from other people. <b>B</b>: the player's own uploaded CGM data. <b>C</b>: a mix of both. One dot per player who played that task; the black dash is the median. Lower is better.</p>
          </div>
          <div class="chart-box"><canvas id="formats" data-title="Error per player by task"></canvas></div>
        </div>
      </section>
      <section>
        <h2>2. Main questions: diabetes and CGM use (H1, H2)</h2>
        <p class="lead">Blue dots are errors on public traces (the main measure), green dots on the player's own data. The dashes mark medians. Each question opens with a plain-language verdict card.</p>
        <div class="tone-legend"><span style="--tone:#16a34a">Supported or useful finding</span><span style="--tone:#eab308">Not enough data yet</span><span style="--tone:#dc2626">Against our assumptions</span></div>
        <div class="pair">
          <div class="prose" id="h1Text"></div>
          <div class="chart-box"><canvas id="h1" data-title="H1: diabetes vs no diabetes"></canvas></div>
        </div>
        <div class="pair">
          <div>
            <h3>H1 compared on the same kind of trace</h3>
            <p class="prose">The fair version of H1. Players with diabetes (red) and without (blue) are compared separately on diabetic traces and on non-diabetic traces, because diabetic traces are harder for everyone. Only players with at least 6 rounds of that kind count. Black dashes are medians.</p>
            <p class="note" id="h1MatchedNote"></p>
          </div>
          <div class="chart-box"><canvas id="h1Matched" data-title="H1 on the same kind of trace"></canvas></div>
        </div>
        <div class="pair">
          <div class="prose" id="h2Text"></div>
          <div class="chart-box"><canvas id="h2" data-title="H2: CGM users vs non-users"></canvas></div>
        </div>
      </section>
      <section>
        <h2>3. Experience and own data (H3–H5)</h2>
        <p class="lead">If experience helps, dots should drift down (smaller errors) as the months increase.</p>
        <div class="pair">
          <div class="prose" id="h3Text"></div>
          <div class="chart-box"><canvas id="h3" data-title="H3: months with diabetes vs error"></canvas></div>
        </div>
        <div class="pair">
          <div class="prose" id="h4Text"></div>
          <div class="chart-box"><canvas id="h4" data-title="H4: months using a CGM vs error"></canvas></div>
        </div>
        <div class="pair">
          <div>
            <div class="prose" id="h5Text"></div>
            <p class="note">Chart: each dot is a player. Dots <b>below the dashed line</b> did better on their own data than on public traces. Bigger dots chose "challenge the unknown".</p>
          </div>
          <div class="chart-box"><canvas id="clusters" data-title="H5: public vs own data, same player"></canvas></div>
        </div>
        <!-- BEGIN_FULL_ONLY -->
        <div class="prose" id="h6Text"></div>
        <!-- END_FULL_ONLY -->
      </section>
      <section>
        <h2>4. Predicting the other group's glucose</h2>
        <div class="pair">
          <div>
            <div class="prose" id="oppositeText"></div>
            <p class="note">Chart: dots <b>above the dashed line</b> made larger errors on the other group's traces than on their own group's traces.</p>
          </div>
          <div class="chart-box"><canvas id="opposite" data-title="Same-group vs other-group traces"></canvas></div>
        </div>
      </section>
      <section>
        <h2>5. Players in the current filter</h2>
        <p class="lead">Errors in mg/dL. Click a row to open that player on the Individual players tab.</p>
        <div class="card wide">
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th data-key="study_id">player id</th>
                  <th data-key="cohort_label">group</th>
                  <th data-key="player_trait">trait</th>
                  <th data-key="n_runs">games</th>
                  <th data-key="formats">tasks</th>
                  <th data-key="challenge_unknown">challenge</th>
                  <th data-key="mae_primary">error</th>
                  <th data-key="mae_generic">public</th>
                  <th data-key="mae_own">own</th>
                  <th data-key="mae_same">same group</th>
                  <th data-key="mae_opposite">other group</th>
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
        <h2>People vs AI on the same task</h2>
        <p class="lead" id="aiPrimaryNote"></p>
        <p class="prose">Each algorithm saw exactly the same three hours the player saw and forecast the same hidden hour. Only players with both a human and an AI score on a task appear. Tasks: <b>A</b> public traces, <b>B</b> own data, <b>C</b> mixed.</p>
        <div class="stats" id="aiTaskStats"></div>
        <div class="chip-row" id="aiCohortChips"></div>
        <div class="mix-wrap">
          <span>People</span>
          <input type="range" id="aiMix" min="0" max="100" value="50" step="1" list="aiMixSnaps" aria-label="Show people, AI, or both" />
          <span>AI</span>
          <button type="button" id="aiMixMiddle">Both</button>
        </div>
        <datalist id="aiMixSnaps">
          <option value="0" label="People"></option>
          <option value="50" label="Both"></option>
          <option value="100" label="AI"></option>
        </datalist>
        <p class="note">Slide left to show only people, right to show only the model, middle for both. The group buttons filter the charts below. Click any chart to enlarge it.</p>
        <div class="pair">
          <div>
            <h3>Error on each task</h3>
            <p class="prose">Same player, same task. <b>Blue</b> dots are people, <b>purple</b> dots are the primary model. Lower is better.</p>
          </div>
          <div class="chart-box"><canvas id="aiTaskMae" class="zoomable" data-title="People vs AI error per task"></canvas></div>
        </div>
        <div class="pair">
          <div>
            <h3>Public vs own data for the same player</h3>
            <p class="prose">Circles are people, triangles are the model, coloured by group. A grey line joins a player's own result to the model's result on their rounds. Diamonds are players with only one of the two tasks, placed on the diagonal so they stay visible. <b>Below the diagonal means better on own data.</b></p>
          </div>
          <div class="chart-box"><canvas id="aiCluster" class="zoomable" data-title="Public vs own data, people and AI"></canvas></div>
        </div>
        <div class="prose" id="aiH6Text"></div>
      </section>
      <section>
        <h2>How errors grow over the hidden hour</h2>
        <p class="prose">The black line at 0% is the real CGM value. Each shaded band shows how far forecasts landed above (+) or below (−) it, minute by minute: the band's edges are the most extreme rounds and the thicker line inside is the median. Narrower bands closer to 0% are better. Each model gets its own row of three panels (tasks A, B, C).</p>
        <div class="legend-grid" id="aiDevLegend"></div>
        <div id="aiDevHost"></div>
      </section>
    </div>

    <div id="tab-people" class="tab-panel">
      <section>
        <h2>One player at a time</h2>
        <p class="prose">Every reconstructed round for one player (up to 12). The thick dark line is the real CGM; everything right of the dotted "forecast starts" line was hidden from the player. Blue is the player's drawing; other colours are the algorithms. Use Previous / Next or the arrow keys to move between players.</p>
        <div class="legend-grid" id="lineLegend"></div>
        <div id="missingModel" class="missing-model" hidden></div>
        <div class="people-toolbar">
          <button type="button" id="prevPerson">← Previous</button>
          <div class="people-caption" id="personCaption">No player selected</div>
          <button type="button" id="nextPerson">Next →</button>
        </div>
        <div id="traceEmpty" class="empty" hidden>No reconstructed 3-hour windows for this filter. Own-data rounds need the saved upload; public rounds need the corpus file.</div>
        <div class="trace-grid" id="traceGrid"></div>
        <h3>This player's errors over the hidden hour</h3>
        <p class="prose">0% is the real value. One shaded band per forecaster (the player in blue, then each model): edges are the extreme rounds, the inner line the median.</p>
        <div class="dev-row" id="personDevRow">
          <div class="dev-panel"><h4>Task A · public traces</h4><div class="chart-box"><canvas id="personDev-A" class="zoomable" data-title="Player errors, task A"></canvas></div></div>
          <div class="dev-panel"><h4>Task B · own data</h4><div class="chart-box"><canvas id="personDev-B" class="zoomable" data-title="Player errors, task B"></canvas></div></div>
          <div class="dev-panel"><h4>Task C · mixed</h4><div class="chart-box"><canvas id="personDev-C" class="zoomable" data-title="Player errors, task C"></canvas></div></div>
        </div>
      </section>
    </div>
    <!-- END_FULL_ONLY -->

    <footer>
      <img src="__LOGO_URI__" alt="" />
      <span>Sugar-Sugar is an open research project by <a href="https://glucosedao.org" target="_blank" rel="noopener noreferrer">GlucoseDAO</a>.</span>
      <a href="https://sugar-sugar.study" target="_blank" rel="noopener noreferrer">Play the game</a>
      <a href="https://github.com/GlucoseDAO/sugar-data-processing" target="_blank" rel="noopener noreferrer">Analysis code</a>
      <span id="meta"></span>
    </footer>
  </main>
  </div>
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
      citras: "crossRot",
      citras_w38_2: "cross",
      kronlstm: "triangle",
    };
    const TRACE_ORDER = ["actual", "human", "persistence", "linear", "glumind", "sugar_one", "citras", "citras_w38_2", "kronlstm"];
    const MAX_ROUNDS = 12;
    let personIndex = 0;
    let aiMix = 0.5;
    let lightboxFocus = null;
    const aiCohortOn = new Set(Object.keys(DATA.cohort_labels || {}));

    Chart.defaults.font.family = "Lato, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
    Chart.defaults.font.size = 13;
    Chart.defaults.color = "#334155";
    Chart.defaults.responsive = true;
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.elements.point.radius = 4;
    Chart.defaults.elements.point.hoverRadius = 6;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 8;
    Chart.defaults.plugins.legend.labels.padding = 14;
    Chart.defaults.plugins.tooltip.backgroundColor = "rgba(15, 23, 42, 0.92)";
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.scale.grid.color = "rgba(15, 23, 42, 0.06)";
    const ERROR_AXIS = "Average error per player (mg/dL) · lower is better";
    const mgTooltip = {
      callbacks: {
        label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(1)} mg/dL`,
      },
    };

    document.getElementById("pageTitle").textContent = DATA.title;
    document.getElementById("editionBadge").textContent = DATA.milestone
      ? "Milestone · human participants only"
      : "Live study dashboard · updated " + DATA.generated_at;
    document.getElementById("reading").innerHTML = DATA.reading_guide_html;
    document.getElementById("editionNote").innerHTML = DATA.edition_html;
    document.getElementById("oppositeText").innerHTML = DATA.opposite_html;
    if (DATA.milestone) {
      const lead = document.getElementById("headerLead");
      if (lead) lead.textContent = "Milestone view: human participants only. Overview tracks recruitment and what we have learned; Results has the groups, the study questions and the player table.";
    }
    const aiPrimaryNote = document.getElementById("aiPrimaryNote");
    if (aiPrimaryNote) {
      aiPrimaryNote.textContent =
        DATA.primary_model
          ? ("Primary model: " + modelLabel(DATA.primary_model) + ". Each dot is one player on one task.")
          : "No model scores yet: the game windows could not be reconstructed.";
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
        const filterable = btn.dataset.tab !== "overview";
        document.getElementById("sharedFilters").hidden = !filterable;
        document.getElementById("stats").hidden = !filterable;
      });
    });

    function modelLabel(name) {
      const known = ((DATA.progress || {}).model_labels || {})[name];
      if (known) return known;
      const style = (DATA.line_styles || {})[name];
      const label = style && style.label ? style.label.replace(/^AI /, "") : String(name || "");
      return label.charAt(0).toUpperCase() + label.slice(1);
    }
    function esc(text) {
      return String(text === null || text === undefined ? "" : text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    }
    function pFmt(value) {
      const p = num(value);
      if (p === null) return "not run yet";
      if (p < 0.001) return "p < 0.001";
      return "p = " + (p >= 0.1 ? p.toFixed(2) : p.toFixed(3));
    }
    function powerPill(value) {
      const p = num(value);
      if (p === null) return '<span class="pill">—</span>';
      const cls = p >= 0.8 ? "good" : p >= 0.5 ? "warn" : "low";
      return `<span class="pill ${cls}">${Math.round(p * 100)}%</span>`;
    }
    function wrapLabel(text) {
      const words = String(text).split(" ");
      if (words.length < 3) return text;
      const at = words.findIndex((word, i) => i > 0 && /^with/i.test(word));
      const mid = at > 0 ? at : Math.ceil(words.length / 2);
      return [words.slice(0, mid).join(" "), words.slice(mid).join(" ")];
    }
    function findingCard(f) {
      return `<article class="finding t-${esc(f.tone || "warn")}">
          <div class="q"><b>${esc(f.code)}</b>${esc(f.question)}</div>
          <span class="verdict">${esc(f.verdict_label)}</span>
          <div class="headline">${esc(f.headline)}</div>
          ${f.detail ? `<div class="detail">${esc(f.detail)}</div>` : ""}
        </article>`;
    }
    function medianOf(values) {
      const xs = values.filter((v) => v !== null && Number.isFinite(v)).sort((a, b) => a - b);
      if (!xs.length) return null;
      const m = Math.floor(xs.length / 2);
      return xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2;
    }
    function medianMarks(groups, label) {
      return {
        label: label || "Median",
        data: groups.map((g) => ({ x: g.x, y: medianOf(g.values) })).filter((pt) => pt.y !== null),
        pointStyle: "line",
        pointRadius: 15,
        pointHoverRadius: 17,
        borderColor: "#0f172a",
        backgroundColor: "#0f172a",
        borderWidth: 3,
        order: -1,
      };
    }
    function diagonal(points, label) {
      const values = points.flatMap((pt) => [pt.x, pt.y]).filter((v) => v !== null && Number.isFinite(v));
      if (!values.length) return null;
      const lo = Math.floor(Math.min(...values) - 1);
      const hi = Math.ceil(Math.max(...values) + 1);
      return {
        label: label || "Same error on both",
        data: [{ x: lo, y: lo }, { x: hi, y: hi }],
        showLine: true,
        borderColor: "#64748b",
        backgroundColor: "#64748b",
        borderDash: [6, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 0,
        pointStyle: "line",
      };
    }

    const barValues = {
      id: "barValues",
      afterDatasetsDraw(chart, _args, opts) {
        if (!opts || !opts.enabled) return;
        const ctx = chart.ctx;
        ctx.save();
        ctx.fillStyle = "#0f172a";
        ctx.font = "700 13px Lato, system-ui, sans-serif";
        ctx.textBaseline = "middle";
        chart.data.datasets.forEach((dataset, d) => {
          const meta = chart.getDatasetMeta(d);
          if (meta.hidden) return;
          meta.data.forEach((bar, i) => {
            const value = dataset.data[i];
            if (value === null || value === undefined) return;
            ctx.fillText(Number(value).toFixed(1), bar.x + 6, bar.y);
          });
        });
        ctx.restore();
      },
    };
    Chart.register(barValues);

    function renderOverview() {
      const P = DATA.progress;
      if (!P) return;
      const t = P.targets;
      const power = Object.fromEntries((P.power || []).map((row) => [row.code, row]));
      const h1 = power.H1 || {};
      const h2 = power.H2 || {};
      document.getElementById("kpis").innerHTML = `
        <div class="kpi"><b>${P.n_people}</b><span>${P.linked_identities ? "people played" : "players"}</span><small>with at least one saved game</small></div>
        <div class="kpi"><b>${P.n_eligible}</b><span>count for the analysis</span><small>6 or more rounds on public traces</small></div>
        <div class="kpi"><b>${t.revised}</b><span>needed in total</span><small>revised target (the plan said ${t.original})</small></div>
        <div class="kpi"><b>+${P.remaining_eligible}</b><span>still needed</span><small>${P.remaining_people ? "about " + P.remaining_people + " new players at today's completion rate" : "spread across the four groups"}</small></div>`;
      const max = Math.max(t.revised, P.n_eligible) * 1.03;
      const at = (v) => (100 * v / max).toFixed(2) + "%";
      document.getElementById("targetTrack").innerHTML = `
        <div class="track" role="img" aria-label="${P.n_eligible} of ${t.revised} eligible players">
          <div class="fill" style="width:${at(P.n_eligible)}"></div>
          <div class="mark up" style="left:${at(t.minimum)}"><span>minimum ${t.minimum}</span></div>
          <div class="mark" style="left:${at(t.original)}"><span>original plan ${t.original}</span></div>
          <div class="mark end" style="left:${at(t.revised)}"><span>revised target ${t.revised}</span></div>
        </div>
        <div class="track-labels"><span><b>${P.n_eligible} of ${t.revised}</b> eligible players (${Math.round(100 * P.n_eligible / t.revised)}% of the revised target)</span></div>`;
      document.getElementById("identityNote").textContent = P.linked_identities
        ? "Players are linked across devices with a pseudonymous key, so each person is counted once."
        : "Players are counted by browser. Someone who clears their browser or switches device counts twice, so the true number of people is somewhat lower.";

      document.getElementById("findings").innerHTML = (P.findings || []).map(findingCard).join("");

      document.getElementById("cellBars").innerHTML = (P.cells || []).map((c) => {
        const gap = Math.max(0, c.target - c.now);
        const color = COLORS[c.key] || "#94a3b8";
        return `<div class="cell">
          <div class="cell-head">
            <span><span class="dot" style="background:${color}"></span><strong>${esc(c.label)}</strong></span>
            <span><b>${c.now}</b> of ${c.target}${gap ? ` · <b>+${gap}</b> needed` : " · quota met"}</span>
          </div>
          <div class="track"><div class="fill" style="width:${Math.min(100, 100 * c.now / c.target)}%;background:${color}"></div></div>
        </div>`;
      }).join("");
      const sparse = (P.cells || []).filter((c) => c.now / c.target < 0.35).map((c) => c.label.charAt(0).toLowerCase() + c.label.slice(1));
      document.getElementById("cellNote").textContent =
        "The quotas overlap so the study can compare 160 people with and 160 without diabetes, and 160 CGM users with 160 non-users. " +
        (sparse.length ? `Most needed: ${sparse.join(" and ")}. More players from the largest group cannot separate the effect of CGM use from diabetes.` : "");

      const why = [
        `<li><b>Original plan</b> (study guidebook §8): about ${t.original} players, 100 per group, expecting a 4 mg/dL difference and a spread (standard deviation) of 10 mg/dL between people.</li>`,
        `<li><b>Two main questions</b> share the usual 5% false-alarm risk, so each must reach p &lt; ${P.primary_alpha}. At that bar, 100 per group gives about 72% power instead of 80%.</li>`,
        `<li><b>Revised target:</b> detect a ${P.planned_effect} mg/dL difference, about half of a CGM sensor's own error, with a cautious spread of ${P.planning_sd} mg/dL. That needs ${h1.need_per_group_plan || "—"} per group: <b>${t.revised} players</b> in total.</li>`,
      ];
      if (num(h1.sd) !== null && num(h2.sd) !== null) {
        why.push(`<li><b>What the data show so far:</b> the observed spread is ${h1.sd.toFixed(1)} mg/dL for H1 and ${h2.sd.toFixed(1)} mg/dL for H2, smaller than assumed. If that holds, about ${Math.max(h1.need_per_group, h2.need_per_group)} per group would be enough.</li>`);
      }
      if (h1.n_a && h1.n_b) {
        why.push(`<li><b>Balance matters:</b> today's ${h1.n_a} players with diabetes and ${h1.n_b} without carry the same power as two equal groups of ${Math.round(h1.n_effective)}. Players from the smaller groups add the most.</li>`);
      }
      document.getElementById("whyTarget").innerHTML = why.join("");
      document.getElementById("effectLabel").textContent = P.planned_effect;

      renderDesign(P);
      renderPowerCurve(P);
      renderPowerTable(P, power);
      renderAiGlance();
    }

    function renderDesign(P) {
      const d = P.design;
      const card = document.getElementById("designCard");
      if (!d || !card) return;
      card.hidden = false;
      const groups = [...new Set(d.cells.map((c) => c.group))];
      const traces = [...new Map(d.cells.map((c) => [c.trace, c.trace_label])).entries()];
      const cap = (text) => text.charAt(0).toUpperCase() + text.slice(1);
      document.getElementById("designTable").innerHTML = `
        <thead><tr><th></th>${traces.map(([, label]) => `<th>${esc(cap(label))}</th>`).join("")}</tr></thead>
        <tbody>${groups.map((group) => `<tr><th class="row">${esc(group)}</th>${traces.map(([trace]) => {
          const c = d.cells.find((cell) => cell.group === group && cell.trace === trace);
          const cls = c.is_default ? "default" : c.players_enough < 20 ? "sparse" : "";
          return `<td class="${cls}">
            <b>${c.players_enough}</b><small>players with ${d.min_rounds}+ rounds</small>
            <small>${c.players_any} played any · ${c.rounds} rounds</small>
            ${c.is_default ? '<span class="tag">default assignment</span>' : ""}
          </td>`;
        }).join("")}</tr>`).join("")}</tbody>`;
      const share = d.n_eligible ? Math.round(100 * d.n_eligible_off_default / d.n_eligible) : 0;
      document.getElementById("designVerdict").innerHTML = `
        <p><b>${d.n_saw_both}</b> of ${d.n_players} players saw both kinds of trace${d.n_saw_both_challenge === null ? "" : `, and ${d.n_saw_both_challenge} of them had chosen "challenge the unknown"`}.</p>
        <p>"Challenge the unknown" gives a player 12 rounds, 6 of each kind, so everyone who chooses it and finishes counts for the fair comparison. The off-default cells are small only because few players choose it: ${share < 50 ? "only " : ""}<b>${d.n_eligible_off_default}</b> of ${d.n_eligible} eligible players (${share}%). Everyone else can only be compared on "their own" kind of trace, where the two groups never meet.</p>
        <div class="callout"><b>Suggested change (game interface, not the protocol):</b> give every player a balanced mix, for example alternating diabetic and non-diabetic traces and starting with a random kind. Each player then shows how much harder diabetic traces are for them, and every eligible player counts for the fair comparison instead of ${share}% today.</div>`;
    }

    function renderPowerCurve(P) {
      const curves = P.curves || {};
      const sizes = curves.group_sizes || [];
      if (!sizes.length) return;
      const palette = { H1: COLORS.diabetic_cgm, H2: "#2185d0", plan: "#94a3b8" };
      const datasets = (curves.series || []).map((series) => ({
        label: series.label,
        data: sizes.map((n, i) => ({ x: n, y: 100 * series.values[i] })),
        showLine: true,
        borderColor: palette[series.code] || "#64748b",
        backgroundColor: palette[series.code] || "#64748b",
        borderWidth: series.code === "plan" ? 2 : 3,
        borderDash: series.code === "plan" ? [6, 4] : [],
        pointRadius: 0,
        pointHoverRadius: 0,
        pointStyle: "line",
        tension: 0.25,
      }));
      datasets.push({
        label: "80% goal",
        data: [{ x: 0, y: 80 }, { x: sizes[sizes.length - 1], y: 80 }],
        showLine: true,
        borderColor: "#0f172a",
        backgroundColor: "#0f172a",
        borderDash: [3, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 0,
        pointStyle: "line",
      });
      (curves.now || []).forEach((pt) => {
        datasets.push({
          label: pt.code + " today",
          data: [{ x: pt.n_per_group, y: 100 * pt.power }],
          backgroundColor: palette[pt.code] || "#334155",
          borderColor: "#ffffff",
          borderWidth: 2,
          pointRadius: 9,
          pointHoverRadius: 11,
        });
      });
      upsertChart("powerCurve", "scatter", { datasets }, {
        plugins: {
          legend: { position: "bottom" },
          tooltip: {
            filter: (item) => item.dataset.pointRadius > 0,
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${Math.round(ctx.parsed.y)}% power, like two groups of ${Math.round(ctx.parsed.x)}`,
            },
          },
        },
        scales: {
          x: { type: "linear", min: 0, max: sizes[sizes.length - 1], title: { display: true, text: "Eligible players in each of the two groups" } },
          y: { min: 0, max: 100, title: { display: true, text: "Power: chance to detect a real difference" }, ticks: { callback: (v) => v + "%" } },
        },
      });
    }

    function renderPowerTable(P, power) {
      const titles = {
        H1: "Diabetes vs no diabetes",
        H2: "CGM users vs non-users",
        H3: "Years with diabetes",
        H4: "Years using a CGM",
        H5: "Own vs public data",
      };
      const rows = (P.power || []).map((row) => {
        const sig = row.significant === true ? '<span class="pill good">significant</span>' : row.p_value !== null && row.p_value !== undefined ? '<span class="pill">not significant</span>' : "";
        let now = "", result = "", needed = "";
        if (row.kind === "groups") {
          now = `${row.n_a} + ${row.n_b}<small>${esc(row.labels[0])} + ${esc(row.labels[1])}</small>`;
          result = row.mean_a === null ? "—" : `${row.mean_a.toFixed(1)} vs ${row.mean_b.toFixed(1)} mg/dL<small>${pFmt(row.p_value)} ${sig}</small>`;
          needed = `${row.need_per_group || row.need_per_group_plan} per group<small>${row.need_per_group_plan} with the cautious SD ${P.planning_sd}</small>`;
          if (row.need_if_real_per_group && !row.significant) needed += `<small>if today's ${Math.abs(row.observed).toFixed(1)} mg/dL gap is real: ${row.need_if_real_per_group} per group</small>`;
        } else if (row.kind === "correlation") {
          now = `${row.n_now}<small>with a known duration</small>`;
          result = row.observed === null || row.observed === undefined ? "not run yet" : `r = ${row.observed >= 0 ? "+" : ""}${row.observed.toFixed(2)}<small>${pFmt(row.p_value)} ${sig}</small>`;
          needed = `${row.need_total} people<small>to detect r = 0.25</small>`;
          if (row.need_if_real && !row.significant) needed += `<small>if today's r is real: ${row.need_if_real}</small>`;
        } else {
          now = `${row.n_now}<small>with 6+ rounds of each kind</small>`;
          result = row.mean_own === null || row.mean_own === undefined ? "not run yet" : `own ${row.mean_own.toFixed(1)} vs public ${row.mean_generic.toFixed(1)} mg/dL<small>${pFmt(row.p_value)} ${sig}</small>`;
          needed = `${row.need_operational} people<small>${row.need_total} is the bare minimum</small>`;
        }
        return `<tr>
          <td><b>${row.code}</b> ${esc(titles[row.code] || row.title || "")}</td>
          <td>${now}</td>
          <td>${result}</td>
          <td>${powerPill(row.power_now)}</td>
          <td>${needed}</td>
          <td><b>${row.still_needed ? "+" + row.still_needed : "reached"}</b></td>
        </tr>`;
      }).join("");
      document.getElementById("powerTable").innerHTML = `
        <thead><tr><th>Question</th><th>Players now</th><th>Result today</th><th>Power today</th><th>Players needed</th><th>Still to recruit</th></tr></thead>
        <tbody>${rows}</tbody>`;
      document.getElementById("powerNote").textContent =
        `Power assumes a real difference of ${P.planned_effect} mg/dL for H1–H2 (two-sided, p < ${P.primary_alpha}), a correlation of 0.25 for H3–H4 and a 3 mg/dL difference for H5 (p < 0.05). ` +
        "These are planning estimates from t-tests and standard approximations; they do not cover redesigned or mixed-effects analyses.";

      const reachable = (P.power || []).filter((row) => !row.significant).map((row) => {
        if (row.kind === "groups" && row.need_if_real_per_group) {
          const more = Math.max(0, row.need_if_real_per_group - row.n_a) + Math.max(0, row.need_if_real_per_group - row.n_b);
          return { row, more, need: row.need_if_real_per_group + " per group", effect: `today's ${Math.abs(row.observed).toFixed(1)} mg/dL gap` };
        }
        if (row.kind === "correlation" && row.need_if_real) {
          return { row, more: Math.max(0, row.need_if_real - row.n_now), need: row.need_if_real + " people", effect: `today's correlation (r = ${row.observed.toFixed(2)})` };
        }
        return null;
      }).filter(Boolean).sort((a, b) => a.more - b.more);
      const already = (P.findings || []).filter((f) => ["supported", "opposite", "significant"].includes(f.verdict)).map((f) => f.code);
      const callout = document.getElementById("reachCallout");
      if (reachable.length || already.length) {
        const best = reachable[0];
        callout.hidden = false;
        callout.innerHTML =
          (already.length ? `<b>Already significant:</b> ${already.join(", ")}. ` : "") +
          (best ? `<b>Closest next result:</b> ${best.row.code} (${esc(titles[best.row.code])}). If ${best.effect} is real, about ${best.need} would be enough: <b>${best.more} more</b> than today.` : "") +
          "<small>Optimistic: early effects often shrink as more people join, which is why the revised target uses a fixed 3.5 mg/dL difference instead.</small>";
      }
    }

    function renderAiGlance() {
      const card = document.getElementById("aiGlanceCard");
      const comps = ((DATA.h6 || {}).comparisons || []).filter((c) => num(c.model_mean_mae) !== null);
      if (!card || !comps.length) return;
      card.hidden = false;
      const trained = comps.some((c) => ["glumind", "sugar_one", "citras", "citras_w38_2", "kronlstm"].includes(c.model_name));
      const models = availableModels();
      const classes = [
        { key: "diabetic", label: "Diabetic traces", color: COLORS.diabetic_cgm },
        { key: "nondiabetic", label: "Non-diabetic traces", color: COLORS.nondiabetic_cgm },
      ];
      const roundMae = (trace, values) => {
        const actual = hiddenHour(trace.actual, trace.visible_end);
        const predicted = hiddenHour(values, trace.visible_end);
        let sum = 0;
        let n = 0;
        actual.forEach((a, i) => {
          const p = num((predicted || [])[i]);
          if (num(a) !== null && p !== null) { sum += Math.abs(p - a); n += 1; }
        });
        return n ? sum / n : null;
      };
      // Only rounds every forecaster scored, so each bar covers the same windows.
      const summary = classes.map((cls) => {
        const rounds = (DATA.traces || []).filter((t) => t.data_class === cls.key
          && roundMae(t, t.human) !== null && models.every((m) => roundMae(t, (t.models || {})[m]) !== null));
        const avg = (fn) => rounds.length ? rounds.reduce((acc, t) => acc + fn(t), 0) / rounds.length : null;
        return {
          ...cls,
          rounds: rounds.length,
          players: new Set(rounds.map((t) => t.study_id)).size,
          values: Object.fromEntries([["People", avg((t) => roundMae(t, t.human))]].concat(
            models.map((m) => [modelLabel(m), avg((t) => roundMae(t, t.models[m]))])
          )),
        };
      });
      const labels = ["People"].concat(models.map(modelLabel));
      const overall = (label) => {
        const vals = summary.map((s) => s.values[label]).filter((v) => v !== null);
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : Infinity;
      };
      const ordered = ["People"].concat(labels.slice(1).sort((a, b) => overall(a) - overall(b)));
      document.getElementById("aiGlanceNote").textContent =
        summary.map((s) => `${s.label}: ${s.rounds} rounds from ${s.players} players`).join(" · ") +
        ". Only rounds that every forecaster scored." +
        (trained ? "" : " Only the simple baselines were scored in this build; the trained models are not included yet.");
      const values = summary.flatMap((s) => Object.values(s.values)).filter((v) => v !== null);
      const top = values.length ? Math.max(...values) * 1.2 : 30;
      upsertChart("aiGlance", "bar", {
        labels: ordered,
        datasets: summary.map((s) => ({
          label: `${s.label} (${s.rounds} rounds)`,
          data: ordered.map((label) => s.values[label]),
          backgroundColor: hexAlpha(s.color, 0.85),
          borderRadius: 5,
          maxBarThickness: 18,
        })),
      }, {
        indexAxis: "y",
        plugins: {
          legend: { position: "bottom" },
          barValues: { enabled: true },
          tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.parsed.x).toFixed(1)} mg/dL` } },
        },
        scales: {
          x: { beginAtZero: true, max: Math.ceil(top / 5) * 5, title: { display: true, text: "Average error per round over the hidden hour (mg/dL) · shorter is better" } },
          y: { grid: { display: false }, ticks: { font: { size: 14, weight: "700" } } },
        },
      });
    },
      });
    }

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
      const plain = Object.fromEntries(((DATA.progress || {}).findings || []).map((f) => [f.code, f]));
      Object.entries(DATA.hypotheses_html || {}).forEach(([key, htmlBlock]) => {
        const el = document.getElementById(key + "Text");
        if (!el) return;
        const code = key.toUpperCase();
        const finding = plain[code] || (code === "H3" || code === "H4" ? plain["H3–H4"] : null);
        const body = document.createElement("div");
        body.innerHTML = htmlBlock;
        const heading = body.querySelector("h2, h3, h4");
        const title = heading ? heading.outerHTML : "";
        if (heading) heading.remove();
        el.innerHTML = title + (finding ? findingCard(finding) : "") +
          `<details class="stat-details"><summary>Statistical details</summary>${body.innerHTML}</details>`;
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
        el.innerHTML = `<p>No model scores on reconstructed windows yet. Sequences live in <code>${DATA.sequences_dir || "data/processed/ai"}</code>. Models only see the 24 visible points of the 3-hour game slice.</p>`;
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
      const haveDeep = (DATA.traces || []).some((t) => t.models && (t.models.glumind || t.models.sugar_one || t.models.citras || t.models.citras_w38_2 || t.models.kronlstm));
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
        title.textContent = `Round ${trace.round_number} · task ${trace.format || "?"} · ${trace.source || trace.data_source_name || ""}`;
        const canvas = document.createElement("canvas");
        canvas.id = "trace-" + i;
        canvas.className = "zoomable";
        const box = document.createElement("div");
        box.className = "chart-box";
        box.appendChild(canvas);
        card.appendChild(title);
        card.appendChild(box);
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
            y: { title: { display: true, text: "Glucose (mg/dL)", font: { size: 11 } } },
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
      { key: "A", label: "Task A · public traces" },
      { key: "B", label: "Task B · own data" },
      { key: "C", label: "Task C · mixed" },
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
          title: { display: true, text: "% above (+) or below (−) the real value" },
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
            <span>People · ${DATA.cohort_labels[key]}</span>
          </div>
        `).join("") + cohortKeys.map((key) => `
          <div class="legend-item">
            <span class="swatch solid" style="border-top-color:${AI_COLORS[key] || "#1e293b"}"></span>
            <span>AI · ${DATA.cohort_labels[key]}</span>
          </div>
        `).join("") + `
          <div class="legend-item"><span class="swatch solid" style="border-top-color:#0f172a"></span><span>0% = real CGM value</span></div>
        `;
      }
      destroyChartsByPrefix("aiDev-");
      const modelBlocks = models.length ? models : ["human_only"];
      host.innerHTML = modelBlocks.map((model) => {
        const title = model === "human_only"
          ? "People only (no model scored)"
          : (((DATA.line_styles || {})[model] || {}).label || model);
        return `
          <div class="dev-model">
            <h3>${title}</h3>
            <p class="note">${traces.filter((t) => (t.models || {})[model] || model === "human_only").length} reconstructed rounds in the current filter.</p>
            <div class="dev-row">
              ${DEV_FORMATS.map((fmt) => `
                <div class="dev-panel">
                  <h4>${fmt.label}</h4>
                  <div class="chart-box"><canvas id="aiDev-${model}-${fmt.key}" class="zoomable" data-title="${title} · ${fmt.label}"></canvas></div>
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
            pushErrorBand(datasets, "People · " + cohortLabel, color, [], "human", humanByCohort[cohort]);
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
        <div class="stat"><b>${rows.length}</b><span>players in filter</span></div>
        <div class="stat"><b>${rows.filter((p) => p.repeat).length}</b><span>played more than once</span></div>
        <div class="stat"><b>${rows.filter((p) => p.challenge_unknown).length}</b><span>chose "challenge the unknown"</span></div>
        <div class="stat"><b>${rows.filter((p) => p.opposite_trait).length}</b><span>predicted the other group</span></div>
        <div class="stat"><b>${fmt(mean(rows.map((p) => num(p.mae_primary))))}</b><span>average error, mg/dL</span></div>
        <div class="stat"><b>${betterOwn}/${allFormats.length || 0}</b><span>better on own data (played A, B and C)</span></div>
        <div class="stat"><b>${(DATA.traces || []).length}</b><span>rounds replayed for AI</span></div>
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
      const groupTicks = {
        stepSize: 1,
        autoSkip: false,
        maxRotation: 0,
        callback: (v) => Number.isInteger(v) ? wrapLabel(DATA.cohort_labels[cohortKeys[v]] || "") : "",
      };

      upsertChart(pieId, "doughnut", {
        labels: cohortKeys.map((k) => DATA.cohort_labels[k]),
        datasets: [{
          data: cohortKeys.map((k) => rows.filter((p) => p.cohort === k).length),
          backgroundColor: cohortKeys.map((k) => COLORS[k] || "#94a3b8"),
          borderColor: "#ffffff",
          borderWidth: 2,
        }],
      }, {
        cutout: "50%",
        plugins: {
          legend: { position: "bottom" },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const total = ctx.dataset.data.reduce((a, b) => a + b, 0) || 1;
                return `${ctx.label}: ${ctx.parsed} players (${Math.round(100 * ctx.parsed / total)}%)`;
              },
            },
          },
        },
      });

      const bothRows = rows.filter((p) => num(p.mae_generic) !== null && num(p.mae_own) !== null);
      const clusterSets = cohortKeys.map((k) => {
        const people = bothRows.filter((p) => p.cohort === k);
        return {
          label: DATA.cohort_labels[k],
          data: people.map((p) => ({ x: num(p.mae_generic), y: num(p.mae_own) })),
          backgroundColor: COLORS[k] || "#94a3b8",
          pointRadius: people.map((p) => p.challenge_unknown ? 7 : 5),
        };
      }).filter((ds) => ds.data.length);
      const clusterLine = diagonal(clusterSets.flatMap((ds) => ds.data), "Same error on both");
      upsertChart(clustersId, "scatter", {
        datasets: clusterLine ? clusterSets.concat([clusterLine]) : clusterSets,
      }, {
        plugins: {
          legend: { position: "bottom" },
          tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: public ${Number(ctx.parsed.x).toFixed(1)}, own ${Number(ctx.parsed.y).toFixed(1)} mg/dL` } },
        },
        scales: {
          x: { title: { display: true, text: "Error on public traces (mg/dL)" } },
          y: { title: { display: true, text: "Error on own data (mg/dL)" } },
        },
      });

      upsertChart(swarmId, "scatter", {
        datasets: cohortKeys.map((k, idx) => ({
          label: DATA.cohort_labels[k],
          data: rows.filter((p) => p.cohort === k && num(p.mae_primary) !== null).map((p) => ({
            x: idx + jitter(p.study_id, 0.35),
            y: num(p.mae_primary),
          })),
          backgroundColor: hexAlpha(COLORS[k] || "#94a3b8", 0.8),
        })).concat([medianMarks(cohortKeys.map((k, idx) => ({
          x: idx,
          values: rows.filter((p) => p.cohort === k).map((p) => num(p.mae_primary)),
        })), "Group median")]),
      }, {
        plugins: { legend: { position: "bottom" }, tooltip: mgTooltip },
        scales: {
          x: { min: -0.6, max: cohortKeys.length - 0.4, grid: { display: false }, ticks: groupTicks },
          y: { title: { display: true, text: ERROR_AXIS } },
        },
      });

      upsertChart(h1Id, "scatter", {
        datasets: layeredCategorySwarm(rows, (p) => p.diabetic === true, (p) => p.diabetic === false),
      }, layeredCategoryScales("With diabetes", "Without diabetes"));

      upsertChart(h2Id, "scatter", {
        datasets: layeredCategorySwarm(rows, (p) => p.uses_cgm === true, (p) => p.uses_cgm === false),
      }, layeredCategoryScales("CGM users", "Non-users"));

      upsertChart(h3Id, "scatter", {
        datasets: durationSourceSeries(rows, (p) => p.diabetic === true, (p) => {
          const months = num(p.diabetes_duration_months);
          const years = num(p.diabetes_duration);
          const value = months !== null ? months : (years === null ? null : years * 12);
          const cap = (DATA.duration_caps || {}).diabetes_duration_months;
          return value === null || (cap && value > cap) ? null : value;
        }),
      }, durationScales("Months living with diabetes"));

      upsertChart(h4Id, "scatter", {
        datasets: durationSourceSeries(rows, (p) => p.uses_cgm === true, (p) => {
          const months = num(p.cgm_duration_months);
          const years = num(p.cgm_duration_years);
          const value = months !== null ? months : (years === null ? null : years * 12);
          const cap = (DATA.duration_caps || {}).cgm_duration_months;
          return value === null || (cap && value > cap) ? null : value;
        }),
      }, durationScales("Months using a CGM"));

      const formatCols = [
        { key: "mae_a", label: "A · public traces", color: "#4C78A8" },
        { key: "mae_b", label: "B · own data", color: "#54A24B" },
        { key: "mae_c", label: "C · mixed", color: "#F58518" },
      ];
      upsertChart(formatsId, "scatter", {
        datasets: formatCols.map((f, idx) => ({
          label: f.label,
          data: rows.filter((p) => num(p[f.key]) !== null).map((p) => ({
            x: idx + jitter(p.study_id + f.key, 0.28),
            y: num(p[f.key]),
          })),
          backgroundColor: hexAlpha(f.color, 0.8),
        })).concat([medianMarks(formatCols.map((f, idx) => ({
          x: idx,
          values: rows.map((p) => num(p[f.key])),
        })), "Task median")]),
      }, {
        plugins: { legend: { position: "bottom" }, tooltip: mgTooltip },
        scales: {
          x: {
            min: -0.5,
            max: 2.5,
            grid: { display: false },
            title: { display: true, text: "Task" },
            ticks: {
              stepSize: 1,
              callback: (v) => Number.isInteger(v) ? ((formatCols[v] || {}).label || "") : "",
            },
          },
          y: { title: { display: true, text: ERROR_AXIS } },
        },
      });

      const minRounds = ((DATA.progress || {}).trace_matched || {}).min_rounds || 6;
      const traceCols = [
        { mae: "mae_dia_data", n: "n_dia_data", label: "Diabetic traces" },
        { mae: "mae_nondia_data", n: "n_nondia_data", label: "Non-diabetic traces" },
      ];
      const playerGroups = [
        { label: "Players with diabetes", match: (p) => p.diabetic === true, color: COLORS.diabetic_cgm, offset: -0.17 },
        { label: "Players without diabetes", match: (p) => p.diabetic === false, color: COLORS.nondiabetic_cgm, offset: 0.17 },
      ];
      const qualifies = (p, t) => (p[t.n] || 0) >= minRounds && num(p[t.mae]) !== null;
      upsertChart("h1Matched", "scatter", {
        datasets: playerGroups.map((g) => ({
          label: g.label,
          data: traceCols.flatMap((t, idx) => rows.filter((p) => g.match(p) && qualifies(p, t))
            .map((p) => ({ x: idx + g.offset + jitter(p.study_id + t.mae, 0.16), y: num(p[t.mae]) }))),
          backgroundColor: hexAlpha(g.color, 0.8),
        })).concat([medianMarks(playerGroups.flatMap((g) => traceCols.map((t, idx) => ({
          x: idx + g.offset,
          values: rows.filter((p) => g.match(p) && qualifies(p, t)).map((p) => num(p[t.mae])),
        }))), "Median")]),
      }, {
        plugins: { legend: { position: "bottom" }, tooltip: mgTooltip },
        scales: {
          x: {
            min: -0.5,
            max: 1.5,
            grid: { display: false },
            afterBuildTicks(scale) {
              scale.ticks = [{ value: 0 }, { value: 1 }];
            },
            ticks: {
              font: { size: 14, weight: "700" },
              callback: (v) => (Number(v) === 0 ? traceCols[0].label : Number(v) === 1 ? traceCols[1].label : ""),
            },
          },
          y: { title: { display: true, text: ERROR_AXIS } },
        },
      });
      const matchedNote = document.getElementById("h1MatchedNote");
      if (matchedNote) {
        matchedNote.textContent = traceCols.map((t) => {
          const counts = playerGroups.map((g) => rows.filter((p) => g.match(p) && qualifies(p, t)).length);
          return `${t.label}: ${counts[0]} with diabetes vs ${counts[1]} without`;
        }).join(" · ") + " (current filter).";
      }

      if (document.getElementById(oppositeId)) {
        const both = rows.filter((p) => num(p.mae_same) !== null && num(p.mae_opposite) !== null)
          .map((p) => ({ x: num(p.mae_same), y: num(p.mae_opposite) }));
        const line = diagonal(both, "Same error on both");
        const sets = [{ label: "Players who predicted both", data: both, backgroundColor: hexAlpha(COLORS.diabetic_cgm, 0.85) }];
        upsertChart(oppositeId, "scatter", {
          datasets: line ? sets.concat([line]) : sets,
        }, {
          plugins: {
            legend: { position: "bottom" },
            tooltip: { callbacks: { label: (ctx) => `own group ${Number(ctx.parsed.x).toFixed(1)}, other group ${Number(ctx.parsed.y).toFixed(1)} mg/dL` } },
          },
          scales: {
            x: { title: { display: true, text: "Error on own group's traces (mg/dL)" } },
            y: { title: { display: true, text: "Error on the other group's traces (mg/dL)" } },
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
        { key: "a", label: "Task A · public" },
        { key: "b", label: "Task B · own" },
        { key: "c", label: "Task C · mixed" },
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
          <div class="stat"><b>${pairs.length}</b><span>players in these groups</span></div>
          <div class="stat"><b>${taskPairs[0].length}</b><span>compared on task A</span></div>
          <div class="stat"><b>${taskPairs[1].length}</b><span>compared on task B</span></div>
          <div class="stat"><b>${taskPairs[2].length}</b><span>compared on task C</span></div>
          <div class="stat"><b>${bothSides.length}</b><span>played public and own</span></div>
          <div class="stat"><b>${linked.length}</b><span>people and AI on both</span></div>
          <div class="stat"><b>${humanOwn}/${linked.length || 0}</b><span>people better on own data</span></div>
        `;
      }

      replaceChart("aiTaskMae", "scatter", {
        datasets: tasks.flatMap((task, idx) => {
          const ready = taskPairs[idx];
          return [
            {
              label: "People · " + task.label,
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
          y: { title: { display: true, text: "Average error on that task (mg/dL) · lower is better" } },
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
          label: "People · " + label,
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
      const clusterLine = diagonal(datasets.filter((ds) => ds._who !== "link").flatMap((ds) => ds.data), "Same error on both");
      if (clusterLine) datasets.push(clusterLine);
      replaceChart("aiCluster", "scatter", { datasets }, {
        plugins: { legend: { position: "bottom", labels: { filter: legendFilter } } },
        scales: {
          x: { title: { display: true, text: "Error on public traces, task A (mg/dL)" } },
          y: { title: { display: true, text: "Error on own data, task B (mg/dL)" } },
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
      const host = src.canvas.closest(".pair, .dev-panel, .trace-card, .card");
      const heading = host ? host.querySelector("h3, h4") : null;
      document.getElementById("lightboxTitle").textContent =
        src.canvas.dataset.title || (heading && heading.textContent) || sourceId;
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
        try {
          stageCanvas.hidden = false;
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
                barValues: { enabled: !!(src.options && src.options.plugins && src.options.plugins.barValues && src.options.plugins.barValues.enabled) },
                zeroPctLine: {
                  enabled: !!(src.options && src.options.plugins && src.options.plugins.zeroPctLine && src.options.plugins.zeroPctLine.enabled),
                },
                forecastGate: src.options && src.options.plugins ? src.options.plugins.forecastGate : undefined,
              },
              indexAxis: src.options && src.options.indexAxis,
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

    const SOURCE_SERIES = [
      { key: "mae_generic", label: "Public traces", color: "#4C78A8", offset: -0.17 },
      { key: "mae_own", label: "Own CGM data", color: "#54A24B", offset: 0.17 },
    ];
    function layeredCategorySwarm(rows, yesFn, noFn) {
      const sides = [yesFn, noFn];
      const sets = SOURCE_SERIES.map((src) => ({
        label: src.label,
        data: sides.flatMap((matches, idx) => rows
          .filter((p) => matches(p) && num(p[src.key]) !== null)
          .map((p) => ({ x: idx + src.offset + jitter(p.study_id + src.key, 0.16), y: num(p[src.key]) }))),
        backgroundColor: hexAlpha(src.color, 0.8),
      }));
      const medians = SOURCE_SERIES.flatMap((src) => sides.map((matches, idx) => ({
        x: idx + src.offset,
        values: rows.filter(matches).map((p) => num(p[src.key])),
      })));
      return sets.concat([medianMarks(medians, "Median")]);
    }
    function layeredCategoryScales(yesLabel, noLabel) {
      return {
        plugins: { legend: { position: "bottom" }, tooltip: mgTooltip },
        scales: {
          x: {
            min: -0.5,
            max: 1.5,
            grid: { display: false },
            afterBuildTicks(scale) {
              scale.ticks = [{ value: 0 }, { value: 1 }];
            },
            ticks: {
              font: { size: 14, weight: "700" },
              callback: (v) => (Number(v) === 0 ? yesLabel : Number(v) === 1 ? noLabel : ""),
            },
          },
          y: { title: { display: true, text: ERROR_AXIS } },
        },
      };
    }
    function durationSourceSeries(rows, matchFn, xFn) {
      return SOURCE_SERIES.map((src) => ({
        label: src.label,
        data: rows.filter((p) => matchFn(p) && xFn(p) !== null && num(p[src.key]) !== null)
          .map((p) => ({ x: xFn(p), y: num(p[src.key]) })),
        backgroundColor: hexAlpha(src.color, 0.8),
      }));
    }
    function durationScales(xLabel) {
      return {
        plugins: {
          legend: { position: "bottom" },
          tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${Math.round(ctx.parsed.x)} months, ${Number(ctx.parsed.y).toFixed(1)} mg/dL` } },
        },
        scales: {
          x: { beginAtZero: true, title: { display: true, text: xLabel } },
          y: { title: { display: true, text: ERROR_AXIS } },
        },
      };
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
      `Updated ${DATA.generated_at} · ${DATA.n_runs} saved games · ${DATA.n_players} players · ${String(DATA.source_csv).split("/").pop()}`;
    renderHypotheses();
    renderBench();
    renderVerify();
    renderAiSummary();
    renderLegend();
    renderOverview();
    render();
  </script>
</body>
</html>
"""
