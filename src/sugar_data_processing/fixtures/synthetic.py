"""Create a synthetic ``prediction_statistics.csv`` that matches current sugar-sugar."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import polars as pl


def write_synthetic_csv(
    path: Path | str,
    *,
    n_participants: int = 120,
    seed: int = 7,
) -> Path:
    """Write a sugar-sugar-compatible statistics CSV with planted effects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[object, object]] = []

    for i in range(n_participants):
        study_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"sugar-fixture-{seed}-{i}"))
        diabetic = bool(i % 2 == 0)
        uses_cgm = bool(i % 3 != 0)
        age = float(rng.integers(18, 75))
        diabetes_duration = float(rng.uniform(0.5, 25.0)) if diabetic else 0.0
        cgm_duration = float(rng.uniform(0.2, 12.0)) if uses_cgm else 0.0
        replay = i % 10 == 0  # some people replay format A

        # Planted effects: PwD / CGM / longer experience → lower MAE
        base = 22.0
        base -= 4.0 if diabetic else 0.0
        base -= 3.0 if uses_cgm else 0.0
        base -= 0.15 * diabetes_duration
        base -= 0.4 * cgm_duration
        if i in {0, 1}:
            cgm_duration = age + 5.0
        if i == 2:
            base = 95.0

        generic_mae = float(np.clip(base + rng.normal(0, 4.0), 5.0, 120.0))
        own_mae = float(np.clip(generic_mae - rng.uniform(1.0, 5.0), 3.0, 120.0))
        mixed_mae = float(np.clip((generic_mae + own_mae) / 2.0 + rng.normal(0, 1.5), 3.0, 120.0))

        n_generic = 8
        n_own = 8 if i % 4 != 3 else 3
        home_source = "D1NAMO-001.csv" if diabetic else "BIGIDEAS-001.csv"
        opposite_source = "BIGIDEAS-001.csv" if diabetic else "D1NAMO-001.csv"
        # Challenge the unknown: opt-in opposite-corpus mix on A/C (T1 and non-PwD)
        challenge = i % 5 == 0

        rows.append(
            _run_row(
                study_id=study_id,
                format_code="A",
                is_example=True,
                diabetic=diabetic,
                uses_cgm=uses_cgm,
                age=age,
                diabetes_duration=diabetes_duration,
                cgm_duration=cgm_duration,
                n_rounds=n_generic,
                mean_mae=generic_mae,
                rng=rng,
                number=0,
                source_name=home_source,
                timestamp_day=1,
                challenge_unknown=challenge,
                opposite_source=opposite_source,
            )
        )
        if replay:
            rows.append(
                _run_row(
                    study_id=study_id,
                    format_code="A",
                    is_example=True,
                    diabetic=diabetic,
                    uses_cgm=uses_cgm,
                    age=age,
                    diabetes_duration=diabetes_duration,
                    cgm_duration=cgm_duration,
                    n_rounds=n_generic,
                    mean_mae=float(np.clip(generic_mae - 1.0, 3.0, 120.0)),
                    rng=rng,
                    number=3,
                    source_name=home_source,
                    timestamp_day=10,
                    challenge_unknown=challenge,
                    opposite_source=opposite_source,
                )
            )
        if n_own >= 6 or i % 2 == 0:
            rows.append(
                _run_row(
                    study_id=study_id,
                    format_code="B",
                    is_example=False,
                    diabetic=diabetic,
                    uses_cgm=uses_cgm,
                    age=age,
                    diabetes_duration=diabetes_duration,
                    cgm_duration=cgm_duration,
                    n_rounds=max(n_own, 6) if i % 4 != 3 else n_own,
                    mean_mae=own_mae,
                    rng=rng,
                    number=1,
                    source_name="clarity_export.csv",
                    timestamp_day=2,
                )
            )
        if i % 3 == 0:
            rows.append(
                _run_row(
                    study_id=study_id,
                    format_code="C",
                    is_example=False,
                    diabetic=diabetic,
                    uses_cgm=uses_cgm,
                    age=age,
                    diabetes_duration=diabetes_duration,
                    cgm_duration=cgm_duration,
                    n_rounds=8,
                    mean_mae=mixed_mae,
                    rng=rng,
                    number=2,
                    source_name=home_source,
                    timestamp_day=3,
                    mixed=True,
                    generic_source=home_source,
                    challenge_unknown=challenge,
                    opposite_source=opposite_source,
                )
            )

    df = pl.DataFrame(rows)
    df.write_csv(path)
    return path


def _run_row(
    *,
    study_id: str,
    format_code: str,
    is_example: bool,
    diabetic: bool,
    uses_cgm: bool,
    age: float,
    diabetes_duration: float,
    cgm_duration: float,
    n_rounds: int,
    mean_mae: float,
    rng: np.random.Generator,
    number: int,
    source_name: str,
    timestamp_day: int,
    mixed: bool = False,
    generic_source: str = "example.csv",
    challenge_unknown: bool = False,
    opposite_source: str = "BIGIDEAS-001.csv",
    n_points: int = 6,
) -> dict[object, object]:
    round_maes = np.clip(rng.normal(mean_mae, 3.0, size=n_rounds), 1.0, 150.0)
    per_round: list[dict[object, object]] = []
    round_context: list[dict[object, object]] = []
    predicted_values: list[dict[object, object]] = []
    real_values: list[dict[object, object]] = []
    prediction_times: list[dict[object, object]] = []
    for r in range(n_rounds):
        round_number = r + 1
        if mixed:
            round_is_example = round_number % 2 == 1
            round_source = generic_source if round_is_example else "clarity_export.csv"
        else:
            round_is_example = is_example
            round_source = source_name
        if challenge_unknown and round_is_example and round_number % 2 == 0:
            round_source = opposite_source
        window_index = r * 12
        hour = 8 + r
        window_start = f"2026-01-01 {hour:02d}:00:00"
        pred_start = f"2026-01-01 {hour:02d}:30:00"
        window_end = f"2026-01-01 {hour:02d}:55:00"
        per_round.append(
            {
                "round_number": round_number,
                "mae": float(round_maes[r]),
                "mse": float(round_maes[r] ** 2),
                "rmse": float(round_maes[r] * 1.2),
                "mape": float(round_maes[r] / 1.5),
                "data_source_name": round_source,
                "is_example_data": round_is_example,
                "generic_slice_key": f"slice-{round_number}" if round_is_example else "",
            }
        )
        round_context.append(
            {
                "round_number": round_number,
                "format": format_code,
                "data_source_name": round_source,
                "is_example_data": round_is_example,
                "generic_slice_key": f"slice-{round_number}" if round_is_example else "",
                "prediction_window_start_index": window_index,
                "prediction_window_size": n_points,
                "window_start_time": window_start,
                "prediction_start_time": pred_start,
                "window_end_time": window_end,
            }
        )
        baseline = 140.0 if "D1NAMO" in round_source else 95.0
        for point_i in range(n_points):
            real = float(np.clip(baseline + rng.normal(0, 8.0), 50.0, 280.0))
            pred = float(np.clip(real + rng.normal(0, max(round_maes[r] / 2.0, 1.0)), 40.0, 300.0))
            minute = point_i * 5
            stamp = f"2026-01-01 {hour:02d}:{minute:02d}:00"
            predicted_values.append(
                {"version": format_code, "round": round_number, "value": f"{pred:.1f}"}
            )
            real_values.append(
                {"version": format_code, "round": round_number, "value": f"{real:.1f}"}
            )
            prediction_times.append(
                {"version": format_code, "round": round_number, "value": stamp}
            )
    overall_mae = float(np.mean(round_maes))
    cgm_cell = f"{cgm_duration:g},years" if uses_cgm else ""
    return {
        "study_id": study_id,
        "run_id": str(uuid.uuid4()),
        "number": number,
        "timestamp": f"2026-06-{timestamp_day:02d} 12:00:00",
        "email": f"participant_{study_id[:8]}@example.com",
        "format": format_code,
        "is_example_data": is_example,
        "data_source_name": source_name,
        "age": age,
        "user_id": abs(hash(study_id)) % 10_000,
        "gender": "F" if number % 2 == 0 else "M",
        "uses_cgm": uses_cgm,
        "cgm_duration_years": cgm_cell,
        "diabetic": diabetic,
        "diabetic_type": "Type 1" if diabetic else "",
        "diabetes_duration": diabetes_duration if diabetic else "",
        "location": "Test City",
        "rounds_played": n_rounds,
        "predicted_values": str(predicted_values),
        "real_values": str(real_values),
        "prediction_times": str(prediction_times),
        "overall_mae_mgdl": overall_mae,
        "overall_mse_mgdl": float(np.mean(round_maes**2)),
        "overall_rmse_mgdl": float(np.sqrt(np.mean(round_maes**2))),
        "overall_mape_pct": float(overall_mae / 1.5),
        "per_round_metrics": str(per_round),
        "generic_intervention": (
            "mix:bigideas=0.50,d1namo=0.50"
            if challenge_unknown
            else ("d1namo" if diabetic else "bigideas")
        ),
        "challenge_unknown": challenge_unknown,
        "challenge_unknown_pct": 50 if challenge_unknown else "",
        "paper_mention": False,
        "paper_full_name": "",
        "round_context": str(round_context),
    }
