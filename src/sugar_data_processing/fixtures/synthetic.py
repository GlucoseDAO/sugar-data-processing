"""Create a synthetic ``prediction_statistics.csv`` that exercises H1–H5."""

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

        # Planted effects: PwD / CGM / longer experience → lower MAE
        base = 22.0
        base -= 4.0 if diabetic else 0.0
        base -= 3.0 if uses_cgm else 0.0
        base -= 0.15 * diabetes_duration
        base -= 0.4 * cgm_duration
        # A few intentional anomalies
        if i in {0, 1}:
            cgm_duration = age + 5.0
        if i == 2:
            base = 95.0

        generic_mae = float(np.clip(base + rng.normal(0, 4.0), 5.0, 120.0))
        own_mae = float(np.clip(generic_mae - rng.uniform(1.0, 5.0), 3.0, 120.0))

        n_generic = 8
        n_own = 8 if i % 4 != 3 else 3  # most people eligible for H5
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
) -> dict[object, object]:
    round_maes = np.clip(rng.normal(mean_mae, 3.0, size=n_rounds), 1.0, 150.0)
    per_round = [
        {
            "round_number": r + 1,
            "mae": float(round_maes[r]),
            "mse": float(round_maes[r] ** 2),
            "rmse": float(round_maes[r] * 1.2),
            "mape": float(round_maes[r] / 1.5),
        }
        for r in range(n_rounds)
    ]
    overall_mae = float(np.mean(round_maes))
    return {
        "study_id": study_id,
        "run_id": str(uuid.uuid4()),
        "number": number,
        "timestamp": f"2026-06-{(number % 28) + 1:02d} 12:00:00",
        "email": f"participant_{study_id[:8]}@example.com",
        "format": format_code,
        "is_example_data": is_example,
        "data_source_name": "example.csv" if is_example else "upload.csv",
        "age": age,
        "user_id": abs(hash(study_id)) % 10_000,
        "gender": "F" if number % 2 == 0 else "M",
        "uses_cgm": uses_cgm,
        "cgm_duration_years": cgm_duration if uses_cgm else None,
        "diabetic": diabetic,
        "diabetic_type": "Type 1" if diabetic else "",
        "diabetes_duration": diabetes_duration,
        "location": "Test City",
        "rounds_played": n_rounds,
        "predicted_values": "[]",
        "real_values": "[]",
        "prediction_times": "[]",
        "overall_mae_mgdl": overall_mae,
        "overall_mse_mgdl": float(np.mean(round_maes**2)),
        "overall_rmse_mgdl": float(np.sqrt(np.mean(round_maes**2))),
        "overall_mape_pct": float(overall_mae / 1.5),
        "per_round_metrics": str(per_round),
    }
