"""Audit and aggregate only completed directly comparable reruns."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import DATASETS


HERE = Path(__file__).resolve().parent
METHODS = ("mgn", "mgn-ni", "gmr-gmus", "pbgmr-gmus", "vit-sea")
EXPECTED_AGGREGATION = (
    "reduce spatial nodes first; arithmetic mean over every held-out "
    "trajectory and forecast step; macro is the arithmetic mean over fields"
)
RESULT_FILES = {
    ("gmr-gmus", "multiphase"): "gmr-gmus_multiphase_finetuned.json",
    ("gmr-gmus", "e1"): "gmr-gmus_e1_step4000.json",
    ("pbgmr-gmus", "cylinder"): (
        "pbgmr-gmus_cylinder_sample20-validation-selection.json"
    ),
    ("pbgmr-gmus", "multiphase"): (
        "pbgmr-gmus_multiphase_sample20-validation-selection.json"
    ),
    ("pbgmr-gmus", "we1"): "pbgmr-gmus_we1_sample20-validation-selection.json",
    ("pbgmr-gmus", "e1"): "pbgmr-gmus_e1_sample20-validation-selection.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    missing = []
    for dataset, spec in DATASETS.items():
        for method in METHODS:
            filename = RESULT_FILES.get(
                (method, dataset), f"{method}_{dataset}.json"
            )
            path = HERE / "results" / filename
            if not path.exists():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text())
            if payload["dataset"] != dataset or payload["method"] != method:
                raise ValueError(f"{path}: method/dataset identity mismatch")
            expected_count = {
                "cylinder": 20,
                "multiphase": 8,
                "we1": 127,
                "e1": 127,
            }[dataset]
            if payload["test_trajectory_count"] != expected_count:
                raise ValueError(f"{path}: wrong test trajectory count")
            if payload["forecast_horizon"] != spec.horizon:
                raise ValueError(f"{path}: wrong forecast horizon")
            if payload.get("aggregation") != EXPECTED_AGGREGATION:
                raise ValueError(f"{path}: non-canonical aggregation")
            macro = float(payload["macro_relmse"])
            fields = [float(value) for value in payload["per_field_relmse"]]
            if (
                not math.isfinite(macro)
                or macro < 0
                or any(not math.isfinite(value) or value < 0 for value in fields)
            ):
                raise ValueError(f"{path}: invalid metric")
            if abs(macro - sum(fields) / len(fields)) > 1e-8:
                raise ValueError(f"{path}: macro is not fieldwise arithmetic mean")
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "steps": spec.horizon,
                    "relmse": macro,
                    "per_field_relmse": fields,
                    "source": str(path),
                }
            )

    output = {
        "protocol": (
            "trajectory-level seed-42 split; validation-selected checkpoints; "
            "decoded spatial RelMSE averaged uniformly over trajectories and steps"
        ),
        "complete": not missing,
        "rows": rows,
        "missing": missing,
    }
    output_path = HERE / "results" / "directly_comparable_summary.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"complete": output["complete"], "rows": len(rows)}))
    if args.require_complete and missing:
        raise SystemExit(f"{len(missing)} reruns are still missing")


if __name__ == "__main__":
    main()
