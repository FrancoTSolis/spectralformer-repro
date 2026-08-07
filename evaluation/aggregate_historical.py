#!/usr/bin/env python3
"""Export clearly labeled fallback evidence from historical logs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from manifest_lib import get_run, load_manifest


HERE = Path(__file__).resolve().parent


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate manifest-pinned historical evidence. Output is explicitly "
            "not a new checkpoint evaluation."
        )
    )
    parser.add_argument("--manifest", type=Path, default=HERE / "run_manifest.json")
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def rows_for_run(run: Mapping[str, Any]) -> Iterable[list[Any]]:
    usability = run["paper_usability"]
    for evidence in run.get("historical_evidence", []):
        source = evidence.get("source", {})
        for metric in evidence.get("metrics", []):
            yield [
                run["id"],
                run["role"],
                run["case"],
                evidence["scope"],
                evidence.get("split", ""),
                run["horizon"]["predicted_steps"],
                metric["field"],
                metric["name"],
                metric["value"],
                source.get("path", ""),
                json.dumps(source.get("lines", [])),
                usability["status"],
                " | ".join(evidence.get("limitations", [])),
            ]


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.run_id:
        runs = [get_run(manifest, run_id) for run_id in args.run_id]
    else:
        runs = manifest["runs"]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (HERE / "results" / f"historical_evidence__{timestamp()}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "provenance": {
            "kind": "historical_log_aggregation_not_new_evaluation",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(args.manifest.resolve()),
            "warning": (
                "Values retain their historical split, minibatch, validation/test, "
                "and aggregation limitations. They must not be presented as rerun metrics."
            ),
        },
        "runs": [
            {
                "id": run["id"],
                "role": run["role"],
                "case": run["case"],
                "code_revision": run["code"]["revision"],
                "wandb": run["wandb"],
                "horizon": run["horizon"],
                "historical_evidence": run.get("historical_evidence", []),
                "paper_usability": run["paper_usability"],
            }
            for run in runs
        ],
    }
    with (output_dir / "historical_evidence.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    with (output_dir / "historical_evidence.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "role",
                "case",
                "scope",
                "split",
                "predicted_steps",
                "field",
                "metric",
                "value",
                "source_path",
                "source_lines",
                "paper_status",
                "limitations",
            ]
        )
        for run in runs:
            writer.writerows(rows_for_run(run))

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

