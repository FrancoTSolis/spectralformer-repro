#!/usr/bin/env python3
"""Orchestrate exact-revision, checkpoint-only SEA evaluations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from manifest_lib import (
    ManifestError,
    get_run,
    load_manifest,
    verify_hashes,
    verify_paths,
    verify_revision,
)


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "run_manifest.json"
DEFAULT_RESULTS = HERE / "results"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run held-out inference only. This program has no training mode and "
            "refuses to overwrite an output directory."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used for the isolated repository worker",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--source-mode",
        choices=("revision", "working-tree"),
        default="revision",
        help="revision archives the exact recorded commit without changing the repository",
    )
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-runs", action="store_true")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def tracked_worktree_is_clean(repository: Path) -> bool:
    commands = (
        ["git", "-C", str(repository), "diff", "--quiet"],
        ["git", "-C", str(repository), "diff", "--cached", "--quiet"],
    )
    return all(subprocess.run(command, check=False).returncode == 0 for command in commands)


def head_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def materialize_revision(repository: Path, revision: str, source_root: Path) -> None:
    source_root.mkdir(parents=True, exist_ok=False)
    process = subprocess.Popen(
        ["git", "-C", str(repository), "archive", "--format=tar", revision],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            if member.name == "configs" or member.name.startswith("configs/"):
                # Runtime configuration is pinned in the reviewed manifest. Historical
                # config modules are omitted because they contain authentication data.
                continue
            destination = (source_root / member.name).resolve()
            if source_root not in destination.parents and destination != source_root:
                process.kill()
                raise RuntimeError(f"unsafe path in git archive: {member.name}")
            if member.issym() or member.islnk():
                continue
            archive.extract(member, path=source_root)
    _, stderr = process.communicate()
    if process.returncode:
        raise RuntimeError(
            f"git archive failed for {revision}: {stderr.decode(errors='replace').strip()}"
        )


def compare_fingerprints(
    before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]
) -> None:
    if before != after:
        changed = sorted(set(before) | set(after))
        details = [
            path for path in changed if before.get(path) != after.get(path)
        ]
        raise RuntimeError(
            "protected checkpoint/data artifact changed during evaluation: "
            + ", ".join(details)
        )


def blocker_payload(stage: str, exc: BaseException, log_path: Path | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "provenance": "evaluation_blocker_not_a_metric_result",
        "stage": stage,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if log_path is not None and log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        payload["evaluation_log"] = str(log_path)
        payload["log_tail"] = text[-8000:]
    return payload


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.list_runs:
        for run in manifest["runs"]:
            if run["evaluation"]["supported"] and not run["evaluation"].get(
                "ready", True
            ):
                state = "blocked"
            else:
                state = (
                    "supported"
                    if run["evaluation"]["supported"]
                    else "historical-only"
                )
            print(f"{run['id']}\t{state}\t{run['paper_usability']['status']}")
        return 0
    if not args.run_id:
        raise ManifestError("--run-id is required unless --list-runs is used")
    if args.batch_size <= 0 or args.bootstrap_samples <= 0:
        raise ManifestError("batch size and bootstrap sample count must be positive")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise ManifestError(f"worker Python is not executable: {args.python}")

    run = get_run(manifest, args.run_id)
    if not run["evaluation"]["supported"]:
        raise ManifestError(
            f"{args.run_id} is historical-only; use aggregate_historical.py"
        )
    if not run["evaluation"].get("ready", True):
        blockers = "; ".join(run["evaluation"].get("known_blockers", []))
        raise ManifestError(f"{args.run_id} is blocked: {blockers}")
    runtime_source = run["evaluation"].get("runtime_source", run["code"])
    repository = Path(runtime_source["repository"]).resolve()
    runtime_revision = runtime_source["revision"]
    if args.repo_root is not None and args.repo_root.resolve() != repository:
        raise ManifestError(
            f"--repo-root {args.repo_root.resolve()} does not match manifest {repository}"
        )

    verify_revision(repository, runtime_revision)
    before = verify_paths(run)
    verified_hashes: Dict[str, str] = {}
    if args.verify_hashes:
        verified_hashes = verify_hashes(run)

    plan = {
        "mode": "checkpoint_only_test",
        "run_id": run["id"],
        "repository": str(repository),
        "checkpoint_recorded_revision": run["code"]["revision"],
        "runtime_source_revision": runtime_revision,
        "source_revision": runtime_revision,
        "source_mode": args.source_mode,
        "device": args.device,
        "worker_python": str(args.python.resolve()),
        "batch_size": args.batch_size,
        "bootstrap_samples": args.bootstrap_samples,
        "test_trajectory_indices": run["split"]["test_indices"],
        "predicted_steps": run["horizon"]["predicted_steps"],
        "protected_artifacts": before,
        "verified_hashes": verified_hashes,
        "unsafe_settings_forced_off": [
            "training",
            "gradient_calculation",
            "optimizer_steps",
            "checkpoint_saves",
            "historical save_dir writes",
            "W&B initialization",
            "initial plotting/tests",
            "time-shift augmentation",
        ],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (DEFAULT_RESULTS / f"{run['id']}__{timestamp()}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(output_dir / "run_spec.json", run)
    write_json_exclusive(output_dir / "execution_plan.json", plan)
    log_path = output_dir / "evaluation.log"
    stage = "source_preparation"

    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{run['id']}__source__", dir=str(output_dir.parent)
        ) as temporary:
            if args.source_mode == "revision":
                source_root = Path(temporary) / "source"
                materialize_revision(repository, runtime_revision, source_root)
            else:
                if head_revision(repository) != runtime_revision:
                    raise RuntimeError(
                        "working-tree HEAD does not match the recorded run revision"
                    )
                if not tracked_worktree_is_clean(repository):
                    raise RuntimeError(
                        "working-tree has tracked modifications; use --source-mode revision"
                    )
                source_root = repository

            stage = "checkpoint_evaluation"
            environment = os.environ.copy()
            environment.pop("WANDB_API_KEY", None)
            environment["WANDB_MODE"] = "disabled"
            environment["WANDB_DISABLED"] = "true"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONUNBUFFERED"] = "1"
            environment["MPLBACKEND"] = "Agg"
            environment["PYTHONPATH"] = str(source_root)
            command = [
                str(args.python.resolve()),
                str(HERE / "evaluator_worker.py"),
                "--run-spec",
                str(output_dir / "run_spec.json"),
                "--source-root",
                str(source_root),
                "--output-dir",
                str(output_dir),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--bootstrap-samples",
                str(args.bootstrap_samples),
            ]
            with log_path.open("x", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=source_root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"isolated evaluator exited with status {completed.returncode}"
                )

        stage = "artifact_integrity_check"
        after = verify_paths(run)
        compare_fingerprints(before, after)
        write_json_exclusive(
            output_dir / "execution.json",
            {
                "schema_version": 1,
                "status": "completed",
                "mode": "checkpoint_only_test",
                "checkpoint_recorded_revision": run["code"]["revision"],
                "runtime_source_revision": runtime_revision,
                "source_revision": runtime_revision,
                "source_mode": args.source_mode,
                "protected_artifacts_unchanged": True,
                "verified_hashes": verified_hashes,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(output_dir)
        return 0
    except BaseException as exc:
        write_json_exclusive(
            output_dir / "blocker.json", blocker_payload(stage, exc, log_path)
        )
        print(
            f"evaluation blocked; details: {output_dir / 'blocker.json'}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestError, OSError, subprocess.SubprocessError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)

