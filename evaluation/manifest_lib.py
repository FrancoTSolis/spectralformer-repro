"""Strict loading and validation for the paper run manifest."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


SCHEMA_VERSION = 1
ALLOWED_REPOSITORY_NAMES = {"SEA", "SEA-baseline"}
FORBIDDEN_KEY = re.compile(
    r"(api[_-]?key|password|passwd|secret|access[_-]?token|auth[_-]?token|credential)",
    re.IGNORECASE,
)
REVISION = re.compile(r"^[0-9a-f]{40}$")


class ManifestError(ValueError):
    """Raised when evidence metadata is missing, ambiguous, or unsafe."""


def _walk(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if FORBIDDEN_KEY.search(str(key)):
                raise ManifestError(f"forbidden secret-bearing key at {child_path}")
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def load_manifest(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot load manifest {path}: {exc}") from exc
    for _ in _walk(manifest):
        pass
    manifest = _resolve_references(manifest)
    validate_manifest(manifest)
    return manifest


def _resolve_references(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = deepcopy(dict(manifest))
    tables = {
        "data_ref": ("data_assets", "data"),
        "split_ref": ("splits", "split"),
        "config_ref": ("config_profiles", "config_overrides"),
        "conditioning_ref": ("conditioning_recipes", "conditioning_recipe"),
    }
    for run in resolved.get("runs", []):
        for reference_key, (table_key, destination_key) in tables.items():
            reference = run.get(reference_key)
            if reference is None:
                continue
            table = resolved.get(table_key, {})
            if reference not in table:
                raise ManifestError(
                    f"{run.get('id', '<unknown>')}: unknown {reference_key} {reference}"
                )
            if destination_key in {"config_overrides", "conditioning_recipe"}:
                run.setdefault("evaluation", {})[destination_key] = deepcopy(
                    table[reference]
                )
            else:
                run[destination_key] = deepcopy(table[reference])
    return resolved


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"schema_version must be {SCHEMA_VERSION}, got {manifest.get('schema_version')}"
        )
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ManifestError("manifest.runs must be a non-empty list")

    seen = set()
    for run in runs:
        if not isinstance(run, Mapping):
            raise ManifestError("every run entry must be an object")
        run_id = run.get("id")
        if not isinstance(run_id, str) or not run_id:
            raise ManifestError("every run requires a non-empty id")
        if run_id in seen:
            raise ManifestError(f"duplicate run id: {run_id}")
        seen.add(run_id)

        revision = run.get("code", {}).get("revision")
        if not isinstance(revision, str) or not REVISION.fullmatch(revision):
            raise ManifestError(f"{run_id}: code.revision must be a full 40-hex commit")
        repository = Path(run.get("code", {}).get("repository", ""))
        if not repository.is_absolute():
            raise ManifestError(f"{run_id}: code.repository must be absolute")
        if repository.name not in ALLOWED_REPOSITORY_NAMES:
            raise ManifestError(
                f"{run_id}: repository must be SEA or SEA-baseline, got {repository}"
            )

        split = run.get("split", {})
        if split.get("seed") != 42 or split.get("unit") != "trajectory":
            raise ManifestError(f"{run_id}: split must be trajectory-level seed 42")
        for split_name in ("train_indices", "validation_indices", "test_indices"):
            if not isinstance(split.get(split_name), list):
                raise ManifestError(f"{run_id}: missing split.{split_name}")
        all_indices = (
            split["train_indices"]
            + split["validation_indices"]
            + split["test_indices"]
        )
        if len(all_indices) != len(set(all_indices)):
            raise ManifestError(f"{run_id}: trajectory split contains duplicates")

        horizon = run.get("horizon", {}).get("predicted_steps")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ManifestError(f"{run_id}: horizon.predicted_steps must be positive")

        evaluation = run.get("evaluation", {})
        if evaluation.get("supported"):
            runtime_source = evaluation.get("runtime_source")
            if runtime_source is not None:
                runtime_repository = Path(runtime_source.get("repository", ""))
                runtime_revision = runtime_source.get("revision")
                if (
                    not runtime_repository.is_absolute()
                    or runtime_repository.name not in ALLOWED_REPOSITORY_NAMES
                ):
                    raise ManifestError(f"{run_id}: invalid evaluation runtime repository")
                if (
                    not isinstance(runtime_revision, str)
                    or not REVISION.fullmatch(runtime_revision)
                ):
                    raise ManifestError(f"{run_id}: invalid evaluation runtime revision")
            for artifact_name in ("temporal_checkpoint", "spatial_checkpoint"):
                artifact = run.get("artifacts", {}).get(artifact_name)
                _validate_artifact(run_id, artifact_name, artifact)
            data = run.get("data", {})
            for data_name in ("field_data", "input_data", "coordinates"):
                _validate_artifact(run_id, f"data.{data_name}", data.get(data_name))
            if len(split["test_indices"]) == 0:
                raise ManifestError(f"{run_id}: supported evaluation has no test indices")
            field_names = evaluation.get("field_names")
            if not isinstance(field_names, list) or not field_names:
                raise ManifestError(f"{run_id}: supported evaluation needs field names")
            required_config = {
                "dimension",
                "field_groups",
                "m",
                "n",
                "k",
                "MLP_hidden_spatial",
                "num_layers_spatial",
                "embed_dim_spatial",
                "n_heads_spatial",
                "block_size_spatial",
                "dropout_spatial",
                "variational_spatial",
                "src_len_spatial",
                "spatial_batch_size",
                "batch_size",
                "num_layers",
                "embed_dim",
                "n_heads",
                "block_size",
                "scale_ratio",
                "src_len",
                "num_fields",
                "down_proj",
                "dropout",
                "exchange_mode",
                "pos_encoding_mode",
                "ib_scale_mode",
                "ib_addition_mode",
                "ib_mlp_layers",
                "ib_num",
                "add_info_after_cross",
                "LN_type",
                "SEA_isolate",
                "SEA_mixed",
            }
            config = evaluation.get("config_overrides", {})
            missing = sorted(required_config - set(config))
            if missing:
                raise ManifestError(
                    f"{run_id}: safe config profile is missing {', '.join(missing)}"
                )
            recipe = evaluation.get("conditioning_recipe")
            if recipe is not None:
                if recipe.get("output_features") != evaluation.get(
                    "required_input_features"
                ):
                    raise ManifestError(
                        f"{run_id}: conditioning recipe/checkpoint feature mismatch"
                    )
                cited_revisions = {
                    source.get("revision") for source in recipe.get("sources", [])
                }
                if revision not in cited_revisions:
                    raise ManifestError(
                        f"{run_id}: conditioning recipe does not cite {revision}"
                    )


def _validate_artifact(run_id: str, name: str, artifact: Any) -> None:
    if not isinstance(artifact, Mapping):
        raise ManifestError(f"{run_id}: missing artifact {name}")
    path = Path(artifact.get("path", ""))
    if not path.is_absolute():
        raise ManifestError(f"{run_id}: {name}.path must be absolute")
    sha256 = artifact.get("sha256")
    if sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", str(sha256)):
        raise ManifestError(f"{run_id}: invalid {name}.sha256")


def get_run(manifest: Mapping[str, Any], run_id: str) -> Dict[str, Any]:
    for run in manifest["runs"]:
        if run["id"] == run_id:
            return dict(run)
    raise ManifestError(f"unknown run id: {run_id}")


def artifact_paths(run: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for artifact in run.get("artifacts", {}).values():
        if isinstance(artifact, Mapping) and artifact.get("path"):
            paths.append(Path(artifact["path"]))
    for artifact in run.get("data", {}).values():
        if isinstance(artifact, Mapping) and artifact.get("path"):
            paths.append(Path(artifact["path"]))
    return paths


def verify_paths(run: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    repository = Path(run["code"]["repository"])
    if not repository.is_dir():
        raise ManifestError(f"repository does not exist: {repository}")
    for path in artifact_paths(run):
        if not path.is_file():
            raise ManifestError(f"required read-only artifact does not exist: {path}")
        stat = path.stat()
        report[str(path)] = {
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return report


def verify_revision(repository: Path, revision: str) -> None:
    command = [
        "git",
        "-C",
        str(repository),
        "cat-file",
        "-e",
        f"{revision}^{{commit}}",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ManifestError(
            f"revision {revision} is unavailable in {repository}: {detail}"
        )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(run: Mapping[str, Any]) -> Dict[str, str]:
    verified: Dict[str, str] = {}
    records = list(run.get("artifacts", {}).values()) + list(run.get("data", {}).values())
    for artifact in records:
        if not isinstance(artifact, Mapping) or not artifact.get("path"):
            continue
        expected = artifact.get("sha256")
        if expected is None:
            continue
        path = Path(artifact["path"])
        observed = sha256_file(path)
        if observed != expected:
            raise ManifestError(
                f"SHA-256 mismatch for {path}: expected {expected}, got {observed}"
            )
        verified[str(path)] = observed
    return verified

