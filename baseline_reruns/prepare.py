"""Materialize immutable split, graph, and normalization artifacts."""

from pathlib import Path

import numpy as np

from common import (
    DATASETS,
    edge_features,
    field_and_delta_stats,
    load_conditioning,
    load_coordinates,
    load_edges,
    trajectory_split,
    write_split_audit,
)


HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    for name, spec in DATASETS.items():
        print(f"[{name}] loading fields", flush=True)
        fields = np.load(spec.fields_path, mmap_mode="r")
        train, validation, test = trajectory_split(
            fields.shape[0], spec.train_fraction, spec.val_fraction
        )
        write_split_audit(CACHE / f"{name}_split.json", spec, train, validation, test)

        coordinates = load_coordinates(spec)
        edges = load_edges(spec, coordinates)
        features = edge_features(coordinates, edges)
        np.savez_compressed(
            CACHE / f"{name}_graph.npz",
            coordinates=coordinates,
            edge_index=edges,
            edge_features=features,
        )

        conditioning = load_conditioning(spec, fields.shape[1])
        np.save(CACHE / f"{name}_conditioning.npy", conditioning)

        print(f"[{name}] computing train-only statistics", flush=True)
        stats = field_and_delta_stats(fields, train)
        np.savez(CACHE / f"{name}_stats.npz", **stats)
        print(
            f"[{name}] split={len(train)}/{len(validation)}/{len(test)} "
            f"nodes={fields.shape[2]} fields={fields.shape[3]} edges={edges.shape[1]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
