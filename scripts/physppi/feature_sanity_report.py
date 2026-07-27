# ruff: noqa: E402
"""Report canonical PhysPPI coverage or a deterministic feature snapshot.

The snapshot mode intentionally stops at the strict reader boundary while
reusing the production occurrence and optional-prior sampling route.  It does
not construct an OF3 dataset, which may require shared reference/CCD caches.
"""

from __future__ import annotations

if __package__:
    from scripts.physppi._bootstrap import ensure_repository_root
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _bootstrap import ensure_repository_root

ensure_repository_root()

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path

from openfold3.projects.of3_all_atom.physppi.data.priors.augmentation import (
    PRIOR_OPTIONAL_SLOTS,
    OptionalSlot,
    available_optional_slots,
    sample_prior_level,
)
from openfold3.projects.of3_all_atom.physppi.data.priors.occurrence import (
    build_occurrence_rng,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.afdb10k import AFDB10kReader
from openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k import (
    QBioLiP10kReader,
)
from openfold3.projects.of3_all_atom.physppi.model.feature_injection import (
    AUDIT_ONLY_PHYS_PPI_KEYS,
    ESM_FORBIDDEN_KEYS,
    LEGACY_PHYS_PPI_MODEL_INPUT_KEYS,
    PHYS_PPI_KEYS,
)

_SNAPSHOT_DATASET_ENV = {"qbiolip": "PHYSPPI_QBIOLIP_ROOT"}
_FORBIDDEN_MODEL_INPUT_KEYS = (
    ESM_FORBIDDEN_KEYS
    | LEGACY_PHYS_PPI_MODEL_INPUT_KEYS
    | AUDIT_ONLY_PHYS_PPI_KEYS
    | frozenset({"pEnergy", "Delta-G", "experimentally_resolved"})
)


def _reader(root: Path, limit: int):
    name = root.name.lower()
    reader_type = AFDB10kReader if "afdb" in name else QBioLiP10kReader
    return reader_type(root, max_samples=limit)


def _sample_summary(sample) -> dict:
    return {
        "sample_id": sample.sample_id,
        "chains": sorted(sample.chain_sequences or sample.paths.monomer_pdbs),
        "main_msa_chains": sorted(sample.paths.main_msa_npzs),
        "paired_msa_keys": sorted(sample.paths.paired_msa_npzs),
        "query_only_main_msa_chains": list(sample.paths.main_msa_query_only_chains),
        "dmap_edge_count": sum(len(edges) for edges in sample.dmap.values()),
        "interaction_count": len(sample.interactions),
        "mesh_point_count": sum(len(rows) for rows in sample.mesh.values()),
        "has_tm_metadata": bool(sample.opm.get("res_loc"))
        if isinstance(sample.opm, dict)
        else False,
    }


def build_report(root: Path, limit: int) -> dict:
    reader = _reader(root, limit)
    samples = [
        _sample_summary(reader.read(sample_id)) for sample_id in reader.sample_ids()
    ]
    return {
        "root": str(root),
        "reader": type(reader).__name__,
        "sample_count": len(samples),
        "samples": samples,
        "coverage": {
            "main_msa": sum(bool(row["main_msa_chains"]) for row in samples),
            "paired_msa": sum(bool(row["paired_msa_keys"]) for row in samples),
            "dmap": sum(row["dmap_edge_count"] > 0 for row in samples),
            "interaction": sum(row["interaction_count"] > 0 for row in samples),
            "mesh": sum(row["mesh_point_count"] > 0 for row in samples),
            "tm": sum(row["has_tm_metadata"] for row in samples),
        },
    }


def _slot_states(selection) -> dict[str, dict[str, bool | str]]:
    return {
        slot.value: {
            "available": state.available,
            "selected": state.selected,
            "valid": state.valid,
            "active": state.active,
            "provenance": state.provenance,
        }
        for slot in PRIOR_OPTIONAL_SLOTS
        if (state := selection.states[slot])
    }


def _snapshot_body_sha256(report: dict) -> str:
    """Hash the report body before its checksum field, not the emitted file."""
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    return sha256(rendered.encode()).hexdigest()


def build_feature_snapshot(
    sample,
    *,
    dataset: str,
    epoch: int,
    global_slots: list[int],
    seed: int,
) -> dict:
    """Summarize two production occurrence draws without building OF3 features."""
    if len(global_slots) != 2:
        raise ValueError("a feature snapshot requires exactly two global slots")
    if len(set(global_slots)) != 2:
        raise ValueError("feature snapshot global slots must be distinct")

    available = available_optional_slots(sample)
    occurrences = []
    for global_slot in global_slots:
        occurrence_rng = build_occurrence_rng(
            seed, epoch, global_slot, dataset, sample.sample_id
        )
        selection = sample_prior_level(available, occurrence_rng.slots)
        # Match production: consume the occurrence contact draw before filtering.
        cutoff = float(occurrence_rng.contact.uniform(8.0, 18.0))
        occurrences.append(
            {
                "sample_id": sample.sample_id,
                "global_slot": global_slot,
                "occurrence_seed": occurrence_rng.seed,
                "requested_prior_level": selection.requested_level,
                "effective_prior_level": selection.effective_level,
                "optional_slots": _slot_states(selection),
                "contact_cutoff": (
                    cutoff if selection.states[OptionalSlot.CONTACT].active else None
                ),
            }
        )

    report = {
        "dataset": dataset,
        "sample_id": sample.sample_id,
        "epoch": epoch,
        "global_slots": global_slots,
        "seed": seed,
        "model_input_keys": sorted(PHYS_PPI_KEYS),
        "forbidden_model_input_keys": sorted(
            PHYS_PPI_KEYS.intersection(_FORBIDDEN_MODEL_INPUT_KEYS)
        ),
        "occurrences": occurrences,
    }
    report["snapshot_body_sha256"] = _snapshot_body_sha256(report)
    return report


def _snapshot_dataset_root(dataset: str) -> Path:
    environment_name = _SNAPSHOT_DATASET_ENV[dataset]
    return Path(
        os.environ.get(
            environment_name,
            Path.home() / "PPI/Prior_out/training/Q-BioLiP-10k-new",
        )
    )


def _snapshot_reader(dataset: str):
    return QBioLiP10kReader(_snapshot_dataset_root(dataset))


def _render(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _write_or_print(report: dict, output: Path | None) -> None:
    rendered = _render(report)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--dataset", choices=sorted(_SNAPSHOT_DATASET_ENV))
    parser.add_argument("--sample-id")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--global-slots", type=int, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.root is not None:
        if any(
            value is not None
            for value in (
                args.dataset,
                args.sample_id,
                args.epoch,
                args.global_slots,
                args.seed,
            )
        ):
            parser.error("--root coverage mode cannot use snapshot arguments")
        if args.limit < 0:
            parser.error("--limit must be non-negative")
        _write_or_print(build_report(args.root, args.limit), args.output)
        return
    if None in (args.dataset, args.sample_id, args.epoch, args.global_slots, args.seed):
        parser.error(
            "snapshot mode requires --dataset, --sample-id, --epoch, "
            "--global-slots, and --seed"
        )
    if args.output is None:
        parser.error("snapshot mode requires --output")

    reader = _snapshot_reader(args.dataset)
    _write_or_print(
        build_feature_snapshot(
            reader.read(args.sample_id),
            dataset=args.dataset,
            epoch=args.epoch,
            global_slots=args.global_slots,
            seed=args.seed,
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
