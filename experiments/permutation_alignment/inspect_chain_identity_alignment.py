"""Inspect token-index identity alignment on one production PhysPPI sample."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from experiments.permutation_alignment.audit_protected_chain_matching import (
    _DEFAULT_REFERENCE_MOLECULES,
    ProductionPaths,
    _build_production_match_inputs,
    cleanup_production_cache,
)
from openfold3.core.utils.permutation_alignment import (
    get_token_identity_gt_token_subset_index,
    single_batch_multi_chain_permutation_alignment,
)
from openfold3.core.utils.protected_chain_matching import ProtectedChainMatchError
from openfold3.projects.of3_all_atom.physppi.screening.models import DatasetName

TOKEN_LABELS = ("token_index", "asym_id", "entity_id", "sym_id")


def _valid_token_mask(features: dict) -> torch.Tensor:
    return features["token_mask"].bool()


def _valid_atom_mask(features: dict) -> torch.Tensor:
    return features["atom_mask"].bool()


def _token_feature(features: dict, name: str) -> torch.Tensor | None:
    if name not in features:
        return None
    return features[name][_valid_token_mask(features)]


def _tensor_int_list(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().tolist()]


def _mismatch_count(
    pred_values: torch.Tensor | None,
    gt_values: torch.Tensor | None,
    gt_token_subset_index: torch.Tensor | None,
) -> int | None:
    if pred_values is None or gt_values is None or gt_token_subset_index is None:
        return None
    matched_gt_values = gt_values[gt_token_subset_index]
    return int((pred_values.detach().cpu() != matched_gt_values.detach().cpu()).sum())


def _token_rows(
    pred_features: dict,
    gt_features: dict,
    gt_token_subset_index: torch.Tensor | None,
    max_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_count = int(_valid_token_mask(pred_features).sum().item())
    for row in range(min(row_count, max_rows)):
        gt_row = (
            int(gt_token_subset_index[row].item())
            if gt_token_subset_index is not None
            else None
        )
        payload: dict[str, Any] = {"pred_row": row, "gt_row": gt_row}
        for label in TOKEN_LABELS:
            pred_values = _token_feature(pred_features, label)
            gt_values = _token_feature(gt_features, label)
            payload[f"pred_{label}"] = (
                int(pred_values[row].item()) if pred_values is not None else None
            )
            payload[f"gt_{label}"] = (
                int(gt_values[gt_row].item())
                if gt_values is not None and gt_row is not None
                else None
            )
        rows.append(payload)
    return rows


def inspect_token_identity_features(
    *,
    dataset: DatasetName,
    sample_id: str,
    pred_features: dict,
    predicted_atom_positions: torch.Tensor,
    max_token_rows: int,
) -> dict[str, Any]:
    """Create a JSON-serializable token identity alignment report."""
    gt_features = pred_features["ground_truth"]
    gt_token_subset_index: torch.Tensor | None = None
    selector_error: str | None = None
    try:
        gt_token_subset_index = get_token_identity_gt_token_subset_index(
            gt_token_index=_token_feature(gt_features, "token_index"),
            pred_token_index=_token_feature(pred_features, "token_index"),
            gt_asym_id=_token_feature(gt_features, "asym_id"),
            pred_asym_id=_token_feature(pred_features, "asym_id"),
        )
    except ProtectedChainMatchError as exc:
        selector_error = str(exc)

    mismatch_counts = {
        label: _mismatch_count(
            _token_feature(pred_features, label),
            _token_feature(gt_features, label),
            gt_token_subset_index,
        )
        for label in TOKEN_LABELS
    }

    try:
        atom_index = single_batch_multi_chain_permutation_alignment(
            pred_features,
            predicted_atom_positions,
            chain_assignment="token_identity",
        )
        alignment: dict[str, Any] = {
            "status": "accepted",
            "reason": pred_features["physppi_chain_match_diagnostic"]["reason"],
            "diagnostics": pred_features["physppi_chain_match_diagnostic"],
            "atom_index_count": int(atom_index.numel()),
            "atom_index_is_identity": bool(
                torch.equal(
                    atom_index.detach().cpu(),
                    torch.arange(atom_index.numel(), dtype=torch.long),
                )
            ),
        }
    except ProtectedChainMatchError as exc:
        alignment = {
            "status": "failed",
            "reason": str(exc),
            "diagnostics": None,
            "atom_index_count": 0,
            "atom_index_is_identity": False,
        }

    return {
        "dataset": dataset.value,
        "sample_id": sample_id,
        "pred_feature_keys": sorted(
            str(key) for key in pred_features if key != "ground_truth"
        ),
        "gt_feature_keys": sorted(str(key) for key in gt_features),
        "chain_label_keys": {
            "pred": {label: label in pred_features for label in TOKEN_LABELS},
            "gt": {label: label in gt_features for label in TOKEN_LABELS},
        },
        "counts": {
            "pred_tokens": int(_valid_token_mask(pred_features).sum().item()),
            "gt_tokens": int(_valid_token_mask(gt_features).sum().item()),
            "pred_atoms": int(_valid_atom_mask(pred_features).sum().item()),
            "gt_atoms": int(_valid_atom_mask(gt_features).sum().item()),
        },
        "selector_error": selector_error,
        "mismatch_counts": mismatch_counts,
        "token_rows": _token_rows(
            pred_features,
            gt_features,
            gt_token_subset_index,
            max_token_rows,
        ),
        "token_identity_alignment": alignment,
    }


def run_production_inspection(
    *,
    paths: ProductionPaths,
    dataset: DatasetName,
    sample_id: str,
    max_token_rows: int,
) -> dict[str, Any]:
    pred_features, _gt_features, atom_positions, _atom_mask = (
        _build_production_match_inputs(paths, dataset, sample_id)
    )
    return inspect_token_identity_features(
        dataset=dataset,
        sample_id=sample_id,
        pred_features=pred_features,
        predicted_atom_positions=atom_positions,
        max_token_rows=max_token_rows,
    )


def write_json_atomic(payload: Mapping[str, Any], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"inspection output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect token-index identity alignment on one PhysPPI sample."
    )
    parser.add_argument(
        "--dataset",
        choices=[item.value for item in DatasetName],
        required=True,
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--qbiolip-root", type=Path, required=True)
    parser.add_argument("--afdb-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--components-cif", type=Path, required=True)
    parser.add_argument(
        "--reference-molecules",
        type=Path,
        default=_DEFAULT_REFERENCE_MOLECULES,
    )
    parser.add_argument("--max-token-rows", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    paths = ProductionPaths(
        qbiolip_root=args.qbiolip_root,
        afdb_root=args.afdb_root,
        formal_root=args.formal_root,
        components_cif=args.components_cif,
        reference_molecules=args.reference_molecules,
    )
    try:
        payload = run_production_inspection(
            paths=paths,
            dataset=DatasetName(args.dataset),
            sample_id=args.sample_id,
            max_token_rows=args.max_token_rows,
        )
        write_json_atomic(payload, args.output)
    finally:
        cleanup_production_cache(paths)


if __name__ == "__main__":
    main()
