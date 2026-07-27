#!/usr/bin/env python3
"""Run immutable PhysPPI data audits once for a manifest and code revision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openfold3.projects.of3_all_atom.physppi.data.priors.schema import (
    parse_prior_payload,
    sample_field,
)


def audit_key(manifest_bytes: bytes, code_head: str) -> str:
    digest = hashlib.sha256()
    digest.update(manifest_bytes)
    digest.update(b"\0")
    digest.update(code_head.encode())
    return digest.hexdigest()


def run_audit_once(
    report_path: Path,
    manifest_bytes: bytes,
    code_head: str,
    audit_fn: Callable[[], dict[str, object]],
) -> tuple[dict[str, object], bool]:
    key = audit_key(manifest_bytes, code_head)
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing.get("audit_key") == key and existing.get("complete") is True:
            return existing, True

    report: dict[str, object] = {
        "audit_key": key,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "code_head": code_head,
        "complete": True,
        "results": audit_fn(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    return report, False


def _load_sibling_module(name: str) -> Any:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load audit module {name!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_train_ids(manifest: dict[str, object], dataset: str) -> list[str]:
    dataset_manifest = manifest["datasets"][dataset]
    train_ids = list(dataset_manifest["train_ids"])
    fatal_ids = list(dataset_manifest["fatal_ids"])
    if len(set(train_ids)) != len(train_ids):
        raise ValueError(f"manifest has duplicate {dataset} train IDs")
    if set(train_ids) & set(fatal_ids):
        raise ValueError(f"manifest overlaps {dataset} train and fatal IDs")
    return train_ids


def _require_selected_found(name: str, report: dict[str, object]) -> None:
    if report["selected"] != report["found"] or report["missing_ids"]:
        raise ValueError(
            f"{name} audit did not resolve every manifest-selected ID: "
            f"selected={report['selected']}, found={report['found']}"
        )


def _cache_evidence(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"reader index cache is missing: {path}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reused": True,
    }


def _require_reusable_index_cache(dataset: str, root: Path, cache: Path) -> None:
    """Reject a stale cache instead of silently rebuilding a full reader index."""

    if not cache.is_file():
        raise FileNotFoundError(f"reader index cache is missing: {cache}")
    try:
        with cache.open("rb") as handle:
            payload = pickle.load(handle)
    except (EOFError, OSError, pickle.UnpicklingError) as exc:
        raise ValueError(f"reader index cache is unreadable: {cache}") from exc
    expected_metadata = {
        "reader": f"{dataset}10k",
        "root": str(root.resolve()),
        "max_samples": None,
    }
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or payload.get("metadata") != expected_metadata
        or "index" not in payload
    ):
        raise ValueError(
            "reader index cache is stale for "
            f"{dataset}; refusing a repeated scan: {cache}"
        )


def _empty_canonical_report(selected_count: int) -> dict[str, Any]:
    return {
        "selected": selected_count,
        "source_scanned": selected_count,
        "allowed": {"opm_none": [], "mesh_charge_nan": []},
        "warnings": {"main_msa_query_only": []},
        "fatal": {"ids": [], "reasons": {}},
    }


def _scan_gt_pdb(path: Path) -> tuple[dict[str, set[int]], list[dict[str, object]]]:
    """Read a GT PDB once for both canonical labels and malformed charges."""

    if not path.is_file():
        raise ValueError("missing_gt")
    residues: dict[str, set[int]] = {}
    invalid_charges: list[dict[str, object]] = []
    with path.open(errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            raw_charge = line.rstrip("\r\n")[78:80]
            if raw_charge != "  ":
                normalized_charge = (
                    raw_charge
                    if raw_charge and raw_charge[0] in "+-"
                    else raw_charge[::-1]
                )
                try:
                    int(normalized_charge)
                except ValueError:
                    invalid_charges.append(
                        {
                            "line": line_number,
                            "raw_charge": raw_charge,
                            "normalized_charge": normalized_charge,
                        }
                    )
            if line[12:16].strip() != "CA":
                continue
            chain_id = line[21].strip()
            if not chain_id:
                continue
            try:
                resseq = int(line[22:26].strip())
                coordinates = tuple(
                    float(line[start : start + 8]) for start in (30, 38, 46)
                )
            except ValueError as exc:
                raise ValueError("invalid_gt_ca_record") from exc
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError("invalid_gt_ca_coordinates")
            residues.setdefault(chain_id, set()).add(resseq)
    return residues, invalid_charges


def _template_paths_by_sample(
    root: Path, sample_ids: set[str]
) -> dict[str, list[Path]]:
    """Visit each Q-BioLiP template file once and group selected samples."""

    templates = {sample_id: [] for sample_id in sample_ids}
    for chunk in sorted((root / "chunk").glob("chunk*")):
        for template_path in sorted((chunk / "bb_input").glob("*.pdb")):
            sample_id = template_path.stem.rsplit("_", 1)[0]
            if sample_id in templates:
                templates[sample_id].append(template_path)
    return templates


def audit_dataset_once(
    dataset: str,
    root: Path,
    sample_ids: list[str],
    index_cache: Path,
) -> dict[str, object]:
    """Audit one selected source in a single sample pass without re-indexing it."""

    dataset = dataset.lower()
    canonical = _load_sibling_module("physppi_audit_canonical_priors")
    template_atom_ids = _load_sibling_module("physppi_audit_template_atom_ids")
    if dataset not in canonical._DATASET_READERS:
        raise ValueError(f"unknown dataset: {dataset}")
    if not root.is_dir():
        raise FileNotFoundError(f"missing {dataset} root: {root}")

    _require_reusable_index_cache(dataset, root, index_cache)
    reader = canonical._DATASET_READERS[dataset](root, index_cache_file=index_cache)
    selected = set(sample_ids)
    found = selected & set(reader.sample_ids())
    missing_ids = sorted(selected - found)
    canonical_report = _empty_canonical_report(len(sample_ids))
    malformed_report: dict[str, object] = {
        "selected": len(sample_ids),
        "found": len(found),
        "missing_ids": missing_ids,
        "invalid_charge_fields": [],
    }
    template_report: dict[str, object] | None = None
    template_paths: dict[str, list[Path]] = {}
    if dataset == "qbiolip":
        template_paths = _template_paths_by_sample(root, selected)
        template_report = {
            "selected": len(sample_ids),
            "found": len(found),
            "missing_ids": missing_ids,
            "skipped_sample_ids": [],
            "invalid_atom_ids": [],
        }

    dataset_name = "Q-BioLiP" if dataset == "qbiolip" else "AFDB"
    chain_sequences = getattr(reader, "_chain_sequences", {})
    for sample_id in sorted(found):
        paths = reader.locate(sample_id)
        reasons: list[str] = []
        try:
            raw_sample = canonical._load_raw_sample(paths.prior_pkl, sample_id)
            if sample_field(raw_sample, "opm", {}) is None:
                canonical_report["allowed"]["opm_none"].append(sample_id)
            if canonical._raw_mesh_has_charge_nan(raw_sample):
                canonical_report["allowed"]["mesh_charge_nan"].append(sample_id)
            sample = parse_prior_payload(
                {sample_id: raw_sample},
                sample_id,
                paths,
                dataset_name,
                chain_sequences.get(sample_id),
            )
            ca_resseq, invalid_charges = _scan_gt_pdb(paths.gt_pdb)
            for record in invalid_charges:
                malformed_report["invalid_charge_fields"].append(
                    {"sample_id": sample_id, "path": str(paths.gt_pdb), **record}
                )
            reasons.extend(canonical._validate_resseq_membership(sample, ca_resseq))
            if sample.paths.main_msa_query_only:
                canonical_report["warnings"]["main_msa_query_only"].append(sample_id)
        except Exception as exc:
            canonical._append_reason(reasons, str(exc))
        if reasons:
            canonical_report["fatal"]["ids"].append(sample_id)
            canonical_report["fatal"]["reasons"][sample_id] = reasons

        if template_report is not None:
            for template_path in template_paths[sample_id]:
                for record in template_atom_ids._nonincreasing_atom_ids(template_path):
                    template_report["invalid_atom_ids"].append(
                        {"sample_id": sample_id, "path": str(template_path), **record}
                    )
                    template_report["skipped_sample_ids"].append(sample_id)

    for values in canonical_report["allowed"].values():
        values.sort()
    for values in canonical_report["warnings"].values():
        values.sort()
    canonical_report["fatal"]["ids"].sort()
    canonical_report["fatal"]["reasons"] = dict(
        sorted(canonical_report["fatal"]["reasons"].items())
    )
    if template_report is not None:
        template_report["skipped_sample_ids"] = sorted(
            set(template_report["skipped_sample_ids"])
        )

    return {
        "canonical": canonical_report,
        "malformed_charge": malformed_report,
        "template_atom_ids": template_report,
    }


def _subset_canonical_report(
    report: dict[str, Any], selected_ids: list[str]
) -> dict[str, object]:
    selected = set(selected_ids)
    fatal_ids = [
        sample_id for sample_id in report["fatal"]["ids"] if sample_id in selected
    ]
    return {
        "selected": len(selected_ids),
        "source_scanned": len(selected_ids),
        "allowed": {
            key: sorted(sample_id for sample_id in values if sample_id in selected)
            for key, values in report["allowed"].items()
        },
        "warnings": {
            key: sorted(sample_id for sample_id in values if sample_id in selected)
            for key, values in report["warnings"].items()
        },
        "fatal": {
            "ids": sorted(fatal_ids),
            "reasons": {
                sample_id: report["fatal"]["reasons"][sample_id]
                for sample_id in sorted(fatal_ids)
            },
        },
    }


def _subset_sample_records(
    records: list[dict[str, object]], selected_ids: list[str]
) -> list[dict[str, object]]:
    selected = set(selected_ids)
    return [
        copy.deepcopy(record) for record in records if record["sample_id"] in selected
    ]


def reuse_subset_audit_results(
    *,
    manifest: dict[str, object],
    prior_manifest: dict[str, object],
    prior_report: dict[str, object],
) -> dict[str, object]:
    """Re-key a verified audit for a manifest that only removes selected IDs."""

    if prior_report.get("complete") is not True:
        raise ValueError("prior audit report is not complete")
    results = prior_report.get("results")
    if not isinstance(results, dict):
        raise ValueError("prior audit report has no results")

    current_ids = {
        dataset: _manifest_train_ids(manifest, dataset)
        for dataset in ("qbiolip", "afdb")
    }
    prior_ids = {
        dataset: _manifest_train_ids(prior_manifest, dataset)
        for dataset in ("qbiolip", "afdb")
    }
    for dataset in current_ids:
        if not set(current_ids[dataset]).issubset(prior_ids[dataset]):
            raise ValueError(
                f"current {dataset} manifest is not a subset of the prior audit"
            )

    canonical_priors = {
        dataset: _subset_canonical_report(
            results["canonical_priors"][dataset], current_ids[dataset]
        )
        for dataset in current_ids
    }
    malformed_datasets: dict[str, object] = {}
    for dataset, selected_ids in current_ids.items():
        prior_malformed = results["malformed_charge"]["datasets"][dataset]
        malformed_datasets[dataset] = {
            "selected": len(selected_ids),
            "found": len(selected_ids),
            "missing_ids": [],
            "invalid_charge_fields": _subset_sample_records(
                prior_malformed["invalid_charge_fields"], selected_ids
            ),
        }

    selected_qbiolip = current_ids["qbiolip"]
    prior_template = results["template_atom_ids"]["qbiolip"]
    template = {
        "selected": len(selected_qbiolip),
        "found": len(selected_qbiolip),
        "missing_ids": [],
        "skipped_sample_ids": sorted(
            sample_id
            for sample_id in prior_template["skipped_sample_ids"]
            if sample_id in set(selected_qbiolip)
        ),
        "invalid_atom_ids": _subset_sample_records(
            prior_template["invalid_atom_ids"], selected_qbiolip
        ),
    }
    return {
        "manifest_registration": {
            "qbiolip_train_ids": len(selected_qbiolip),
            "afdb_train_ids": len(current_ids["afdb"]),
        },
        "reader_index_caches": copy.deepcopy(results["reader_index_caches"]),
        "canonical_priors": canonical_priors,
        "malformed_charge": {
            "recoverable_by_local_retry": True,
            "datasets": malformed_datasets,
        },
        "template_atom_ids": {"qbiolip": template},
        "reused_from": {
            "audit_key": prior_report.get("audit_key"),
            "mode": "verified-manifest-subset",
        },
    }


def build_training_audit(
    *,
    manifest: dict[str, object],
    qbiolip_root: Path,
    afdb_root: Path,
    qbiolip_index_cache: Path,
    afdb_index_cache: Path,
    prior_manifest: dict[str, object] | None = None,
    prior_report: dict[str, object] | None = None,
) -> dict[str, object]:
    qbiolip_ids = _manifest_train_ids(manifest, "qbiolip")
    afdb_ids = _manifest_train_ids(manifest, "afdb")
    if (prior_manifest is None) != (prior_report is None):
        raise ValueError("prior manifest and report must be supplied together")
    if prior_manifest is not None and prior_report is not None:
        result = reuse_subset_audit_results(
            manifest=manifest,
            prior_manifest=prior_manifest,
            prior_report=prior_report,
        )
    else:
        combined = {
            "qbiolip": audit_dataset_once(
                "qbiolip", qbiolip_root, qbiolip_ids, qbiolip_index_cache
            ),
            "afdb": audit_dataset_once("afdb", afdb_root, afdb_ids, afdb_index_cache),
        }
        result = {
            "manifest_registration": {
                "qbiolip_train_ids": len(qbiolip_ids),
                "afdb_train_ids": len(afdb_ids),
            },
            "reader_index_caches": {},
            "canonical_priors": {
                dataset: report["canonical"] for dataset, report in combined.items()
            },
            "malformed_charge": {
                "recoverable_by_local_retry": True,
                "datasets": {
                    dataset: report["malformed_charge"]
                    for dataset, report in combined.items()
                },
            },
            "template_atom_ids": {"qbiolip": combined["qbiolip"]["template_atom_ids"]},
        }

    current_caches = {
        "qbiolip": _cache_evidence(qbiolip_index_cache),
        "afdb": _cache_evidence(afdb_index_cache),
    }
    if prior_report is not None:
        for dataset, evidence in current_caches.items():
            prior_evidence = result["reader_index_caches"].get(dataset, {})
            if prior_evidence.get("sha256") != evidence["sha256"]:
                raise ValueError(f"{dataset} reader index changed since prior audit")
    result["reader_index_caches"] = current_caches

    malformed = result["malformed_charge"]["datasets"]
    for dataset, report in malformed.items():
        _require_selected_found(f"malformed-charge {dataset}", report)

    template = result["template_atom_ids"]["qbiolip"]
    if template is None:
        raise AssertionError("Q-BioLiP audit must include template atom IDs")
    _require_selected_found("template-atom-ID qbiolip", template)
    if template["invalid_atom_ids"]:
        raise ValueError("manifest contains nonincreasing template atom IDs")

    canonical_reports = result["canonical_priors"]
    for dataset, report in canonical_reports.items():
        if report["fatal"]["ids"]:
            raise ValueError(f"canonical-prior audit found fatal {dataset} IDs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--code-head", required=True)
    parser.add_argument("--qbiolip-root", type=Path, required=True)
    parser.add_argument("--afdb-root", type=Path, required=True)
    parser.add_argument("--qbiolip-index-cache", type=Path, required=True)
    parser.add_argument("--afdb-index-cache", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if (args.prior_manifest is None) != (args.prior_report is None):
        parser.error("--prior-manifest and --prior-report must be supplied together")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    prior_manifest = None
    prior_report = None
    if args.prior_manifest is not None and args.prior_report is not None:
        prior_manifest_bytes = args.prior_manifest.read_bytes()
        prior_manifest = json.loads(prior_manifest_bytes)
        prior_report = json.loads(args.prior_report.read_text())
        if (
            prior_report.get("manifest_sha256")
            != hashlib.sha256(prior_manifest_bytes).hexdigest()
        ):
            raise ValueError("prior report does not match the supplied prior manifest")
    report, reused = run_audit_once(
        args.output,
        manifest_bytes,
        args.code_head,
        lambda: build_training_audit(
            manifest=manifest,
            qbiolip_root=args.qbiolip_root,
            afdb_root=args.afdb_root,
            qbiolip_index_cache=args.qbiolip_index_cache,
            afdb_index_cache=args.afdb_index_cache,
            prior_manifest=prior_manifest,
            prior_report=prior_report,
        ),
    )
    print(json.dumps({"audit_key": report["audit_key"], "reused": reused}))


if __name__ == "__main__":
    main()
