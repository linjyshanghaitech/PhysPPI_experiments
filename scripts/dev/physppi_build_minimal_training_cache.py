"""Build minimal ClusteredDatasetCache from preprocessing metadata + MSA output."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path


def _structure_id_for_sample(sample_id: str, structure_data: dict) -> str:
    candidates = [sample_id, sample_id.split("_", maxsplit=1)[0]]
    for candidate in candidates:
        if candidate in structure_data:
            return candidate
    raise KeyError(f"missing PhysPPI samples in metadata: {sample_id}")


def _resolve_structure_id(
    *,
    sample_id: str,
    structure_data: dict,
    sample_to_structure_id: dict[str, str] | None = None,
) -> str:
    if sample_to_structure_id is not None and sample_id in sample_to_structure_id:
        structure_id = sample_to_structure_id[sample_id]
        if structure_id not in structure_data:
            raise KeyError(
                "missing PhysPPI source structure in metadata: "
                f"sample={sample_id} structure={structure_id}"
            )
        return structure_id
    return _structure_id_for_sample(sample_id, structure_data)


def _default_output(base: Path, sample_ids: list[str]) -> Path:
    if sample_ids == ["7z1e_1"]:
        return base / "training_cache_7z1e_1.json"
    return base / "training_cache_small_source1.json"


def sample_ids_from_manifest(manifest: Path) -> list[str]:
    payload = json.loads(manifest.read_text())
    sample_ids = payload.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError(f"manifest has no non-empty sample_ids list: {manifest}")
    return [str(sample_id) for sample_id in sample_ids]


def sample_to_structure_id_from_json(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"sample structure map must be a non-empty object: {path}")
    return {
        str(sample_id): str(structure_id) for sample_id, structure_id in payload.items()
    }


def sample_to_structure_id_from_manifest(manifest: Path) -> dict[str, str] | None:
    payload = json.loads(manifest.read_text())
    mapping = payload.get("sample_to_structure_id")
    if isinstance(mapping, dict) and mapping:
        return {
            str(sample_id): str(structure_id)
            for sample_id, structure_id in mapping.items()
        }
    mapping_path = payload.get("sample_to_structure_id_json")
    if isinstance(mapping_path, str) and mapping_path:
        path = Path(mapping_path)
        if not path.is_absolute():
            path = manifest.parent / path
        return sample_to_structure_id_from_json(path)
    return None


def _format_interfaces(interfaces: list[list[str]] | dict) -> dict:
    if isinstance(interfaces, dict):
        return interfaces
    formatted = {}
    for pair in interfaces:
        interface_id = "_".join(str(chain_id) for chain_id in pair)
        formatted[interface_id] = {
            "cluster_id": "0",
            "cluster_size": 1,
        }
    return formatted


def _format_reference_molecule_data(reference_data: dict) -> dict:
    formatted = {}
    for ref_id, data in reference_data.items():
        formatted[ref_id] = {
            "conformer_gen_strategy": data.get("conformer_gen_strategy", "default"),
            "fallback_conformer_pdb_id": data.get("fallback_conformer_pdb_id"),
            "canonical_smiles": data.get("canonical_smiles", ""),
            "set_fallback_to_nan": data.get("set_fallback_to_nan", False),
        }
    return formatted


# Hardcoded canonical SMILES for the 20 standard amino acids (zwitterionic forms
# used in CCD).  Used as a fallback when preprocessing metadata lacks
# reference_molecule_data entries.
_STANDARD_AMINO_ACID_SMILES: dict[str, str] = {
    "ALA": "C[C@@H](N)C(=O)O",
    "ARG": "NC(=N)NCCC[C@@H](N)C(=O)O",
    "ASN": "NC(=O)C[C@@H](N)C(=O)O",
    "ASP": "OC(=O)C[C@@H](N)C(=O)O",
    "CYS": "SC[C@@H](N)C(=O)O",
    "GLN": "NC(=O)CC[C@@H](N)C(=O)O",
    "GLU": "OC(=O)CC[C@@H](N)C(=O)O",
    "GLY": "NCC(=O)O",
    "HIS": "C1=C(NC=N1)C[C@@H](N)C(=O)O",
    "ILE": "CC[C@H](C)[C@@H](N)C(=O)O",
    "LEU": "CC(C)C[C@@H](N)C(=O)O",
    "LYS": "NCCCC[C@@H](N)C(=O)O",
    "MET": "CSCC[C@@H](N)C(=O)O",
    "PHE": "C1=CC=C(C=C1)C[C@@H](N)C(=O)O",
    "PRO": "C1C[C@H](NC1)C(=O)O",
    "SER": "OC[C@@H](N)C(=O)O",
    "THR": "C[C@@H](O)[C@@H](N)C(=O)O",
    "TRP": "C1=CC=C2C(=C1)C(=CN2)C[C@@H](N)C(=O)O",
    "TYR": "C1=CC(=CC=C1C[C@@H](N)C(=O)O)O",
    "VAL": "CC(C)[C@@H](N)C(=O)O",
}


def _synthesize_reference_molecule_data(base: Path) -> dict:
    """Build default reference molecule metadata from SDF files on disk.

    When the preprocessing pipeline produced SDF files but omitted
    ``reference_molecule_data`` from its metadata, this function generates
    safe defaults so the training dataloader does not hit a ``KeyError``.

    Every entry receives ``conformer_gen_strategy="default"`` and
    ``set_fallback_to_nan=False``.  Canonical SMILES are taken from
    ``_STANDARD_AMINO_ACID_SMILES`` when the CCD code is recognised;
    otherwise the field is set to an empty string (not consumed during
    training).
    """
    ref_mol_dir = base / "preprocessed_pdb" / "reference_mols"
    synthesized: dict[str, dict] = {}
    for sdf_path in sorted(ref_mol_dir.glob("*.sdf")):
        mol_id = sdf_path.stem
        synthesized[mol_id] = {
            "conformer_gen_strategy": "default",
            "fallback_conformer_pdb_id": None,
            "canonical_smiles": _STANDARD_AMINO_ACID_SMILES.get(mol_id, ""),
            "set_fallback_to_nan": False,
        }
    return synthesized


def ensure_sample_id_structure_aliases(
    *,
    base: Path,
    metadata: dict,
    sample_ids: list[str],
    sample_to_structure_id: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    structure_root = base / "preprocessed_pdb" / "structure_files"
    structure_data = metadata.get("structure_data", {})
    for sample_id in sample_ids:
        structure_id = _resolve_structure_id(
            sample_id=sample_id,
            structure_data=structure_data,
            sample_to_structure_id=sample_to_structure_id,
        )
        sample_dir = structure_root / sample_id
        copied = []
        for suffix in ("fasta", "npz", "pkl"):
            source = sample_dir / f"{structure_id}.{suffix}"
            if not source.is_file():
                source = structure_root / structure_id / f"{structure_id}.{suffix}"
            target = sample_dir / f"{sample_id}.{suffix}"
            if source.is_file() and source != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.append(str(target))
        aliases[sample_id] = copied
    return aliases


def build_training_cache_payload(
    *,
    metadata: dict,
    chain_id_to_rep_id: dict[str, str],
    sample_ids: list[str],
    sample_to_structure_id: dict[str, str] | None = None,
    base: Path | None = None,
) -> dict:
    if not sample_ids:
        raise ValueError("sample_ids must not be empty")

    structure_data = metadata.get("structure_data", {})
    selected_structure_data = {}
    for sample_id in sample_ids:
        structure_id = _resolve_structure_id(
            sample_id=sample_id,
            structure_data=structure_data,
            sample_to_structure_id=sample_to_structure_id,
        )
        struct = structure_data[structure_id]
        chains = {}
        fallback_rep_ids = list(chain_id_to_rep_id.values())
        for i, chain_id in enumerate(sorted(struct.get("chains", {}).keys())):
            chain_metadata = copy.deepcopy(struct["chains"][chain_id])
            rep_key = f"{sample_id}-{chain_id}"
            structure_rep_key = f"{structure_id}-{chain_id}"
            fallback = None
            if fallback_rep_ids:
                fallback = (
                    fallback_rep_ids[i]
                    if i < len(fallback_rep_ids)
                    else fallback_rep_ids[0]
                )
            rep_id = chain_id_to_rep_id.get(rep_key)
            if rep_id is None:
                rep_id = chain_id_to_rep_id.get(structure_rep_key)
            if rep_id is None:
                rep_id = fallback
            if rep_id is None:
                raise KeyError(
                    "missing alignment representative for "
                    f"sample={sample_id} chain={chain_id}"
                )
            chain_metadata.setdefault("label_asym_id", chain_id)
            chain_metadata.setdefault("auth_asym_id", chain_id)
            chain_metadata.setdefault("entity_id", i + 1)
            chain_metadata.setdefault("molecule_type", "PROTEIN")
            chain_metadata.setdefault("reference_mol_id", None)
            chain_metadata.setdefault("template_ids", None)
            chains[chain_id] = {
                **chain_metadata,
                "alignment_representative_id": rep_id,
                "cluster_id": "0",
                "cluster_size": 1,
            }
        selected_structure_data[sample_id] = {
            "chains": chains,
            "interfaces": _format_interfaces(struct.get("interfaces", [])),
            "release_date": struct.get("release_date"),
            "resolution": struct.get("resolution"),
        }

    cache_name = (
        "physppi-minimal-7z1e-1"
        if sample_ids == ["7z1e_1"]
        else "physppi-small-source1"
    )
    ref_mol_data_raw = metadata.get("reference_molecule_data", {})
    ref_mol_data = dict(ref_mol_data_raw) if ref_mol_data_raw else {}

    # Always merge synthesized defaults for any SDF files that exist on disk
    # but are missing from the preprocessing metadata.  This handles:
    #   - preprocessing metadata that completely omits reference_molecule_data
    #   - preprocessing metadata that includes ligands but not standard
    #     amino-acid residues (e.g. after chunked-preprocess merge)
    # Existing metadata entries take precedence — they are never overwritten.
    if base is not None:
        synthesized = _synthesize_reference_molecule_data(base)
        for mol_id, mol_data in synthesized.items():
            if mol_id not in ref_mol_data:
                ref_mol_data[mol_id] = mol_data

    return {
        "_type": "ClusteredDatasetCache",
        "name": cache_name,
        "structure_data": selected_structure_data,
        "reference_molecule_data": _format_reference_molecule_data(ref_mol_data),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("output/physppi_msa_cache"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        help="Sample id to include. Repeat for multi-sample caches.",
    )
    parser.add_argument("--sample-structure-map", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base = args.base
    meta = json.loads((base / "preprocessed_pdb/metadata.json").read_text())
    chain_map = json.loads(
        (base / "colabfold_msas/mappings/chain_id_to_rep_id.json").read_text()
    )
    sample_ids = (
        sample_ids_from_manifest(args.manifest)
        if args.manifest is not None
        else args.sample_ids or ["7z1e_1"]
    )
    sample_to_structure_id = (
        sample_to_structure_id_from_json(args.sample_structure_map)
        if args.sample_structure_map is not None
        else (
            sample_to_structure_id_from_manifest(args.manifest)
            if args.manifest is not None
            else None
        )
    )
    aliases = ensure_sample_id_structure_aliases(
        base=base,
        metadata=meta,
        sample_ids=sample_ids,
        sample_to_structure_id=sample_to_structure_id,
    )

    cache = build_training_cache_payload(
        metadata=meta,
        chain_id_to_rep_id=chain_map,
        sample_ids=sample_ids,
        sample_to_structure_id=sample_to_structure_id,
        base=base,
    )
    out = args.output or _default_output(base, sample_ids)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, indent=2) + "\n")
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    print(
        json.dumps(
            {
                "sample_ids": sample_ids,
                "structure_ids": list(cache["structure_data"].keys()),
                "structure_aliases": aliases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
