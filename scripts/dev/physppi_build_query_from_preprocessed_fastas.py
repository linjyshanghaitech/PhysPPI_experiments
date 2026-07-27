"""Build OpenFold3 MSA query JSON from preprocessed structure FASTA files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    header: str | None = None
    sequence_parts: list[str] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records[header] = "".join(sequence_parts)
            header = line[1:].strip()
            sequence_parts = []
        else:
            sequence_parts.append(line)

    if header is not None:
        records[header] = "".join(sequence_parts)
    if not records:
        raise ValueError(f"no FASTA records: {path}")
    return records


def sample_ids_from_manifest(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text())
    sample_ids = payload.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError(f"manifest has no non-empty sample_ids list: {manifest_path}")
    return [str(sample_id) for sample_id in sample_ids]


def sample_to_structure_id_from_json(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"sample structure map must be a non-empty object: {path}")
    return {
        str(sample_id): str(structure_id) for sample_id, structure_id in payload.items()
    }


def _find_single_fasta(sample_dir: Path, sample_id: str) -> Path:
    fasta_paths = sorted(sample_dir.glob("*.fasta"))
    if len(fasta_paths) != 1:
        raise FileNotFoundError(
            f"expected exactly one FASTA for {sample_id} in {sample_dir}; "
            f"found {len(fasta_paths)}"
        )
    return fasta_paths[0]


def build_query_from_preprocessed_fastas(
    *,
    sample_ids: list[str],
    structure_files_directory: Path,
    sample_to_structure_id: dict[str, str] | None = None,
) -> dict:
    queries = {}
    for sample_id in sample_ids:
        structure_id = (
            sample_to_structure_id.get(sample_id, sample_id)
            if sample_to_structure_id is not None
            else sample_id
        )
        sample_dir = structure_files_directory / structure_id
        fasta_path = _find_single_fasta(sample_dir, structure_id)
        chains = parse_fasta(fasta_path)
        queries[sample_id] = {
            "chains": [
                {
                    "chain_ids": [chain_id],
                    "molecule_type": "protein",
                    "sequence": sequence,
                }
                for chain_id, sequence in sorted(chains.items())
            ]
        }
    return {"queries": queries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-structure-map", type=Path)
    parser.add_argument("--structure-files-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = build_query_from_preprocessed_fastas(
        sample_ids=sample_ids_from_manifest(args.manifest),
        structure_files_directory=args.structure_files_directory,
        sample_to_structure_id=(
            sample_to_structure_id_from_json(args.sample_structure_map)
            if args.sample_structure_map is not None
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_queries": len(payload["queries"]),
                "sample_ids": list(payload["queries"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
