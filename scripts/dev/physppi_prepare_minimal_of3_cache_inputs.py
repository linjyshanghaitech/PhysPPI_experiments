from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from biotite.structure import filter_amino_acids
from biotite.structure.io import pdb, pdbx


@dataclass(frozen=True)
class RequiredInputFiles:
    sample_id: str
    pkl_path: Path
    pdb_path: Path


def find_required_input_files(
    sample_id: str,
    pkl_dir: Path,
    pdb_dir: Path,
) -> RequiredInputFiles:
    pkl_path = pkl_dir / f"{sample_id}.pkl"
    pdb_path = pdb_dir / f"{sample_id}.pdb"
    if not pkl_path.is_file():
        raise FileNotFoundError(f"missing PhysPPI PKL: {pkl_path}")
    if not pdb_path.is_file():
        raise FileNotFoundError(f"missing PhysPPI PDB: {pdb_path}")
    return RequiredInputFiles(sample_id=sample_id, pkl_path=pkl_path, pdb_path=pdb_path)


def read_protein_chain_sequences(pdb_path: Path) -> dict[str, str]:
    pdb_file = pdb.PDBFile.read(pdb_path)
    atom_array = pdb_file.get_structure(model=1, extra_fields=["b_factor"])
    atom_array = atom_array[filter_amino_acids(atom_array)]
    residue_names = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "MSE": "M",
    }
    chains: dict[str, list[tuple[int, str]]] = {}
    for chain_id, res_id, res_name in zip(
        atom_array.chain_id.tolist(),
        atom_array.res_id.tolist(),
        atom_array.res_name.tolist(),
        strict=True,
    ):
        if res_name not in residue_names:
            continue
        chain_rows = chains.setdefault(str(chain_id), [])
        if not chain_rows or chain_rows[-1][0] != int(res_id):
            chain_rows.append((int(res_id), residue_names[res_name]))
    sequences = {
        chain_id: "".join(residue for _, residue in rows)
        for chain_id, rows in sorted(chains.items())
    }
    if not sequences:
        raise ValueError(f"no protein sequences parsed from {pdb_path}")
    return sequences


def convert_pdb_to_cif(
    pdb_path: Path,
    cif_path: Path,
    data_block: str | None = None,
) -> None:
    pdb_file = pdb.PDBFile.read(pdb_path)
    atom_array = pdb_file.get_structure(model=1, extra_fields=["b_factor"])
    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, atom_array, data_block=data_block or pdb_path.stem)
    cif_path.parent.mkdir(parents=True, exist_ok=True)
    pdbx.CIFFile.write(cif_file, cif_path)


def write_minimal_inference_query(
    sample_id: str,
    chains: dict[str, str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "queries": {
            sample_id: {
                "chains": [
                    {
                        "chain_ids": [chain_id],
                        "molecule_type": "protein",
                        "sequence": sequence,
                    }
                    for chain_id, sequence in sorted(chains.items())
                ]
            }
        }
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default="7z1e_1")
    parser.add_argument(
        "--pkl-dir",
        type=Path,
        default=Path("/101045/jinyu/PhysPPI/prior_processed/v3-Qbiolip-M"),
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=Path("/101045/jinyu/PhysPPI/prior_processed/v3-Qbiolip-M-pdb"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/physppi_msa_cache"),
    )
    args = parser.parse_args()

    files = find_required_input_files(args.sample_id, args.pkl_dir, args.pdb_dir)
    staged_pdb = args.output_dir / "raw_pdb" / f"{args.sample_id}.pdb"
    staged_cif = args.output_dir / "raw_cif" / f"{args.sample_id}.cif"
    query_json = args.output_dir / "query_7z1e_1.json"
    manifest_path = args.output_dir / "run_manifest.json"

    staged_pdb.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(files.pdb_path, staged_pdb)
    convert_pdb_to_cif(staged_pdb, staged_cif)
    chains = read_protein_chain_sequences(staged_pdb)
    write_minimal_inference_query(args.sample_id, chains, query_json)

    manifest = {
        "sample_id": args.sample_id,
        "source": asdict(files),
        "staged_pdb": str(staged_pdb),
        "staged_cif": str(staged_cif),
        "query_json": str(query_json),
        "chains": chains,
        "output_dir": str(args.output_dir),
    }
    manifest["source"]["pkl_path"] = str(files.pkl_path)
    manifest["source"]["pdb_path"] = str(files.pdb_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
