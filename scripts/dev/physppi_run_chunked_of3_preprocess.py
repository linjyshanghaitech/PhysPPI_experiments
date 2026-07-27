from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PIXI_PYTHON = [
    "/101045/jinyu/.pixi/bin/pixi",
    "run",
    "-e",
    "openfold3-cpu",
    "python",
]
PROTEIN_REFERENCE_MOL_IDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "MSE",
        "PHE",
        "PRO",
        "SEC",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)


@dataclass(frozen=True)
class ChunkCommand:
    chunk_name: str
    cif_dir: Path
    output_dir: Path
    argv: list[str]


def split_cif_paths(
    cif_paths: list[Path],
    *,
    max_cifs_per_chunk: int,
) -> list[list[Path]]:
    if max_cifs_per_chunk < 1:
        raise ValueError("max_cifs_per_chunk must be >= 1")
    sorted_paths = sorted(cif_paths, key=lambda path: path.name)
    return [
        sorted_paths[index : index + max_cifs_per_chunk]
        for index in range(0, len(sorted_paths), max_cifs_per_chunk)
    ]


def write_chunk_cif_dirs(*, chunks: list[list[Path]], output_dir: Path) -> list[Path]:
    chunk_dirs: list[Path] = []
    root = output_dir / "chunks" / "cif_dirs"
    for index, chunk in enumerate(chunks):
        chunk_dir = root / f"chunk_{index:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for source in chunk:
            shutil.copy2(source, chunk_dir / source.name)
        chunk_dirs.append(chunk_dir)
    return chunk_dirs


def _status_path(status_dir: Path, chunk_name: str) -> Path:
    return status_dir / f"{chunk_name}.json"


def _read_status(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _write_status(status_dir: Path, chunk_name: str, payload: dict) -> Path:
    status_dir.mkdir(parents=True, exist_ok=True)
    path = _status_path(status_dir, chunk_name)
    path.write_text(json.dumps({"chunk": chunk_name, **payload}, indent=2) + "\n")
    return path


def _execute_argv(argv: list[str], *, execute, timeout_seconds: int | None) -> int:
    if timeout_seconds is None or execute is not subprocess.call:
        return int(execute(argv))
    try:
        return int(subprocess.run(argv, timeout=timeout_seconds).returncode)
    except subprocess.TimeoutExpired:
        return 124


def _replace_arg(argv: list[str], flag: str, value: Path) -> list[str]:
    updated = list(argv)
    try:
        index = updated.index(flag)
    except ValueError as exc:
        raise ValueError(f"argv missing required flag {flag}") from exc
    if index + 1 >= len(updated):
        raise ValueError(f"argv flag {flag} has no value")
    updated[index + 1] = str(value)
    return updated


def _run_fallback_subchunks(
    *,
    command: ChunkCommand,
    execute,
    max_cifs_per_chunk: int,
    timeout_seconds: int | None,
) -> dict:
    cif_paths = sorted(command.cif_dir.glob("*.cif"), key=lambda path: path.name)
    subchunks = split_cif_paths(
        cif_paths,
        max_cifs_per_chunk=max_cifs_per_chunk,
    )
    fallback_root = command.output_dir.parent / "fallback"
    if fallback_root.exists():
        shutil.rmtree(fallback_root)
    fallback_cif_dirs = write_chunk_cif_dirs(
        chunks=subchunks,
        output_dir=fallback_root,
    )

    subchunk_outputs: list[Path] = []
    subchunk_reports: list[dict] = []
    for index, fallback_cif_dir in enumerate(fallback_cif_dirs):
        subchunk_name = f"{command.chunk_name}_fallback_{index:04d}"
        subchunk_output = fallback_root / "chunks" / subchunk_name / "preprocessed_pdb"
        argv = _replace_arg(command.argv, "--cif-dir", fallback_cif_dir)
        argv = _replace_arg(argv, "--out-dir", subchunk_output)
        return_code = _execute_argv(
            argv,
            execute=execute,
            timeout_seconds=timeout_seconds,
        )
        subchunk_reports.append(
            {
                "chunk": subchunk_name,
                "cif_count": len(subchunks[index]),
                "return_code": return_code,
            }
        )
        if return_code != 0:
            raise RuntimeError(
                "fallback preprocess subchunk failed: "
                f"{subchunk_name} return_code={return_code}"
            )
        subchunk_outputs.append(subchunk_output)

    if command.output_dir.exists():
        shutil.rmtree(command.output_dir)
    merge_report = merge_chunk_outputs(
        chunk_output_dirs=subchunk_outputs,
        output_dir=command.output_dir,
    )
    return {
        "fallback_used": True,
        "fallback_max_cifs_per_chunk": max_cifs_per_chunk,
        "fallback_subchunk_count": len(subchunk_outputs),
        "fallback_subchunks": subchunk_reports,
        "fallback_merge_report": merge_report,
    }


def run_chunk_command(
    *,
    command: ChunkCommand,
    status_dir: Path,
    execute=subprocess.call,
    force: bool = False,
    fallback_max_cifs_per_chunk: int | None = None,
    timeout_seconds: int | None = None,
) -> dict:
    prior = _read_status(_status_path(status_dir, command.chunk_name))
    if prior and prior.get("status") == "passed" and not force:
        return {"chunk": command.chunk_name, "status": "skipped"}
    if (
        force or prior is None or prior.get("status") == "failed"
    ) and command.output_dir.exists():
        shutil.rmtree(command.output_dir)

    return_code = _execute_argv(
        command.argv,
        execute=execute,
        timeout_seconds=timeout_seconds,
    )
    status = "passed" if return_code == 0 else "failed"
    report = {
        "status": status,
        "argv": command.argv,
        "return_code": return_code,
    }
    if return_code != 0 and fallback_max_cifs_per_chunk is not None:
        fallback_report = _run_fallback_subchunks(
            command=command,
            execute=execute,
            max_cifs_per_chunk=fallback_max_cifs_per_chunk,
            timeout_seconds=timeout_seconds,
        )
        report = {
            "status": "passed",
            "argv": command.argv,
            "return_code": 0,
            "initial_return_code": return_code,
            "timeout_seconds": timeout_seconds,
            **fallback_report,
        }
    else:
        report["timeout_seconds"] = timeout_seconds
    _write_status(status_dir, command.chunk_name, report)
    if report["status"] != "passed":
        raise RuntimeError(
            f"preprocess chunk failed: {command.chunk_name} return_code={return_code}"
        )
    return {"chunk": command.chunk_name, **report}


def _copy_structure_dirs(source_root: Path, target_root: Path) -> tuple[int, int]:
    if not source_root.is_dir():
        return 0, 0
    target_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    duplicates = 0
    for source in sorted(source_root.iterdir(), key=lambda path: path.name):
        if not source.is_dir():
            continue
        target = target_root / source.name
        if target.exists():
            duplicates += 1
            continue
        shutil.copytree(source, target)
        copied += 1
    return copied, duplicates


def _copy_reference_mols(source_root: Path, target_root: Path) -> tuple[int, int]:
    if not source_root.is_dir():
        return 0, 0
    target_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    duplicates = 0
    for source in sorted(source_root.iterdir(), key=lambda path: path.name):
        if not source.is_file():
            continue
        target = target_root / source.name
        if target.exists():
            duplicates += 1
            continue
        shutil.copy2(source, target)
        copied += 1
    return copied, duplicates


def collect_atom_site_comp_ids(
    cif_paths: list[Path],
    *,
    candidate_comp_ids: set[str] | frozenset[str] | None = None,
) -> set[str]:
    comp_ids: set[str] = set()
    remaining = set(candidate_comp_ids or ())
    comp_pattern = re.compile(r"^ATOM\s+\S+\s+\S+\s+\S+\s+(\S+)")
    for cif_path in cif_paths:
        if candidate_comp_ids is not None and not remaining:
            break
        atom_site_columns: list[str] = []
        in_loop = False
        with cif_path.open() as handle:
            lines = handle
            for raw_line in lines:
                line = raw_line.strip()
                if not line or line == "#":
                    in_loop = False
                    atom_site_columns = []
                    continue
                if line == "loop_":
                    in_loop = True
                    atom_site_columns = []
                    continue
                if in_loop and line.startswith("_atom_site."):
                    atom_site_columns.append(line.split()[0])
                    continue
                if not atom_site_columns:
                    continue
                if line.startswith("_"):
                    atom_site_columns = []
                    in_loop = False
                    continue
                if candidate_comp_ids is not None:
                    match = comp_pattern.match(line)
                    if match:
                        comp_id = match.group(1).strip("'\"")
                        if comp_id in remaining:
                            comp_ids.add(comp_id)
                            remaining.remove(comp_id)
                    continue

                try:
                    group_index = atom_site_columns.index("_atom_site.group_PDB")
                    comp_index = atom_site_columns.index("_atom_site.label_comp_id")
                except ValueError:
                    atom_site_columns = []
                    in_loop = False
                    continue

                tokens = line.split()
                if len(tokens) <= max(group_index, comp_index):
                    continue
                if tokens[group_index] == "ATOM":
                    comp_ids.add(tokens[comp_index])
    return comp_ids


def _reference_mol_ids(reference_root: Path) -> set[str]:
    if not reference_root.is_dir():
        return set()
    return {path.stem for path in reference_root.iterdir() if path.is_file()}


def _unexpected_reference_mols(
    reference_root: Path,
    *,
    allowed_reference_mol_ids: set[str] | frozenset[str] = PROTEIN_REFERENCE_MOL_IDS,
) -> list[str]:
    if not reference_root.is_dir():
        return []
    return sorted(
        path.stem
        for path in reference_root.iterdir()
        if path.is_file() and path.stem not in allowed_reference_mol_ids
    )


def merge_chunk_outputs(
    *,
    chunk_output_dirs: list[Path],
    output_dir: Path,
    protein_only_reference_mols: bool = False,
    allowed_reference_mol_ids: set[str] | None = None,
) -> dict:
    allowed_reference_mol_ids = allowed_reference_mol_ids or set(
        PROTEIN_REFERENCE_MOL_IDS
    )
    unexpected_reference_mols: dict[str, list[str]] = {}
    if protein_only_reference_mols:
        for chunk_output in chunk_output_dirs:
            unexpected = _unexpected_reference_mols(
                chunk_output / "reference_mols",
                allowed_reference_mol_ids=allowed_reference_mol_ids,
            )
            if unexpected:
                unexpected_reference_mols[str(chunk_output)] = unexpected[:20]
        if unexpected_reference_mols:
            raise ValueError(
                "protein-only preprocessing found non-protein reference_mols: "
                + json.dumps(unexpected_reference_mols, sort_keys=True)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir_name in ["structure_files", "reference_mols"]:
        subdir = output_dir / subdir_name
        if subdir.exists():
            shutil.rmtree(subdir)
    structure_count = 0
    duplicate_structure_count = 0
    reference_count = 0
    duplicate_reference_count = 0
    for chunk_output in chunk_output_dirs:
        copied, duplicates = _copy_structure_dirs(
            chunk_output / "structure_files",
            output_dir / "structure_files",
        )
        structure_count += copied
        duplicate_structure_count += duplicates
        copied_refs, duplicate_refs = _copy_reference_mols(
            chunk_output / "reference_mols",
            output_dir / "reference_mols",
        )
        reference_count += copied_refs
        duplicate_reference_count += duplicate_refs

    report = {
        "status": "passed",
        "chunk_count": len(chunk_output_dirs),
        "structure_dir_count": structure_count,
        "duplicate_structure_dir_count": duplicate_structure_count,
        "reference_mol_count": reference_count,
        "duplicate_reference_mol_count": duplicate_reference_count,
        "protein_only_reference_mols": protein_only_reference_mols,
        "allowed_reference_mol_count": len(allowed_reference_mol_ids),
    }
    (output_dir / "chunked_preprocess_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def build_chunk_commands(
    *,
    chunk_cif_dirs: list[Path],
    output_dir: Path,
    ccd_path: Path,
    max_polymer_chains: int,
    num_workers_per_chunk: int,
    preprocess_chunksize: int,
) -> list[ChunkCommand]:
    commands: list[ChunkCommand] = []
    for index, chunk_cif_dir in enumerate(chunk_cif_dirs):
        chunk_name = f"chunk_{index:04d}"
        chunk_output = output_dir / "chunks" / chunk_name / "preprocessed_pdb"
        argv = PIXI_PYTHON + [
            "scripts/data_preprocessing/preprocess_pdb_of3.py",
            "--cif-dir",
            str(chunk_cif_dir),
            "--ccd-path",
            str(ccd_path),
            "--out-dir",
            str(chunk_output),
            "--max-polymer-chains",
            str(max_polymer_chains),
            "--num-workers",
            str(num_workers_per_chunk),
            "--chunksize",
            str(preprocess_chunksize),
            "--output-format",
            "npz",
            "--output-format",
            "pkl",
            "--log-level",
            "INFO",
        ]
        commands.append(
            ChunkCommand(
                chunk_name=chunk_name,
                cif_dir=chunk_cif_dir,
                output_dir=chunk_output,
                argv=argv,
            )
        )
    return commands


def run_chunked_preprocess(
    *,
    cif_dir: Path,
    ccd_path: Path,
    output_dir: Path,
    max_cifs_per_chunk: int,
    num_workers_per_chunk: int,
    max_polymer_chains: int,
    preprocess_chunksize: int,
    fallback_max_cifs_per_chunk: int | None = None,
    force_chunk: bool = False,
    protein_only_reference_mols: bool = False,
    chunk_timeout_seconds: int | None = None,
) -> dict:
    cif_paths = sorted(cif_dir.glob("*.cif"), key=lambda path: path.name)
    chunks = split_cif_paths(cif_paths, max_cifs_per_chunk=max_cifs_per_chunk)
    chunk_cif_dirs = write_chunk_cif_dirs(chunks=chunks, output_dir=output_dir)
    commands = build_chunk_commands(
        chunk_cif_dirs=chunk_cif_dirs,
        output_dir=output_dir,
        ccd_path=ccd_path,
        max_polymer_chains=max_polymer_chains,
        num_workers_per_chunk=num_workers_per_chunk,
        preprocess_chunksize=preprocess_chunksize,
    )

    status_dir = output_dir / "chunk_status"
    chunk_results: list[dict] = []
    for command in commands:
        chunk_results.append(
            run_chunk_command(
                command=command,
                status_dir=status_dir,
                force=force_chunk,
                fallback_max_cifs_per_chunk=fallback_max_cifs_per_chunk,
                timeout_seconds=chunk_timeout_seconds,
            )
        )
    allowed_reference_mol_ids = set(PROTEIN_REFERENCE_MOL_IDS)
    if protein_only_reference_mols:
        reference_mol_ids = set()
        for command in commands:
            reference_mol_ids.update(
                _reference_mol_ids(command.output_dir / "reference_mols")
            )
        candidate_comp_ids = reference_mol_ids - allowed_reference_mol_ids
        allowed_reference_mol_ids.update(
            collect_atom_site_comp_ids(
                cif_paths,
                candidate_comp_ids=candidate_comp_ids,
            )
        )
    merge_report = merge_chunk_outputs(
        chunk_output_dirs=[command.output_dir for command in commands],
        output_dir=output_dir,
        protein_only_reference_mols=protein_only_reference_mols,
        allowed_reference_mol_ids=allowed_reference_mol_ids,
    )
    report = {
        "status": "passed",
        "cif_dir": str(cif_dir),
        "output_dir": str(output_dir),
        "max_cifs_per_chunk": max_cifs_per_chunk,
        "num_workers_per_chunk": num_workers_per_chunk,
        "max_polymer_chains": max_polymer_chains,
        "preprocess_chunksize": preprocess_chunksize,
        "fallback_max_cifs_per_chunk": fallback_max_cifs_per_chunk,
        "protein_only_reference_mols": protein_only_reference_mols,
        "chunk_timeout_seconds": chunk_timeout_seconds,
        "input_cif_count": len(cif_paths),
        "chunk_count": len(chunks),
        "chunks": chunk_results,
        "merge": merge_report,
    }
    (output_dir / "chunked_preprocess_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif-dir", type=Path, required=True)
    parser.add_argument("--ccd-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-cifs-per-chunk", type=int, default=128)
    parser.add_argument("--num-workers-per-chunk", type=int, default=8)
    parser.add_argument("--max-polymer-chains", type=int, default=300)
    parser.add_argument("--preprocess-chunksize", type=int, default=8)
    parser.add_argument("--fallback-max-cifs-per-chunk", type=int, default=None)
    parser.add_argument("--chunk-timeout-seconds", type=int, default=None)
    parser.add_argument("--force-chunk", action="store_true")
    parser.add_argument("--protein-only-reference-mols", action="store_true")
    args = parser.parse_args()

    report = run_chunked_preprocess(
        cif_dir=args.cif_dir,
        ccd_path=args.ccd_path,
        output_dir=args.out_dir,
        max_cifs_per_chunk=args.max_cifs_per_chunk,
        num_workers_per_chunk=args.num_workers_per_chunk,
        max_polymer_chains=args.max_polymer_chains,
        preprocess_chunksize=args.preprocess_chunksize,
        fallback_max_cifs_per_chunk=args.fallback_max_cifs_per_chunk,
        force_chunk=args.force_chunk,
        protein_only_reference_mols=args.protein_only_reference_mols,
        chunk_timeout_seconds=args.chunk_timeout_seconds,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
