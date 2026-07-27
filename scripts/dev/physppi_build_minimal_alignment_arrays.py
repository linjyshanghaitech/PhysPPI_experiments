from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_rep_sequences(mapping_dir: Path) -> dict[str, str]:
    rep_to_seq_path = mapping_dir / "rep_id_to_seq.json"
    if not rep_to_seq_path.is_file():
        raise FileNotFoundError(
            f"missing ColabFold rep sequence map: {rep_to_seq_path}"
        )
    data = json.loads(rep_to_seq_path.read_text())
    reps = {str(rep_id): str(seq) for rep_id, seq in data.items()}
    if not reps:
        raise ValueError(f"empty ColabFold rep sequence map: {rep_to_seq_path}")
    return dict(sorted(reps.items()))


def write_alignment_representatives_fasta(
    rep_to_sequence: dict[str, str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for rep_id, sequence in sorted(rep_to_sequence.items()):
        lines.append(f">{rep_id}")
        lines.append(sequence)
    output_path.write_text("\n".join(lines) + "\n")


def copy_colabfold_main_npz(
    rep_to_sequence: dict[str, str],
    main_msa_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for rep_id in sorted(rep_to_sequence):
        source = main_msa_dir / f"{rep_id}.npz"
        if not source.is_file():
            raise FileNotFoundError(f"missing ColabFold main MSA npz: {source}")
        target = output_dir / f"{rep_id}.npz"
        shutil.copy2(source, target)
        copied[rep_id] = target
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--colabfold-dir",
        type=Path,
        default=Path("output/physppi_msa_cache/colabfold_msas"),
    )
    parser.add_argument(
        "--mapping-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/physppi_msa_cache"),
    )
    args = parser.parse_args()

    mapping_dir = args.mapping_dir or (args.colabfold_dir / "mappings")
    rep_to_sequence = load_rep_sequences(mapping_dir)
    fasta_path = args.output_dir / "alignment_representatives.fasta"
    alignment_array_dir = args.output_dir / "alignment_arrays"
    write_alignment_representatives_fasta(rep_to_sequence, fasta_path)
    copied = copy_colabfold_main_npz(
        rep_to_sequence=rep_to_sequence,
        main_msa_dir=args.colabfold_dir / "main",
        output_dir=alignment_array_dir,
    )
    report = {
        "alignment_representatives_fasta": str(fasta_path),
        "alignment_array_directory": str(alignment_array_dir),
        "representatives": rep_to_sequence,
        "copied_npz": {rep_id: str(path) for rep_id, path in copied.items()},
    }
    report_path = args.output_dir / "alignment_array_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
