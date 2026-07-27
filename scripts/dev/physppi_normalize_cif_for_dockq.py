#!/usr/bin/env python3
"""Normalize OpenFold3 mmCIF outputs for DockQ parsing.

OpenFold3 prediction mmCIF files can omit _atom_site.occupancy. The DockQ
parser treats that as a parse failure, not as a structure-quality failure.
This script adds DockQ-required atom-site fields without changing coordinates.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _is_tag_line(line: str) -> bool:
    return line.startswith("_")


def _is_loop_data_line(line: str) -> bool:
    return bool(line.strip()) and not line.startswith(("#", "loop_", "_", "data_"))


def _insertion_index(tags: list[str], preferred_after: str) -> int:
    if preferred_after in tags:
        return tags.index(preferred_after) + 1
    for fallback in [
        "_atom_site.pdbx_formal_charge",
        "_atom_site.Cartn_x",
        "_atom_site.pdbx_PDB_model_num",
        "_atom_site.id",
    ]:
        if fallback in tags:
            return tags.index(fallback)
    return len(tags)


def normalize_cif_text(text: str) -> str:
    """Return mmCIF text with DockQ-required atom-site fields present."""
    lines = text.splitlines()
    out: list[str] = []
    idx = 0
    changed = False

    while idx < len(lines):
        line = lines[idx]
        if line.strip() != "loop_":
            out.append(line)
            idx += 1
            continue

        loop_start = line
        tag_idx = idx + 1
        tags: list[str] = []
        while tag_idx < len(lines) and _is_tag_line(lines[tag_idx].strip()):
            tags.append(lines[tag_idx].strip())
            tag_idx += 1

        if not tags or not all(tag.startswith("_atom_site.") for tag in tags):
            out.append(loop_start)
            out.extend(lines[idx + 1 : tag_idx])
            idx = tag_idx
            continue

        insertions: list[tuple[int, str, str]] = []
        working_tags = list(tags)
        if "_atom_site.B_iso_or_equiv" not in working_tags:
            pos = _insertion_index(working_tags, "_atom_site.auth_atom_id")
            insertions.append((pos, "_atom_site.B_iso_or_equiv", "0.0"))
            working_tags.insert(pos, "_atom_site.B_iso_or_equiv")
        if "_atom_site.occupancy" not in working_tags:
            pos = _insertion_index(working_tags, "_atom_site.B_iso_or_equiv")
            insertions.append((pos, "_atom_site.occupancy", "1.0"))
            working_tags.insert(pos, "_atom_site.occupancy")

        out.append(loop_start)
        if not insertions:
            out.extend(lines[idx + 1 : tag_idx])
        else:
            changed = True
            out.extend(working_tags)

        data_idx = tag_idx
        while data_idx < len(lines) and _is_loop_data_line(lines[data_idx].strip()):
            data_line = lines[data_idx].strip()
            if insertions and data_line.startswith(("ATOM", "HETATM")):
                fields = data_line.split()
                if len(fields) == len(tags):
                    for pos, _tag, value in sorted(insertions, key=lambda x: x[0]):
                        fields.insert(pos, value)
                    out.append(" ".join(fields))
                elif len(fields) == len(working_tags):
                    out.append(" ".join(fields))
                else:
                    raise ValueError(
                        "atom-site row has unexpected field count: "
                        f"{len(fields)} fields for {len(tags)} tags"
                    )
            else:
                out.append(lines[data_idx])
            data_idx += 1

        idx = data_idx

    normalized = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    return normalized if changed else text


def normalize_file(model_cif: Path, output_cif: Path) -> None:
    output_cif.parent.mkdir(parents=True, exist_ok=True)
    output_cif.write_text(normalize_cif_text(model_cif.read_text()))


def normalize_run(
    *,
    input_run: Path,
    output_run: Path,
    sample_ids: list[str],
    seed: int,
) -> None:
    output_run.mkdir(parents=True, exist_ok=True)
    for sample_id in sample_ids:
        src_seed = input_run / sample_id / f"seed_{seed}"
        dst_seed = output_run / sample_id / f"seed_{seed}"
        dst_seed.mkdir(parents=True, exist_ok=True)

        src_model = src_seed / f"{sample_id}_seed_{seed}_sample_1_model.cif"
        dst_model = dst_seed / src_model.name
        normalize_file(src_model, dst_model)

        for suffix in [
            "confidences_aggregated.json",
            "confidences.json",
            "timing.json",
        ]:
            src = src_seed / f"{sample_id}_seed_{seed}_sample_1_{suffix}"
            if suffix == "timing.json":
                src = src_seed / suffix
            if src.is_file():
                dst = dst_seed / src.name
                shutil.copy2(src, dst)

    (output_run / "predictions").unlink(missing_ok=True)
    (output_run / "predictions").symlink_to(".")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cif", type=Path)
    parser.add_argument("--output-cif", type=Path)
    parser.add_argument("--input-run", type=Path)
    parser.add_argument("--output-run", type=Path)
    parser.add_argument("--sample-ids-json", type=Path)
    parser.add_argument("--seed", type=int, default=2746317213)
    args = parser.parse_args()

    if args.model_cif or args.output_cif:
        if not args.model_cif or not args.output_cif:
            raise SystemExit("--model-cif and --output-cif must be provided together")
        normalize_file(args.model_cif, args.output_cif)
        return

    missing = [
        name
        for name, value in [
            ("--input-run", args.input_run),
            ("--output-run", args.output_run),
            ("--sample-ids-json", args.sample_ids_json),
        ]
        if value is None
    ]
    if missing:
        raise SystemExit(
            f"missing required arguments for run mode: {', '.join(missing)}"
        )

    sample_ids = [str(x) for x in json.loads(args.sample_ids_json.read_text())]
    normalize_run(
        input_run=args.input_run,
        output_run=args.output_run,
        sample_ids=sample_ids,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
