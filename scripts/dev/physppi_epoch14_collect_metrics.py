#!/usr/bin/env python3
"""Collect OF3 confidence metrics and DockQ for epoch14 evaluation outputs."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _prediction_path(run_dir: Path, sample_id: str, seed: int) -> Path:
    return (
        run_dir
        / "predictions"
        / sample_id
        / f"seed_{seed}"
        / f"{sample_id}_seed_{seed}_sample_1_model.cif"
    )


def _confidence_path(run_dir: Path, sample_id: str, seed: int) -> Path:
    return (
        run_dir
        / "predictions"
        / sample_id
        / f"seed_{seed}"
        / f"{sample_id}_seed_{seed}_sample_1_confidences_aggregated.json"
    )


def _native_path(run_dir: Path, sample_id: str) -> Path:
    return run_dir / "native" / f"{sample_id}_native.cif"


def _flatten_confidence(conf: dict[str, Any]) -> dict[str, Any]:
    return {
        key: conf.get(key)
        for key in [
            "avg_plddt",
            "gpde",
            "iptm",
            "ptm",
            "disorder",
            "has_clash",
            "sample_ranking_score",
        ]
    }


def _run_dockq(
    dockq_bin: str,
    model: Path,
    native: Path,
    raw_json: Path,
) -> dict[str, Any]:
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            dockq_bin,
            "--short",
            "--json",
            str(raw_json),
            str(model),
            str(native),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "dockq_status": "error",
            "dockq_error": proc.stderr.strip()[-2000:],
            "dockq_stdout": proc.stdout.strip()[-2000:],
        }
    payload = _load_json(raw_json)
    best = payload.get("best_result", {})
    interface_values = list(best.values()) if isinstance(best, dict) else []

    def mean_field(key: str) -> float | None:
        values = [
            v.get(key) for v in interface_values if isinstance(v.get(key), (int, float))
        ]
        return round(statistics.fmean(values), 6) if values else None

    interface_dockqs = [
        v.get("DockQ")
        for v in interface_values
        if isinstance(v.get("DockQ"), (int, float))
    ]
    return {
        "dockq_status": "ok",
        "dockq": payload.get("GlobalDockQ")
        if isinstance(payload.get("GlobalDockQ"), (int, float))
        else (max(interface_dockqs) if interface_dockqs else None),
        "global_dockq": payload.get("GlobalDockQ"),
        "dockq_sum": payload.get("best_dockq"),
        "best_interface_dockq": max(interface_dockqs) if interface_dockqs else None,
        "mean_interface_dockq": mean_field("DockQ"),
        "dockq_mapping": payload.get("best_mapping_str"),
        "irmsd": mean_field("iRMSD"),
        "lrmsd": mean_field("LRMSD"),
        "fnat": mean_field("fnat"),
        "f1": mean_field("F1"),
        "dockq_clashes": mean_field("clashes"),
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 6) if values else None


def _numeric(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "count": len(records),
        "dockq_ok": sum(1 for r in records if r.get("dockq_status") == "ok"),
        "dockq_error": sum(1 for r in records if r.get("dockq_status") != "ok"),
    }
    for key in [
        "dockq",
        "irmsd",
        "lrmsd",
        "fnat",
        "f1",
        "avg_plddt",
        "iptm",
        "ptm",
        "sample_ranking_score",
    ]:
        values = _numeric(records, key)
        summary[f"{key}_mean"] = _mean(values)
        summary[f"{key}_median"] = _median(values)
    return summary


def collect_run(
    *,
    sample_ids: list[str],
    run_dir: Path,
    output_dir: Path,
    split: str,
    variant: str,
    seed: int,
    dockq_bin: str,
) -> dict[str, Any]:
    records = []
    raw_dir = output_dir / "dockq_raw" / split / variant
    for sample_id in sample_ids:
        pred = _prediction_path(run_dir, sample_id, seed)
        native = _native_path(run_dir, sample_id)
        conf_path = _confidence_path(run_dir, sample_id, seed)
        record = {
            "split": split,
            "variant": variant,
            "sample_id": sample_id,
            "prediction": str(pred),
            "native": str(native),
        }
        missing = [
            name
            for name, path in [
                ("prediction", pred),
                ("native", native),
                ("confidence", conf_path),
            ]
            if not path.is_file()
        ]
        if missing:
            record["status"] = "missing"
            record["missing"] = missing
        else:
            record["status"] = "ok"
            record.update(_flatten_confidence(_load_json(conf_path)))
            record.update(
                _run_dockq(
                    dockq_bin,
                    pred,
                    native,
                    raw_dir / f"{sample_id}.json",
                )
            )
        records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"{split}_{variant}_metrics.jsonl"
    records_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = _summarize(records)
    summary.update(
        {
            "split": split,
            "variant": variant,
            "run_dir": str(run_dir),
            "records_jsonl": str(records_path),
        }
    )
    return summary


def _write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = [
        "| Split | Variant | N | DockQ ok | DockQ mean | DockQ median | "
        "iRMSD mean | LRMSD mean | Fnat mean | avg pLDDT | ipTM | pTM |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        rows.append(
            "| {split} | {variant} | {count} | {dockq_ok} | {dockq_mean} | "
            "{dockq_median} | {irmsd_mean} | {lrmsd_mean} | {fnat_mean} | "
            "{avg_plddt_mean} | {iptm_mean} | {ptm_mean} |".format(**s)
        )
    path.write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dockq-bin", default="/101045/jinyu/.local/bin/DockQ")
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        metavar=("SPLIT", "VARIANT", "SAMPLE_IDS_JSON", "RUN_DIR"),
        required=True,
    )
    args = parser.parse_args()

    summaries = []
    for split, variant, sample_ids_json, run_dir in args.run:
        summaries.append(
            collect_run(
                sample_ids=[str(x) for x in _load_json(Path(sample_ids_json))],
                run_dir=Path(run_dir),
                output_dir=args.output_dir,
                split=split,
                variant=variant,
                seed=args.seed,
                dockq_bin=args.dockq_bin,
            )
        )
    summary_path = args.output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    _write_markdown(args.output_dir / "metrics_summary.md", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
