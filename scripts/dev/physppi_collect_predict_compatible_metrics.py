from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = {
    "R0-official": {
        "path": (
            "output/physppi_baseline_first/official_predict_32x32/"
            "run_20260615_024151/metrics/metrics_summary.md"
        ),
        "include_variants": {"official_predict"},
        "status": "accepted_official_predict",
    },
    "R1-A1-existing": {
        "path": (
            "output/physppi_ablation_15epoch_deployable/"
            "eval_a1_15epoch_32x32/run_20260614_185646/"
            "metrics/metrics_summary.md"
        ),
        "include_variants": {"a1_epoch15"},
        "status": "accepted_physppi_row_embedded_official_rows_invalid",
    },
    "R2-L1a-existing": {
        "path": (
            "output/physppi_lora_15epoch/L1a_eval_32x32/"
            "run_20260614_180310/metrics/metrics_summary.md"
        ),
        "include_variants": {"l1a"},
        "status": "accepted_physppi_row_embedded_official_rows_invalid",
    },
    "R3-seed43-existing": {
        "path": (
            "output/physppi_baseline_first/adapter_only_seed43/"
            "eval_32x32/run_20260616_112306/metrics/metrics_summary.md"
        ),
        "include_variants": {"adapter_seed43"},
        "status": "accepted_physppi_row",
    },
}


def _to_number(value: str):
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_markdown_table(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.startswith("|") or "---" in line or "DockQ mean" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 12:
            continue
        rows.append(
            {
                "split": cells[0],
                "variant": cells[1],
                "n": _to_number(cells[2]),
                "dockq_ok": _to_number(cells[3]),
                "dockq_mean": _to_number(cells[4]),
                "dockq_median": _to_number(cells[5]),
                "irmsd_mean": _to_number(cells[6]),
                "lrmsd_mean": _to_number(cells[7]),
                "fnat_mean": _to_number(cells[8]),
                "avg_plddt": _to_number(cells[9]),
                "iptm": _to_number(cells[10]),
                "ptm": _to_number(cells[11]),
                "source": str(path),
            }
        )
    return rows


def collect() -> list[dict]:
    output = []
    for run_name, cfg in RUNS.items():
        path = Path(cfg["path"])
        parsed = _parse_markdown_table(path)
        invalid_embedded = [
            row for row in parsed if row["variant"] == "official"
        ]
        for row in parsed:
            if row["variant"] not in cfg["include_variants"]:
                continue
            row["run"] = run_name
            row["status"] = cfg["status"]
            row["invalid_embedded_official_rows_in_source"] = len(
                invalid_embedded
            )
            output.append(row)
        if not path.exists():
            output.append(
                {
                    "run": run_name,
                    "status": "missing_metrics_summary",
                    "source": str(path),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows, indent=2, sort_keys=True))

    has_official_test = any(
        row.get("run") == "R0-official"
        and row.get("split") == "test"
        and row.get("variant") == "official_predict"
        for row in rows
    )
    if not has_official_test:
        raise SystemExit("METRICS_NOT_READY: official predict test row missing")


if __name__ == "__main__":
    main()
