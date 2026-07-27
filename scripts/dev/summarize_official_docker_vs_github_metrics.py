#!/usr/bin/env python3
"""Summarize Docker-code vs GitHub-original official checkpoint metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeAlias

MetricRows: TypeAlias = dict[tuple[str, str], dict[str, Any]]


KEYS = [
    "dockq_mean",
    "dockq_median",
    "irmsd_mean",
    "lrmsd_mean",
    "fnat_mean",
    "f1_mean",
    "avg_plddt_mean",
    "iptm_mean",
    "ptm_mean",
    "sample_ranking_score_mean",
]


def load_summary(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def by_split_variant(rows: list[dict[str, Any]]) -> MetricRows:
    return {(str(row["split"]), str(row["variant"])): row for row in rows}


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "null"
    return str(value)


def better_decision(rows: MetricRows) -> str:
    decisions = []
    for split in ["train", "test"]:
        docker = rows.get((split, "docker_code"))
        github = rows.get((split, "github_original"))
        if docker is None or github is None:
            return "NOT_ACCEPTED: missing split/variant metrics"
        if docker.get("count") != 32 or github.get("count") != 32:
            return "NOT_ACCEPTED: expected 32 predictions per split/variant"
        if docker.get("dockq_ok") != 32 or github.get("dockq_ok") != 32:
            return "NOT_ACCEPTED: expected DockQ ok for all samples"
        delta = (docker.get("dockq_mean") or 0.0) - (github.get("dockq_mean") or 0.0)
        decisions.append(delta)

    # Require consistent test improvement before allowing migration by metrics.
    if decisions[1] > 0.02 and decisions[0] >= -0.02:
        return "DOCKER_CODE_MATERIALLY_BETTER: eligible for isolated PhysPPI migration"
    if abs(decisions[1]) <= 0.02 and abs(decisions[0]) <= 0.02:
        return "SIMILAR: do not migrate solely for metrics"
    return "GITHUB_OR_DOCKER_NOT_BETTER: do not migrate"


def write_markdown(
    path: Path,
    rows: MetricRows,
    decision: str,
) -> None:
    lines = [
        "# Official Docker Code vs GitHub Original Metrics Comparison",
        "",
        f"Decision: `{decision}`",
        "",
        "## Summary",
        "",
        "| Split | Metric | Docker code | GitHub original | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for split in ["train", "test"]:
        docker = rows.get((split, "docker_code"), {})
        github = rows.get((split, "github_original"), {})
        for key in KEYS:
            d_val = docker.get(key)
            g_val = github.get(key)
            delta = (
                round(float(d_val) - float(g_val), 6)
                if isinstance(d_val, (int, float)) and isinstance(g_val, (int, float))
                else None
            )
            lines.append(
                f"| {split} | {key} | {fmt(d_val)} | {fmt(g_val)} | {fmt(delta)} |"
            )

    lines.extend(
        [
            "",
            "## Acceptance Notes",
            "",
            "- Metrics are valid only if all four inference runs completed "
            "with 32 predictions.",
            "- Migration is allowed only after Docker code is materially better "
            "on the fixed test subset.",
            "- If metrics are similar, keep the current codebase and do not "
            "migrate just because source diffs exist.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-summary",
        type=Path,
        default=Path(
            "output/openfold3_official_docker_migration/"
            "comparison_metrics/metrics_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/openfold3_official_docker_migration/reports/"
            "docker_vs_github_metrics_comparison.md"
        ),
    )
    args = parser.parse_args()

    rows = by_split_variant(load_summary(args.metrics_summary))
    decision = better_decision(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.output, rows, decision)
    print(json.dumps({"decision": decision, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
