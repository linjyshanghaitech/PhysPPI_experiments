#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NON_INFERIOR_MARGIN = 0.01
LORA_WIN_MARGIN = 0.03
UNDERPERFORM_MARGIN = 0.02


def decide_next_step(
    *,
    official_test_mean: float,
    official_test_median: float,
    adapter_test_mean: float,
    adapter_test_median: float,
    lora_test_mean: float,
    lora_test_median: float,
) -> dict[str, Any]:
    adapter_non_inferior = (
        adapter_test_mean >= official_test_mean - NON_INFERIOR_MARGIN
    )
    lora_beats_adapter = (
        lora_test_mean >= adapter_test_mean + LORA_WIN_MARGIN
        and lora_test_median >= adapter_test_median
    )
    all_physppi_underperform = (
        adapter_test_mean <= official_test_mean - UNDERPERFORM_MARGIN
        and lora_test_mean <= official_test_mean - UNDERPERFORM_MARGIN
        and adapter_test_median < official_test_median
        and lora_test_median < official_test_median
    )
    if lora_beats_adapter:
        decision = "WRITE_LORA_CONFIRMATION_PLAN"
        allow_training = False
        allow_lora_expansion = True
        rationale = (
            "LoRA won broad heldout by mean margin and did not lose median."
        )
    elif adapter_non_inferior:
        decision = "TRAIN_ADAPTER_REPLICATION"
        allow_training = True
        allow_lora_expansion = False
        rationale = (
            "Adapter-only is broad-heldout non-inferior to official; "
            "replicate seeds before adding LoRA."
        )
    elif all_physppi_underperform:
        decision = "PAUSE_AND_DEBUG_PRIOR_COVERAGE"
        allow_training = False
        allow_lora_expansion = False
        rationale = (
            "All PhysPPI variants underperform official on broad heldout; "
            "inspect prior coverage and residual scale first."
        )
    else:
        decision = "RUN_BROADER_OR_MULTI_SEED_EVAL"
        allow_training = False
        allow_lora_expansion = False
        rationale = (
            "Broad result is ambiguous; reduce sampling/eval noise before "
            "training."
        )
    return {
        "decision": decision,
        "allow_training": allow_training,
        "allow_lora_expansion": allow_lora_expansion,
        "rationale": rationale,
        "thresholds": {
            "adapter_non_inferior_margin": NON_INFERIOR_MARGIN,
            "lora_win_margin": LORA_WIN_MARGIN,
            "underperform_margin": UNDERPERFORM_MARGIN,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    metrics = json.loads(args.metrics_json.read_text())
    decision = decide_next_step(
        official_test_mean=float(
            metrics["official_predict"]["test"]["dockq_mean"]
        ),
        official_test_median=float(
            metrics["official_predict"]["test"]["dockq_median"]
        ),
        adapter_test_mean=float(
            metrics["adapter_seed43"]["test"]["dockq_mean"]
        ),
        adapter_test_median=float(
            metrics["adapter_seed43"]["test"]["dockq_median"]
        ),
        lora_test_mean=float(metrics["l1a_lora"]["test"]["dockq_mean"]),
        lora_test_median=float(metrics["l1a_lora"]["test"]["dockq_median"]),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(decision, indent=2) + "\n")


if __name__ == "__main__":
    main()
