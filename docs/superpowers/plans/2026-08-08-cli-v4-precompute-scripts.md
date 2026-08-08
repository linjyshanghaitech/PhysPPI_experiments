# cli_v4 预计算脚本 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 PhysPPI cli_v4 编写 msa / pdb_cache / residue_prior / split_training_val 预计算脚本和文档（mesh_prior 已由师姐实现，只补 HANDOFF），不运行任何生成。

**Architecture:** 每个生成脚本沿用师姐 `generate_mesh_tensors_v4.py` 的 CLI 与输出约定（`--dataset-dir` / `--output-root` / `--dataset-name` / `--chunks` / `--max-samples` / `--num-workers` / `--force`），输出到 `{output_root}/{dataset_name}/{chunk}/{case}_{suffix}.pt`，数据集根目录写独立 summary JSON；并行用 `multiprocessing.Pool` + `imap_unordered(chunksize=4)`。脚本不 import `_runtime.py`，不写 adapter。

**Tech Stack:** Python 3.10、numpy、torch、biotite、openfold3 core data 模块、MMseqs2（split 部分）、PhysPPI conda 环境 `/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python`。

---

## 任务总览

| 任务 | 内容 | 产出 |
|---|---|---|
| 1 | 确认目录与依赖 | `experiments_cli_v4/` 5 个文件夹 |
| 2 | msa 生成脚本 + 文档 | `msa/generate_msa_tensors_v4.py`、`README.md`、`HANDOFF20260808.md` |
| 3 | pdb_cache 生成脚本 + 文档 | `pdb_cache/generate_pdb_tensors_v4.py`、`README.md`、`HANDOFF20260808.md` |
| 4 | residue_prior 生成脚本 + 文档 | `residue_prior/generate_residue_tensors_v4.py`、`README.md`、`HANDOFF20260808.md` |
| 5 | split_training_val 脚本 + 文档 | `extract_sequences_v4.py`、`run_mmseqs_v4.py`、`build_split_v4.py`、`run_all_v4.py`、`README.md`、`HANDOFF20260808.md` |
| 6 | mesh_prior HANDOFF | `mesh_prior/HANDOFF20260808.md` |
| 7 | 全量验证与提交 | `py_compile` 全部通过 + git commit |

---

### Task 1: 确认目录与参考文件

**Files:**
- Read: `experiments_cli_v4/mesh_prior/generate_mesh_tensors_v4.py`（师姐脚本，CLI 基准）
- Read: `experiments/msa/generate_msa_tensors.py`
- Read: `experiments/pdb_cache/generate_pdb_tensors.py`
- Read: `experiments/residue_prior/generate_residue_tensors.py`
- Read: `experiments/split_training_val/{extract_sequences,run_mmseqs,build_split,run_all}.py`

- [ ] **Step 1: 确认目录存在**

```bash
ls -d /101045/jinyu/PhysPPI_jinyu/PhysPPI/experiments_cli_v4/{mesh_prior,msa,pdb_cache,residue_prior,split_training_val}
```

Expected: 5 个目录都存在，其中 `mesh_prior/` 已有 `generate_mesh_tensors_v4.py` 和 `README.md`，其余为空。

- [ ] **Step 2: 确认输出根目录存在（不写入）**

```bash
ls -d /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4
```

Expected: 目录存在。

---

### Task 2: msa 生成脚本 + 文档

**Files:**
- Create: `experiments_cli_v4/msa/generate_msa_tensors_v4.py`
- Create: `experiments_cli_v4/msa/README.md`
- Create: `experiments_cli_v4/msa/HANDOFF20260808.md`

- [ ] **Step 1: 写 `generate_msa_tensors_v4.py`**

```python
#!/usr/bin/env python3
"""Generate pre-computed MSA tensors for the v4 prior pipeline.

Reads msa_local/{main,paired}/{case}/{case}.npz, projects MSA columns onto GT
residues, computes profile/deletion_mean, and writes {case}_msa.pt.

Output layout mirrors the v4 mesh generator:
    {output_root}/{dataset_name}/{chunk_name}/{case_id}_msa.pt
    {output_root}/{dataset_name}/msa_generation_summary.json

The npz files in v4 are stored under msa_local/{main,paired}/{case}/ and their
internal keys may carry unpaired/ or paired/ prefixes (e.g.
"unpaired/1abc_1" or "paired/1abc_1"); parse_msas_preparsed handles them and
this script consumes the parsed MsaArray values directly.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openfold3.core.data.io.sequence.msa import parse_msas_preparsed
from openfold3.core.data.primitives.sequence.msa import MsaArray
from openfold3.core.data.pipelines.sample_processing.msa import calculate_profile
from openfold3.core.data.resources.residues import (
    STANDARD_RESIDUES_WITH_GAP_1,
    MoleculeType,
)

MAX_ROWS = 16384
MAX_ROWS_PAIRED = 8192

_RESIDUES_ARR = np.array(STANDARD_RESIDUES_WITH_GAP_1)
_CHAR_TO_IDX: dict[str, int] = {}
for _i, _r in enumerate(STANDARD_RESIDUES_WITH_GAP_1):
    if _r not in _CHAR_TO_IDX:
        _CHAR_TO_IDX[_r] = _i
_GAP_CHAR = "-"

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V",
}
_STANDARD_1 = set("ACDEFGHIKLMNPQRSTVWYX")


def _parse_gt_chain_data(gt_pdb: Path) -> dict[str, dict]:
    chain_atoms: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    with gt_pdb.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 27:
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resname = line[17:20].strip().upper()
            residue_1 = AA3_TO_1.get(resname)
            if residue_1 is None:
                continue
            chain_id = line[21].strip() or "_"
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            insertion = line[26].strip()
            if resseq not in chain_atoms[chain_id]:
                chain_atoms[chain_id][resseq] = (residue_1, insertion)

    result: dict[str, dict] = {}
    for chain_id in sorted(chain_atoms):
        residues = sorted(chain_atoms[chain_id].items())
        sequence = "".join(info[0] for _, info in residues)
        resseq_list = [r for r, _ in residues]
        if sequence and all(c in _STANDARD_1 for c in sequence):
            result[chain_id] = {"sequence": sequence, "resseq": resseq_list}
    return result


def _msa_to_chars(msa: np.ndarray) -> np.ndarray:
    if msa.dtype == np.dtype("<U1") or msa.dtype.kind == "U":
        return msa
    if msa.dtype == np.uint8 or msa.dtype.kind in ("i", "u"):
        return _RESIDUES_ARR[msa.astype(int)]
    return msa.astype(str).astype("<U1")


def _read_msa_npz_chars(path: Path, max_rows: int | None = None) -> MsaArray:
    parsed = parse_msas_preparsed([path])
    if not parsed:
        raise ValueError(f"MSA npz does not contain any arrays: {path}")
    arrays = list(parsed.values())
    if len(arrays) == 1:
        result = arrays[0]
    else:
        result = MsaArray.multi_concatenate(arrays, axis=0)

    if max_rows is not None and len(result) > max_rows:
        result = result.truncate(max_rows)
        result.msa = result.msa.copy()

    if result.deletion_matrix.dtype == np.int64:
        result.deletion_matrix = result.deletion_matrix.astype(np.int32)
    result.msa = _msa_to_chars(result.msa)
    return result


def _msa_query_sequence(msa: MsaArray) -> str:
    if len(msa) == 0:
        return ""
    row = np.asarray(msa.msa[0])
    if row.dtype.kind == "U":
        return "".join(row.astype(str).tolist())
    return "".join(_RESIDUES_ARR[row.astype(int)].tolist())


def _project_msa_to_gt_residues(msa: MsaArray, gt_sequence: str) -> MsaArray | None:
    source = np.asarray(msa.msa)
    if source.ndim != 2 or source.shape[0] == 0 or not gt_sequence:
        return None
    if np.asarray(msa.deletion_matrix).shape != source.shape:
        return None

    source_query_chars = _msa_to_chars(source[0:1])
    source_columns = np.flatnonzero(source_query_chars[0] != _GAP_CHAR)
    source_query = "".join(source_query_chars[0, source_columns].astype(str).tolist())
    if not source_query:
        return None

    if source_query == gt_sequence and source.shape[1] == len(gt_sequence):
        return msa

    def _select_columns(allow_unknown: bool) -> list[int] | None:
        selected: list[int] = []
        target_index = 0
        for column in source_columns:
            if target_index == len(gt_sequence):
                break
            residue = str(_msa_to_chars(source[0:1, column:column + 1])[0, 0])
            target_residue = gt_sequence[target_index]
            if residue == target_residue or (allow_unknown and target_residue == "X"):
                selected.append(int(column))
                target_index += 1
        return selected if target_index == len(gt_sequence) else None

    selected_columns = _select_columns(allow_unknown=False)
    if selected_columns is None:
        selected_columns = _select_columns(allow_unknown=True)
    if selected_columns is None:
        return None

    selected = np.asarray(selected_columns, dtype=int)
    return MsaArray(
        msa=source[:, selected],
        deletion_matrix=np.asarray(msa.deletion_matrix)[:, selected],
        metadata=msa.metadata,
    )


def _find_main_npz(msa_main_dir: Path, case_id: str) -> dict[str, Path]:
    case_dir = msa_main_dir / case_id
    if case_dir.is_dir():
        search_dir = case_dir
        glob_pattern = "*.npz"
    else:
        search_dir = msa_main_dir
        glob_pattern = f"{case_id}_*.npz"

    result: dict[str, Path] = {}
    for f in sorted(search_dir.glob(glob_pattern)):
        stem = f.stem
        if stem == case_id:
            result[case_id] = f
        elif stem.startswith(case_id + "_"):
            suffix = stem[len(case_id) + 1:]
            if "chain=" in suffix:
                chain = suffix.split("chain=")[-1]
                result[chain] = f
                continue
            parts = suffix.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) <= 3 and parts[1].isalpha():
                chain = parts[1]
            elif len(parts) == 1 and len(parts[0]) <= 3 and parts[0].isalpha():
                chain = parts[0]
            else:
                chain = suffix
            result[chain] = f
    return result


def _find_paired_npz(msa_paired_dir: Path, case_id: str) -> dict[str, Path]:
    return _find_main_npz(msa_paired_dir, case_id)


def _process_one_sample_data(
    case_id: str,
    gt_pdb: Path,
    msa_main_dir: Path,
    msa_paired_dir: Path,
) -> dict | None:
    chain_data = _parse_gt_chain_data(gt_pdb)
    if not chain_data:
        return None
    chain_ids = sorted(chain_data)

    main_files = _find_main_npz(msa_main_dir, case_id)
    paired_files = _find_paired_npz(msa_paired_dir, case_id)

    main_msa: dict[str, np.ndarray] = {}
    main_deletion: dict[str, np.ndarray] = {}
    main_loaded: dict[str, MsaArray] = {}

    for chain_id in chain_ids:
        npz_path = main_files.get(chain_id, main_files.get(case_id))
        if npz_path is None:
            continue
        try:
            msa = _read_msa_npz_chars(npz_path, max_rows=MAX_ROWS)
        except Exception:
            continue
        projected = _project_msa_to_gt_residues(msa, chain_data[chain_id]["sequence"])
        if projected is None:
            n_res = len(chain_data[chain_id]["sequence"])
            seq = chain_data[chain_id]["sequence"]
            projected = MsaArray(
                msa=np.array([list(seq)], dtype="<U1"),
                deletion_matrix=np.zeros((1, n_res), dtype=np.int32),
                metadata=[],
            )
        main_loaded[chain_id] = projected
        main_msa[chain_id] = np.asarray(projected.msa)
        main_deletion[chain_id] = np.asarray(projected.deletion_matrix)

    if not main_msa:
        return None

    paired_msa: dict[str, np.ndarray] = {}
    paired_deletion: dict[str, np.ndarray] = {}
    for chain_id in chain_ids:
        npz_path = paired_files.get(chain_id, paired_files.get(case_id))
        if npz_path is None:
            continue
        try:
            msa = _read_msa_npz_chars(npz_path, max_rows=MAX_ROWS_PAIRED)
        except Exception:
            continue
        if msa.msa.shape[0] <= 1 and msa.msa.shape[1] <= 1:
            continue
        projected = _project_msa_to_gt_residues(msa, chain_data[chain_id]["sequence"])
        if projected is None:
            continue
        paired_msa[chain_id] = np.asarray(projected.msa)
        paired_deletion[chain_id] = np.asarray(projected.deletion_matrix)

    profile: dict[str, np.ndarray] = {}
    deletion_mean: dict[str, np.ndarray] = {}
    for chain_id in chain_ids:
        if chain_id in main_loaded:
            profile[chain_id] = calculate_profile(
                msa_array=main_loaded[chain_id].msa,
                molecule_type=MoleculeType.PROTEIN,
                chunk_size=1000,
            ).astype(np.float32)
            deletion_mean[chain_id] = np.mean(
                main_loaded[chain_id].deletion_matrix, axis=0
            ).astype(np.float32)
        else:
            n_res = len(chain_data[chain_id]["sequence"])
            profile[chain_id] = np.zeros((n_res, 32), dtype=np.float32)
            deletion_mean[chain_id] = np.zeros(n_res, dtype=np.float32)

    return {
        "chain_ids": chain_ids,
        "main_msa": main_msa,
        "main_deletion": main_deletion,
        "paired_msa": paired_msa,
        "paired_deletion": paired_deletion,
        "profile": profile,
        "deletion_mean": deletion_mean,
        "resseq": {c: chain_data[c]["resseq"] for c in chain_ids},
    }


def _process_one_sample(args: tuple) -> dict[str, Any]:
    case_id, gt_pdb, msa_main_dir, msa_paired_dir, output_path, force = args
    if output_path.exists() and not force:
        return {"case_id": case_id, "status": "exists", "error": None}
    try:
        data = _process_one_sample_data(case_id, gt_pdb, msa_main_dir, msa_paired_dir)
    except Exception as exc:
        return {"case_id": case_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if data is None:
        return {"case_id": case_id, "status": "skipped", "error": None}
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(data, tmp_path)
        tmp_path.replace(output_path)
    except Exception as exc:
        return {"case_id": case_id, "status": "error", "error": f"save failed: {exc}"}
    finally:
        del data
        gc.collect()
    return {"case_id": case_id, "status": "ok", "error": None}


def collect_tasks(
    chunk_dir: Path,
    output_chunk_dir: Path,
    max_samples: int | None,
    force: bool,
) -> list[tuple]:
    msa_main_dir = chunk_dir / "msa_local" / "main"
    msa_paired_dir = chunk_dir / "msa_local" / "paired"
    gt_dir = chunk_dir / "gt"
    if not msa_main_dir.is_dir():
        return []

    case_ids = sorted({d.name for d in msa_main_dir.iterdir() if d.is_dir()})
    if not case_ids:
        case_ids = sorted({f.stem for f in gt_dir.glob("*.pdb") if f.is_file()})
    if max_samples is not None:
        case_ids = case_ids[:max_samples]

    tasks = []
    for case_id in case_ids:
        gt_pdb = gt_dir / f"{case_id}.pdb"
        if not gt_pdb.is_file():
            continue
        output_path = output_chunk_dir / f"{case_id}_msa.pt"
        tasks.append(
            (case_id, gt_pdb, msa_main_dir, msa_paired_dir, output_path, force)
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pre-computed MSA .pt tensors for cli_v4."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4"),
    )
    parser.add_argument("--dataset-name", type=str, default="AFDB-10k-1")
    parser.add_argument("--chunks", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    chunk_parent = args.dataset_dir / "chunk"
    if chunk_parent.is_dir():
        all_chunk_dirs = sorted(d for d in chunk_parent.iterdir() if d.is_dir())
    else:
        all_chunk_dirs = sorted(d for d in args.dataset_dir.glob("chunk*") if d.is_dir())
    if not all_chunk_dirs:
        all_chunk_dirs = [args.dataset_dir]
    if args.chunks is not None:
        allowed = set(c.strip() for c in args.chunks.split(","))
        all_chunk_dirs = [d for d in all_chunk_dirs if d.name in allowed]

    output_root = args.output_root / args.dataset_name
    output_root.mkdir(parents=True, exist_ok=True)

    chunk_tasks: dict[str, list[tuple]] = {}
    for chunk_dir in all_chunk_dirs:
        if not (chunk_dir / "msa_local" / "main").is_dir():
            continue
        chunk_output = output_root / chunk_dir.name
        tasks = collect_tasks(chunk_dir, chunk_output, args.max_samples, args.force)
        if tasks:
            chunk_tasks[chunk_dir.name] = tasks

    total_tasks = sum(len(t) for t in chunk_tasks.values())
    print(f"Dataset: {args.dataset_dir}")
    print(f"Chunks: {len(chunk_tasks)}")
    print(f"Samples: {total_tasks}")
    print(f"Output: {output_root}")

    flat_tasks = [
        task + (chunk_name,)
        for chunk_name, tasks in chunk_tasks.items()
        for task in tasks
    ]

    def _run(task: tuple) -> dict[str, Any]:
        chunk_name = task[-1]
        result = _process_one_sample(task[:-1])
        result["chunk"] = chunk_name
        return result

    global_stats: dict[str, Any] = {"chunks": {}}
    for chunk_name in chunk_tasks:
        global_stats["chunks"][chunk_name] = {
            "chunk": chunk_name,
            "total": len(chunk_tasks[chunk_name]),
            "ok": 0,
            "exists": 0,
            "skipped": 0,
            "errors": 0,
            "error_list": [],
        }

    if args.num_workers <= 1:
        results = map(_run, flat_tasks)
    else:
        with Pool(processes=args.num_workers) as pool:
            results = pool.imap_unordered(_run, flat_tasks, chunksize=4)

    for result in results:
        stats = global_stats["chunks"][result["chunk"]]
        status = result["status"]
        stats[status if status in stats else "errors"] += 1
        if result.get("error"):
            stats["error_list"].append({"case_id": result["case_id"], "error": result["error"]})
            print(f"    SKIP {result['case_id']}: {result['error']}", file=sys.stderr)

    for chunk_name, stats in global_stats["chunks"].items():
        print(
            f"  {chunk_name}: {stats['ok']} ok, {stats['exists']} existing, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )

    summary_path = output_root / "msa_generation_summary.json"
    summary_path.write_text(json.dumps(global_stats, indent=2, default=str))
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法检查**

Run:
```bash
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m py_compile experiments_cli_v4/msa/generate_msa_tensors_v4.py
```

Expected: 退出码 0，无输出。

- [ ] **Step 3: 写 `README.md`**

```markdown
# msa (v4) — 预计算 MSA tensors

把 `msa_local/{main,paired}/{case}/{case}.npz` 投影到 GT 残基，计算 profile /
deletion_mean，输出 `{case}_msa.pt`，供训练 adapter 直接加载。

输出格式（与 v3 相同）：

```python
{
    "chain_ids": ["A", ...],
    "main_msa": {chain: np.ndarray[N_rows, N_res]},
    "main_deletion": {chain: np.ndarray[N_rows, N_res]},
    "paired_msa": {...},
    "paired_deletion": {...},
    "profile": {chain: np.ndarray[N_res, 32]},
    "deletion_mean": {chain: np.ndarray[N_res]},
    "resseq": {chain: [resseq, ...]},
}
```

用法：

```bash
python generate_msa_tensors_v4.py \
  --dataset-dir /101045/jinyu/PPI/Prior_out/training/cli_v4/AFDB-10k-1 \
  --output-root /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4 \
  --dataset-name AFDB-10k-1 \
  --chunks chunk0001 \
  --num-workers 8
```

参数：`--chunks`、`--max-samples`、`--num-workers`、`--force`（默认跳过已有）。

注意：
- AFDB 的 MSA 是 uint8 索引，Q-BioLiP 是 `<U1` char，脚本自动转换。
- MSA 行数先截断（main 16384 / paired 8192）再算 profile，避免 Q-BioLiP
  超大 MSA 占满内存。
- npz 内部 key 可能是 `unpaired/{case}` / `paired/{case}` 前缀，由
  `parse_msas_preparsed` 统一处理。
```

- [ ] **Step 4: 写 `HANDOFF20260808.md`**

```markdown
# MSA 预计算交接文档（v4）— 2026-08-08

## 状态

- `generate_msa_tensors_v4.py` 已写好，`py_compile` 通过。
- **未运行生成**（按用户要求只写脚本 + 文档）。
- Q-BioLiP 的 `msa_local` 用户后续补齐；脚本按 v4 完整布局编写，数据到位后即可运行。

## 与 v3 的差异

1. case id 带 `_G1` 等分组后缀，MSA 目录为 `msa_local/{main,paired}/{case_id}/`。
2. npz 内部 key 可能是 `unpaired/{原 case id}` / `paired/{原 case id}`，
   解析时直接用 `parse_msas_preparsed` 的 values。
3. CLI/输出布局对齐 mesh_prior 师姐脚本：
   `{output_root}/{dataset_name}/{chunk}/{case}_msa.pt`，
   summary 为 `msa_generation_summary.json`（与 mesh 的 `generation_summary.json` 分开）。

## 踩坑提醒

- Q-BioLiP MSA 很大（400K-870K 行），必须先在 `_read_msa_npz_chars` 里按
  `MAX_ROWS` 截断并 `copy()` 释放底层数组。
- `torch.save` 先写 `.tmp` 再 `replace`。
- 每个 sample 独立 try/except，错误进 summary 不中断。
```

- [ ] **Step 5: Commit**

```bash
git add experiments_cli_v4/msa/generate_msa_tensors_v4.py experiments_cli_v4/msa/README.md experiments_cli_v4/msa/HANDOFF20260808.md
git commit -m "feat(cli_v4): add msa precompute generator + docs"
```

---

### Task 3: pdb_cache 生成脚本 + 文档

**Files:**
- Create: `experiments_cli_v4/pdb_cache/generate_pdb_tensors_v4.py`
- Create: `experiments_cli_v4/pdb_cache/README.md`
- Create: `experiments_cli_v4/pdb_cache/HANDOFF20260808.md`

- [ ] **Step 1: 写 `generate_pdb_tensors_v4.py`**

```python
#!/usr/bin/env python3
"""Generate pre-computed PDB tensors for the v4 prior pipeline.

For each sample produces:
    {case_id}_gt.pt                — canonized GT AtomArray (all heavy atoms)
    {case_id}_{chain}_monomer.pt   — per-chain processed monomer AtomArray
    {case_id}_seq.pt               — per-chain sequence metadata + derived
                                     resseq_to_sequence_index

The v4 final chunk layout has no per-chain PDB files; monomers are split from
the all-atom complex PDB at chunk/input/{case_id}.pdb by chain.  H atoms are
removed during canonicalization; all heavy atoms are preserved.

Output layout:
    {output_root}/{dataset_name}/{chunk_name}/{case_id}_{suffix}.pt
    {output_root}/{dataset_name}/pdb_generation_summary.json
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PRIOR_SRC = Path("/101045/jinyu/PPI/Prior/src")
if str(_PRIOR_SRC) not in sys.path:
    sys.path.insert(0, str(_PRIOR_SRC))

import biotite.structure as struc  # noqa: E402
from biotite.structure.io import pdbx  # noqa: E402

from openfold3.core.data.io.structure.mol import (  # noqa: E402
    read_single_annotated_sdf,
)
from openfold3.core.data.io.structure.pdb import parse_pdb_af2  # noqa: E402
from openfold3.core.data.pipelines.sample_processing.structure import (  # noqa: E402
    process_target_atom_array_of3,
)
from openfold3.core.data.primitives.caches.format import (  # noqa: E402
    PreprocessingChainData,
)
from openfold3.core.data.primitives.structure.cleanup import (  # noqa: E402
    canonicalize_atom_order,
    convert_MSE_to_MET,
    remove_hydrogens,
    remove_non_CCD_atoms,
    remove_std_residue_terminal_atoms,
)
from openfold3.core.data.primitives.structure.tokenization import (  # noqa: E402
    add_token_positions,
)
from openfold3.core.data.resources.residues import MoleculeType  # noqa: E402
from experiments.pdb_cache.pdb_adapter import save_atomarray_as_pt  # noqa: E402

_PROTEIN_REFERENCE_MOLS = (
    _REPO_ROOT
    / "openfold3/core/data/resources/canonical_protein_data/reference_mols"
)

_AA3_TO_1: dict[str, str] = {}
try:
    from physppi_prior.aa_map import AA_MAP as _AA3_TO_1  # noqa: E402
except ImportError:
    _AA3_TO_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
        "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
        "VAL": "V",
    }

# Worker-global CCD (loaded once per worker, lazily)
_CCD_PATH: str | None = None
_CCD_CACHE: Any = None


@functools.lru_cache(maxsize=64)
def _canonical_atom_names(res_name: str) -> frozenset[str]:
    mol = read_single_annotated_sdf(
        _PROTEIN_REFERENCE_MOLS / f"{res_name}.sdf"
    )
    return frozenset(
        atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()
    )


def _parse_pdb(path: Path):
    parsed = parse_pdb_af2(path, allow_malformed_charge=True)
    if not hasattr(parsed, "atom_array"):
        raise ValueError(f"failed to parse PDB: {path}")
    atom_array = parsed.atom_array
    if len(atom_array) > 0 and "occupancy" in atom_array.get_annotation_categories():
        occupancy = np.asarray(atom_array.occupancy)
        if not np.any(occupancy > 0):
            atom_array.occupancy[:] = 1.0
    return atom_array


def canonize_atom_array(atom_array, *, ccd=None):
    aa = atom_array.copy()
    convert_MSE_to_MET(aa)
    aa = aa[aa.molecule_type_id == MoleculeType.PROTEIN]
    if len(aa) == 0:
        raise ValueError("no protein heavy atoms")
    aa = remove_hydrogens(aa)
    aa = remove_std_residue_terminal_atoms(aa)

    keep = np.ones(len(aa), dtype=bool)
    needs_ccd = False
    noncanon: set[str] = set()
    starts = struc.get_residue_starts(aa)
    stops = np.append(starts[1:], len(aa))

    for s, e in zip(starts, stops):
        rn = str(aa.res_name[s])
        ref_path = _PROTEIN_REFERENCE_MOLS / f"{rn}.sdf"
        if not ref_path.is_file():
            needs_ccd = True
            noncanon.add(rn)
            continue
        allowed = _canonical_atom_names(rn)
        keep[s:e] = np.isin(aa.atom_name[s:e], list(allowed))

    aa = aa[keep]
    if needs_ccd:
        if ccd is None:
            raise ValueError(
                "non-canonical residues require CCD: " + ", ".join(sorted(noncanon))
            )
        aa = remove_non_CCD_atoms(aa, ccd)
        aa = canonicalize_atom_order(aa, ccd)
    if len(aa) == 0:
        raise ValueError("no heavy atoms after cleanup")
    return aa


def annotate_entity_ids(atom_array):
    chains = sorted(set(str(c) for c in atom_array.chain_id))
    seq_to_eid: dict[str, int] = {}
    entity_ids = np.zeros(len(atom_array), dtype=np.int32)
    for ch in chains:
        mask = np.asarray(atom_array.chain_id).astype(str) == ch
        ch_aa = atom_array[mask]
        ch_starts = struc.get_residue_starts(ch_aa)
        seq = "".join(
            _AA3_TO_1.get(str(ch_aa.res_name[s]), "X")
            for s in ch_starts
        )
        if seq not in seq_to_eid:
            seq_to_eid[seq] = len(seq_to_eid) + 1
        entity_ids[np.flatnonzero(mask)] = seq_to_eid[seq]
    atom_array.set_annotation("entity_id", entity_ids)
    atom_array.set_annotation("mol_entity_id", entity_ids.copy())
    return atom_array


def build_chain_metadata(atom_array) -> dict[str, PreprocessingChainData]:
    return {
        str(c): PreprocessingChainData(
            label_asym_id=str(c),
            auth_asym_id=str(c),
            entity_id=1,
            molecule_type=MoleculeType.PROTEIN,
            reference_mol_id=None,
        )
        for c in np.unique(atom_array.chain_id)
    }


def _needs_ccd(atom_array) -> bool:
    protein = atom_array[atom_array.molecule_type_id == MoleculeType.PROTEIN]
    residue_names: set[str] = set(map(str, protein.res_name))
    residue_names.discard("MSE")
    return any(
        not (_PROTEIN_REFERENCE_MOLS / f"{name}.sdf").is_file()
        for name in residue_names
    )


def _init_worker(ccd_path: str | None) -> None:
    global _CCD_PATH
    _CCD_PATH = ccd_path


def _worker_ccd(atom_array):
    if not _CCD_PATH:
        return None
    if not _needs_ccd(atom_array):
        return None
    global _CCD_CACHE
    if _CCD_CACHE is None:
        _CCD_CACHE = pdbx.CIFFile.read(Path(_CCD_PATH))
    return _CCD_CACHE


def process_monomer_array(raw_chain, case_id: str, ccd):
    canonized = canonize_atom_array(raw_chain, ccd=ccd)
    annotated = annotate_entity_ids(canonized)
    processed = process_target_atom_array_of3(
        atom_array=annotated,
        pdb_id=case_id,
        crop_config={
            "chain_crop": {"enabled": False},
            "token_crop": {"enabled": False},
        },
        preferred_chain_or_interface=None,
        per_chain_metadata=build_chain_metadata(annotated),
    )
    add_token_positions(processed.atom_array_gt)
    return processed.atom_array_gt


def process_gt_array(raw_gt, chain_ids: list[str], ccd):
    restricted = raw_gt[
        np.isin(np.asarray(raw_gt.chain_id).astype(str), chain_ids)
    ]
    canonized = canonize_atom_array(restricted, ccd=ccd)
    annotated = annotate_entity_ids(canonized)
    return annotated


def parse_chain_residues(atom_array) -> dict[str, list[tuple[int, str]]]:
    """Return {chain_id: [(resseq, resname_3), ...]} in first-appearance order."""
    result: dict[str, list[tuple[int, str]]] = {}
    chains = sorted(set(str(c) for c in np.asarray(atom_array.chain_id).astype(str)))
    for ch in chains:
        mask = np.asarray(atom_array.chain_id).astype(str) == ch
        ch_aa = atom_array[mask]
        starts = struc.get_residue_starts(ch_aa)
        residues: list[tuple[int, str]] = []
        for s in starts:
            residues.append(
                (int(ch_aa.res_id[s]), str(ch_aa.res_name[s]).strip().upper())
            )
        result[ch] = residues
    return result


def derive_resseq_to_sequence_index(
    chain_residues: list[tuple[int, str]],
    seq_3aa: list[str],
) -> dict[int, int]:
    """Greedy 3-letter name matching, mirroring the v3 mesh fallback."""
    mapping: dict[int, int] = {}
    r_idx = 0
    for seq_idx, seq_res in enumerate(seq_3aa):
        while r_idx < len(chain_residues):
            rseq, rname = chain_residues[r_idx]
            r_idx += 1
            if rname == seq_res:
                mapping[rseq] = seq_idx
                break
    if not mapping and len(chain_residues) == len(seq_3aa):
        mapping = {rseq: i for i, (rseq, _) in enumerate(chain_residues)}
    return mapping


def extract_seq_sample(
    seq_cache: dict,
    case_id: str,
    input_aa,
) -> dict[str, Any] | None:
    case_seqs = seq_cache.get(case_id, {})
    if not case_seqs:
        return None
    chain_residues = parse_chain_residues(input_aa)
    result: dict[str, Any] = {}
    for chain_id, info in case_seqs.items():
        entry: dict[str, Any] = {
            "full_seq_1aa": info.get("full_seq_1aa", ""),
            "full_seq_3aa": info.get("full_seq_3aa", []),
        }
        r2s = derive_resseq_to_sequence_index(
            chain_residues.get(str(chain_id), []),
            entry["full_seq_3aa"],
        )
        if r2s:
            entry["resseq_to_sequence_index"] = {
                int(k): int(v) for k, v in r2s.items()
            }
        for opt_key in ("mask", "msa_eligible"):
            if opt_key in info:
                entry[opt_key] = info[opt_key]
        result[str(chain_id)] = entry
    return result


def _load_seq_cache(chunk_dir: Path) -> dict:
    path = chunk_dir / "seq_cache.pkl"
    if not path.is_file():
        return {}
    return pickle.loads(path.read_bytes())


def _process_one_sample(args: tuple) -> dict[str, Any]:
    chunk_dir, case_id, output_dir, force = args
    gt_pdb = chunk_dir / "gt" / f"{case_id}.pdb"
    input_pdb = chunk_dir / "input" / f"{case_id}.pdb"
    if not input_pdb.is_file():
        return {"case_id": case_id, "status": "skipped", "error": "no input pdb"}
    try:
        raw_gt = _parse_pdb(gt_pdb) if gt_pdb.is_file() else None
        raw_input = _parse_pdb(input_pdb)
        chain_ids = sorted(set(str(c) for c in np.unique(raw_input.chain_id)))
        if not chain_ids:
            return {"case_id": case_id, "status": "skipped", "error": "no chains"}
        ccd = _worker_ccd(raw_input)

        gt_pt = output_dir / f"{case_id}_gt.pt"
        if not gt_pt.exists() or force:
            if raw_gt is None:
                raise ValueError("missing gt pdb")
            gt_aa = process_gt_array(raw_gt, chain_ids, ccd)
            save_atomarray_as_pt(gt_aa, gt_pt)

        for chain_id in chain_ids:
            mono_pt = output_dir / f"{case_id}_{chain_id}_monomer.pt"
            if mono_pt.exists() and not force:
                continue
            mask = np.asarray(raw_input.chain_id).astype(str) == chain_id
            raw_chain = raw_input[mask]
            mono_aa = process_monomer_array(raw_chain, case_id, ccd)
            save_atomarray_as_pt(mono_aa, mono_pt)

        seq_pt = output_dir / f"{case_id}_seq.pt"
        if not seq_pt.exists() or force:
            seq_cache = _load_seq_cache(chunk_dir)
            seq_data = extract_seq_sample(seq_cache, case_id, raw_input)
            if seq_data:
                tmp = seq_pt.with_suffix(".pt.tmp")
                torch.save(seq_data, tmp)
                tmp.replace(seq_pt)
    except Exception as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"case_id": case_id, "status": "ok", "error": None}


def collect_tasks(
    chunk_dir: Path,
    output_chunk_dir: Path,
    max_samples: int | None,
    force: bool,
) -> list[tuple]:
    input_dir = chunk_dir / "input"
    if not input_dir.is_dir():
        return []
    case_ids = sorted({p.stem for p in input_dir.glob("*.pdb") if p.is_file()})
    if max_samples is not None:
        case_ids = case_ids[:max_samples]
    return [
        (chunk_dir, case_id, output_chunk_dir, force)
        for case_id in case_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pre-computed PDB tensors (GT + monomer + seq) for cli_v4."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4"),
    )
    parser.add_argument("--dataset-name", type=str, default="AFDB-10k-1")
    parser.add_argument("--chunks", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ccd-file", type=Path, default=None)
    args = parser.parse_args()

    chunk_parent = args.dataset_dir / "chunk"
    if chunk_parent.is_dir():
        all_chunk_dirs = sorted(d for d in chunk_parent.iterdir() if d.is_dir())
    else:
        all_chunk_dirs = sorted(d for d in args.dataset_dir.glob("chunk*") if d.is_dir())
    if not all_chunk_dirs:
        all_chunk_dirs = [args.dataset_dir]
    if args.chunks is not None:
        allowed = set(c.strip() for c in args.chunks.split(","))
        all_chunk_dirs = [d for d in all_chunk_dirs if d.name in allowed]

    output_root = args.output_root / args.dataset_name
    output_root.mkdir(parents=True, exist_ok=True)

    chunk_tasks: dict[str, list[tuple]] = {}
    for chunk_dir in all_chunk_dirs:
        if not (chunk_dir / "input").is_dir():
            continue
        chunk_output = output_root / chunk_dir.name
        tasks = collect_tasks(chunk_dir, chunk_output, args.max_samples, args.force)
        if tasks:
            chunk_tasks[chunk_dir.name] = tasks

    total_tasks = sum(len(t) for t in chunk_tasks.values())
    print(f"Dataset: {args.dataset_dir}")
    print(f"Chunks: {len(chunk_tasks)}")
    print(f"Samples: {total_tasks}")
    print(f"Output: {output_root}")
    print(f"CCD: {args.ccd_file}")

    flat_tasks = [
        task + (chunk_name,)
        for chunk_name, tasks in chunk_tasks.items()
        for task in tasks
    ]

    def _run(task: tuple) -> dict[str, Any]:
        chunk_name = task[-1]
        result = _process_one_sample(task[:-1])
        result["chunk"] = chunk_name
        return result

    global_stats: dict[str, Any] = {"chunks": {}}
    for chunk_name in chunk_tasks:
        global_stats["chunks"][chunk_name] = {
            "chunk": chunk_name,
            "total": len(chunk_tasks[chunk_name]),
            "ok": 0,
            "skipped": 0,
            "errors": 0,
            "error_list": [],
        }

    if args.num_workers <= 1:
        results = map(_run, flat_tasks)
    else:
        with Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(str(args.ccd_file) if args.ccd_file is not None else None,),
        ) as pool:
            results = pool.imap_unordered(_run, flat_tasks, chunksize=4)

    for result in results:
        stats = global_stats["chunks"][result["chunk"]]
        status = result["status"]
        if status in stats:
            stats[status] += 1
        else:
            stats["errors"] += 1
        if result.get("error"):
            stats["error_list"].append(
                {"case_id": result["case_id"], "error": result["error"]}
            )
            print(f"    SKIP {result['case_id']}: {result['error']}", file=sys.stderr)

    for chunk_name, stats in global_stats["chunks"].items():
        print(
            f"  {chunk_name}: {stats['ok']} ok, {stats['skipped']} skipped, "
            f"{stats['errors']} errors"
        )

    summary_path = output_root / "pdb_generation_summary.json"
    summary_path.write_text(json.dumps(global_stats, indent=2, default=str))
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法检查**

Run:
```bash
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m py_compile experiments_cli_v4/pdb_cache/generate_pdb_tensors_v4.py
```

Expected: 退出码 0，无输出。

- [ ] **Step 3: 写 `README.md`**

```markdown
# pdb_cache (v4) — 预计算 PDB tensors

把 v4 final chunk 的 `gt/` 和 `input/`（全原子 complex PDB）离线转成：

- `{case}_gt.pt`：canonize 后的 GT AtomArray（全重原子，pre-crop）
- `{case}_{chain}_monomer.pt`：从 `input/{case}.pdb` 按 chain 拆分、处理后
  的 monomer AtomArray（全重原子，no-crop + token positions）
- `{case}_seq.pt`：seq_cache 字段 + 推导的 `resseq_to_sequence_index`

用法：

```bash
python generate_pdb_tensors_v4.py \
  --dataset-dir /101045/jinyu/PPI/Prior_out/training/cli_v4/AFDB-10k-1 \
  --output-root /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4 \
  --dataset-name AFDB-10k-1 \
  --num-workers 8
```

Q-BioLiP 需要 CCD：

```bash
python generate_pdb_tensors_v4.py \
  --dataset-dir /101045/jinyu/PPI/Prior_out/training/cli_v4/Q-BioLiP-10k-new \
  --dataset-name Q-BioLiP-10k-new \
  --ccd-file /101045/jinyu/PhysPPI/assets/openfold3-physppi-20260724/ccd/components.cif
```

注意：
- 不依赖 `.chunk/` 中间产物，monomer 直接从最终 `input/` 按链拆分。
- H 原子在 canonize 时移除，保留全部重原子。
- `save_atomarray_as_pt` / `load_atomarray_from_pt` 复用 v3 的
  `experiments/pdb_cache/pdb_adapter.py`，格式完全兼容。
```

- [ ] **Step 4: 写 `HANDOFF20260808.md`**

```markdown
# PDB Cache 交接文档（v4）— 2026-08-08

## 状态

- `generate_pdb_tensors_v4.py` 已写好，`py_compile` 通过。
- **未运行生成**。

## v4 与 v3 的差异

1. 输入没有 `bb_input/{case}_{chain}.pdb`，改为 `input/{case}.pdb` 整复合物，
   脚本按 chain 切片生成 monomer。
2. monomer 保存全重原子（v3 只有 backbone）。
3. v4 `seq_cache.pkl` 没有 `resseq_to_sequence_index`，脚本用
   `derive_resseq_to_sequence_index`（greedy 三字母匹配）推导并写入 `_seq.pt`。
4. CLI/输出布局对齐 mesh_prior 师姐脚本；summary 为
   `pdb_generation_summary.json`。

## 踩坑提醒

- 必须用 PhysPPI conda 环境（biotite + openfold3）。
- CCD 只在 worker 内惰性加载一次；Q-BioLiP 必须传 `--ccd-file`。
- 不要 import `_runtime.py`（CUDA 依赖链）。
- 若上游 `_runtime.py` 的 canonize / monomer 逻辑有改动，本脚本需要同步。
```

- [ ] **Step 5: Commit**

```bash
git add experiments_cli_v4/pdb_cache/generate_pdb_tensors_v4.py experiments_cli_v4/pdb_cache/README.md experiments_cli_v4/pdb_cache/HANDOFF20260808.md
git commit -m "feat(cli_v4): add pdb_cache precompute generator + docs"
```

---

### Task 4: residue_prior 生成脚本 + 文档

**Files:**
- Create: `experiments_cli_v4/residue_prior/generate_residue_tensors_v4.py`
- Create: `experiments_cli_v4/residue_prior/README.md`
- Create: `experiments_cli_v4/residue_prior/HANDOFF20260808.md`

- [ ] **Step 1: 写 `generate_residue_tensors_v4.py`**

```python
#!/usr/bin/env python3
"""Generate pre-computed residue prior tensors from v4 prior PKL files.

Extracts .dmap, .interactions, and .tmdet from each sample's prior_pkl and
saves three independent .pt files:

    {case_id}_interface_residues.pt       — CA-CA contact pairs (dmap)
    {case_id}_interaction_pairs.pt        — 5 typed interactions
    {case_id}_membrane_relative_location.pt — membrane intervals + plane

The v4 prior PKL stores membrane information in the ``tmdet`` field
(``res_loc`` / ``outer_centroid`` / ``outer_normal`` / ``thicken``); the
legacy v3 ``opm`` field is used as a fallback.

Output layout:
    {output_root}/{dataset_name}/{chunk_name}/{case_id}_{suffix}.pt
    {output_root}/{dataset_name}/residue_generation_summary.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import torch

_INTERACTION_TYPE_MAP: dict[str, int] = {
    "hbond": 0,
    "saltbridge": 1,
    "hydrophobic": 2,
    "pistacking": 3,
    "pication": 4,
}


def _normalize_atom_group(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        name = value.strip().upper()
        return (name,) if name else None
    if isinstance(value, (list, tuple)):
        names = tuple(
            atom.strip().upper()
            for atom in value
            if isinstance(atom, str) and atom.strip()
        )
        if len(names) == len(value) and names:
            return names
    return None


def _load_pdb_atom_sets(pdb_path: Path) -> dict[tuple[str, int], set[str]]:
    atom_sets: dict[tuple[str, int], set[str]] = {}
    try:
        lines = pdb_path.read_text(errors="replace").splitlines()
    except OSError:
        return atom_sets
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) > 100:
            continue
        atom_name = line[12:16].strip().upper()
        chain_id = line[21].strip()
        try:
            resseq = int(line[22:26].strip())
        except ValueError:
            continue
        if not atom_name:
            continue
        atom_sets.setdefault((chain_id, resseq), set()).add(atom_name)
    return atom_sets


def _load_membrane_attr(obj: Any, name: str) -> dict:
    value = getattr(obj, name, None)
    if value is not None and isinstance(value, dict):
        return value
    return {}


def generate_sample(
    pkl_path: Path,
) -> tuple[dict | None, dict | None, dict | None]:
    try:
        raw = pickle.loads(pkl_path.read_bytes())
    except Exception:
        return None, None, None

    case_id = pkl_path.stem
    if not isinstance(raw, dict) or case_id not in raw:
        return None, None, None

    obj = raw[case_id]
    gt_pdb = pkl_path.parent.parent / "gt" / f"{case_id}.pdb"
    atom_sets = _load_pdb_atom_sets(gt_pdb) if gt_pdb.is_file() else {}

    # --- dmap (interface residues) ----------------------------------------
    raw_dmap: dict = {}
    if hasattr(obj, "dmap") and isinstance(obj.dmap, dict):
        raw_dmap = obj.dmap

    dmap_out: dict[str, torch.Tensor] = {}
    for chain_pair, edges in raw_dmap.items():
        if not isinstance(chain_pair, str) or "|" not in chain_pair:
            continue
        if not isinstance(edges, (list, tuple)) or not edges:
            continue
        rows: list[list[float]] = []
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) < 3:
                continue
            try:
                r_a = int(edge[0])
                r_b = int(edge[1])
                dist = float(edge[2])
            except (ValueError, TypeError):
                continue
            rows.append([r_a, r_b, dist])
        if rows:
            dmap_out[chain_pair] = torch.tensor(rows, dtype=torch.float32)

    # --- interactions ------------------------------------------------------
    raw_interactions: dict = {}
    if (
        hasattr(obj, "interactions")
        and obj.interactions is not None
        and isinstance(obj.interactions, dict)
    ):
        raw_interactions = obj.interactions

    interactions_out: dict[str, dict[str, Any]] = {}
    for chain_pair, records in raw_interactions.items():
        if not isinstance(chain_pair, str) or "|" not in chain_pair:
            continue
        if not isinstance(records, (list, tuple)) or not records:
            continue
        left_ch, right_ch = chain_pair.split("|", 1)

        pairs: list[list[int]] = []
        atom_1_list: list[tuple[str, ...]] = []
        atom_2_list: list[tuple[str, ...]] = []
        dist_list: list[float] = []

        for item in records:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("t", ""))
            type_id = _INTERACTION_TYPE_MAP.get(kind)
            if type_id is None:
                continue
            try:
                r_a = int(item["r1"])
                r_b = int(item["r2"])
            except (KeyError, ValueError, TypeError):
                continue

            a1 = _normalize_atom_group(item.get("a1"))
            a2 = _normalize_atom_group(item.get("a2"))
            if a1 is None or a2 is None:
                continue
            if atom_sets:
                left_atoms = atom_sets.get((left_ch, r_a), ())
                right_atoms = atom_sets.get((right_ch, r_b), ())
                direct_ok = all(a in left_atoms for a in a1) and all(
                    a in right_atoms for a in a2
                )
                if not direct_ok:
                    swapped_ok = all(a in right_atoms for a in a1) and all(
                        a in left_atoms for a in a2
                    )
                    if swapped_ok:
                        a1, a2 = a2, a1
            try:
                dist = float(item["d"]) if item.get("d") is not None else float("nan")
            except (KeyError, ValueError, TypeError, OverflowError):
                dist = float("nan")

            pairs.append([r_a, r_b, type_id])
            atom_1_list.append(a1)
            atom_2_list.append(a2)
            dist_list.append(dist)

        if pairs:
            interactions_out[chain_pair] = {
                "pairs": torch.tensor(pairs, dtype=torch.int32),
                "atom_1": atom_1_list,
                "atom_2": atom_2_list,
                "dist": torch.tensor(dist_list, dtype=torch.float32),
            }

    # --- membrane (tmdet preferred, opm fallback) --------------------------
    membrane_out: dict[str, Any] | None = None
    raw_membrane = _load_membrane_attr(obj, "tmdet")
    if not raw_membrane:
        raw_membrane = _load_membrane_attr(obj, "opm")

    if raw_membrane:
        res_loc_raw = raw_membrane.get("res_loc")
        plane = {
            "normal": torch.tensor(
                raw_membrane.get("outer_normal", [0, 0, 0]),
                dtype=torch.float32,
            ),
            "centroid": torch.tensor(
                raw_membrane.get("outer_centroid", [0, 0, 0]),
                dtype=torch.float32,
            ),
            "thicken": float(raw_membrane.get("thicken", 0.0)),
        }
        res_loc_out: dict[str, torch.Tensor] = {}
        if isinstance(res_loc_raw, (list, tuple)):
            by_chain: dict[str, list[list[int]]] = {}
            for row in res_loc_raw:
                if not isinstance(row, (list, tuple)) or len(row) < 4:
                    continue
                try:
                    chain_id = str(row[0])
                    start = int(row[1])
                    end = int(row[2])
                    side = int(row[3])
                except (ValueError, TypeError):
                    continue
                by_chain.setdefault(chain_id, []).append([start, end, side])
            for chain_id, intervals in by_chain.items():
                res_loc_out[chain_id] = torch.tensor(intervals, dtype=torch.int32)

        if res_loc_out:
            membrane_out = {"res_loc": res_loc_out, "plane": plane}

    return (
        {"dmap": dmap_out} if dmap_out else None,
        {"interactions": interactions_out} if interactions_out else None,
        membrane_out,
    )


def _save_pt(output_dir: Path, filename: str, payload: dict) -> None:
    tmp_path = output_dir / (filename + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_dir / filename)


def _process_one_sample(args: tuple) -> dict[str, Any]:
    pkl_path, output_dir, force = args
    case_id = pkl_path.stem
    interface_pt = output_dir / f"{case_id}_interface_residues.pt"
    interaction_pt = output_dir / f"{case_id}_interaction_pairs.pt"
    membrane_pt = output_dir / f"{case_id}_membrane_relative_location.pt"

    if all(p.exists() for p in (interface_pt, interaction_pt, membrane_pt)) and not force:
        return {
            "case_id": case_id,
            "status": "exists",
            "error": None,
            "interface": 0,
            "interaction": 0,
            "membrane": 0,
        }

    try:
        dmap_payload, inter_payload, memb_payload = generate_sample(pkl_path)
    except Exception as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "interface": 0,
            "interaction": 0,
            "membrane": 0,
        }

    n_iface = 0
    n_inter = 0
    n_memb = 0
    try:
        if dmap_payload is not None:
            n_iface = sum(t.shape[0] for t in dmap_payload["dmap"].values())
            if not interface_pt.exists() or force:
                _save_pt(output_dir, f"{case_id}_interface_residues.pt", dmap_payload)
        if inter_payload is not None:
            n_inter = sum(
                v["pairs"].shape[0]
                for v in inter_payload["interactions"].values()
            )
            if not interaction_pt.exists() or force:
                _save_pt(output_dir, f"{case_id}_interaction_pairs.pt", inter_payload)
        if memb_payload is not None:
            n_memb = sum(t.shape[0] for t in memb_payload["res_loc"].values())
            if not membrane_pt.exists() or force:
                _save_pt(
                    output_dir,
                    f"{case_id}_membrane_relative_location.pt",
                    memb_payload,
                )
    except Exception as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "error": f"save failed: {exc}",
            "interface": 0,
            "interaction": 0,
            "membrane": 0,
        }

    return {
        "case_id": case_id,
        "status": "ok",
        "error": None,
        "interface": n_iface,
        "interaction": n_inter,
        "membrane": n_memb,
    }


def collect_tasks(
    chunk_dir: Path,
    output_chunk_dir: Path,
    max_samples: int | None,
    force: bool,
) -> list[tuple]:
    prior_pkl_dir = chunk_dir / "prior_pkl"
    if not prior_pkl_dir.is_dir():
        return []
    pkl_paths = sorted(prior_pkl_dir.glob("*.pkl"))
    if max_samples is not None:
        pkl_paths = pkl_paths[:max_samples]
    return [(p, output_chunk_dir, force) for p in pkl_paths]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-sample residue prior tensors from v4 PKL."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4"),
    )
    parser.add_argument("--dataset-name", type=str, default="AFDB-10k-1")
    parser.add_argument("--chunks", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    chunk_parent = args.dataset_dir / "chunk"
    if chunk_parent.is_dir():
        all_chunk_dirs = sorted(d for d in chunk_parent.iterdir() if d.is_dir())
    else:
        all_chunk_dirs = sorted(d for d in args.dataset_dir.glob("chunk*") if d.is_dir())
    if not all_chunk_dirs:
        all_chunk_dirs = [args.dataset_dir]
    if args.chunks is not None:
        allowed = set(c.strip() for c in args.chunks.split(","))
        all_chunk_dirs = [d for d in all_chunk_dirs if d.name in allowed]

    output_root = args.output_root / args.dataset_name
    output_root.mkdir(parents=True, exist_ok=True)

    chunk_tasks: dict[str, list[tuple]] = {}
    for chunk_dir in all_chunk_dirs:
        if not (chunk_dir / "prior_pkl").is_dir():
            continue
        chunk_output = output_root / chunk_dir.name
        tasks = collect_tasks(chunk_dir, chunk_output, args.max_samples, args.force)
        if tasks:
            chunk_tasks[chunk_dir.name] = tasks

    total_tasks = sum(len(t) for t in chunk_tasks.values())
    print(f"Dataset: {args.dataset_dir}")
    print(f"Chunks: {len(chunk_tasks)}")
    print(f"Samples: {total_tasks}")
    print(f"Output: {output_root}")

    flat_tasks = [
        task + (chunk_name,)
        for chunk_name, tasks in chunk_tasks.items()
        for task in tasks
    ]

    def _run(task: tuple) -> dict[str, Any]:
        chunk_name = task[-1]
        result = _process_one_sample(task[:-1])
        result["chunk"] = chunk_name
        return result

    global_stats: dict[str, Any] = {"chunks": {}}
    for chunk_name in chunk_tasks:
        global_stats["chunks"][chunk_name] = {
            "chunk": chunk_name,
            "total": len(chunk_tasks[chunk_name]),
            "ok": 0,
            "exists": 0,
            "errors": 0,
            "error_list": [],
            "total_dmap_pairs": 0,
            "total_interaction_pairs": 0,
            "total_tm_intervals": 0,
        }

    if args.num_workers <= 1:
        results = map(_run, flat_tasks)
    else:
        with Pool(processes=args.num_workers) as pool:
            results = pool.imap_unordered(_run, flat_tasks, chunksize=4)

    for result in results:
        stats = global_stats["chunks"][result["chunk"]]
        status = result["status"]
        if status in stats:
            stats[status] += 1
        else:
            stats["errors"] += 1
        stats["total_dmap_pairs"] += result["interface"]
        stats["total_interaction_pairs"] += result["interaction"]
        stats["total_tm_intervals"] += result["membrane"]
        if result.get("error"):
            stats["error_list"].append(
                {"case_id": result["case_id"], "error": result["error"]}
            )
            print(f"    SKIP {result['case_id']}: {result['error']}", file=sys.stderr)

    for chunk_name, stats in global_stats["chunks"].items():
        print(
            f"  {chunk_name}: {stats['ok']} ok, {stats['exists']} existing, "
            f"{stats['errors']} errors"
        )

    summary_path = output_root / "residue_generation_summary.json"
    summary_path.write_text(json.dumps(global_stats, indent=2, default=str))
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法检查**

Run:
```bash
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m py_compile experiments_cli_v4/residue_prior/generate_residue_tensors_v4.py
```

Expected: 退出码 0，无输出。

- [ ] **Step 3: 写 `README.md`**

```markdown
# residue_prior (v4) — 预计算 residue prior tensors

从 v4 `prior_pkl/{case}.pkl` 抽取：

- `{case}_interface_residues.pt`：dmap（CA-CA ≤ 18Å）
- `{case}_interaction_pairs.pt`：5 种 PLIP 相互作用（hbond / saltbridge /
  hydrophobic / pistacking / pication）
- `{case}_membrane_relative_location.pt`：膜区间 + 平面（优先 `tmdet`，
  `opm` 兜底）

用法：

```bash
python generate_residue_tensors_v4.py \
  --dataset-dir /101045/jinyu/PPI/Prior_out/training/cli_v4/AFDB-10k-1 \
  --output-root /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4 \
  --dataset-name AFDB-10k-1 \
  --num-workers 8
```

说明：
- dmap / interactions / membrane 三个 `.pt` 独立存在，缺哪个就不写哪个。
- 所有坐标用 PDB resseq，训练 adapter 直接 `token_index.get((chain, resseq))`。
- `torch.load(..., weights_only=False)`（.pt 里有 tuple 字符串）。
```

- [ ] **Step 4: 写 `HANDOFF20260808.md`**

```markdown
# Residue Prior 交接文档（v4）— 2026-08-08

## 状态

- `generate_residue_tensors_v4.py` 已写好，`py_compile` 通过。
- **未运行生成**。

## v4 与 v3 的差异

1. 膜信息字段从 `opm` 改为 `tmdet`（`res_loc` / `outer_centroid` /
   `outer_normal` / `thicken`），脚本保留 `opm` 兜底。
2. case id 带 `_G1` 等分组后缀；一个 case 的 dmap / interactions 可能有
   多对链（如 3-chain 的 `A|B`、`A|C`、`B|C`）。
3. CLI/输出布局对齐 mesh_prior 师姐脚本；summary 为
   `residue_generation_summary.json`（避免覆盖 mesh 的
   `generation_summary.json`）。

## 踩坑提醒

- interactions 只保留 5 种类型；atom names 不可丢（loss 要算质心距离）。
- 三个 `.pt` 独立保存，避免加载“半空”文件。
- 不要用单 chunk 推断全数据集（AFDB 不是所有 case 都有 interactions）。
```

- [ ] **Step 5: Commit**

```bash
git add experiments_cli_v4/residue_prior/generate_residue_tensors_v4.py experiments_cli_v4/residue_prior/README.md experiments_cli_v4/residue_prior/HANDOFF20260808.md
git commit -m "feat(cli_v4): add residue_prior precompute generator + docs"
```

---

### Task 5: split_training_val 脚本 + 文档

**Files:**
- Create: `experiments_cli_v4/split_training_val/extract_sequences_v4.py`
- Create: `experiments_cli_v4/split_training_val/run_mmseqs_v4.py`
- Create: `experiments_cli_v4/split_training_val/build_split_v4.py`
- Create: `experiments_cli_v4/split_training_val/run_all_v4.py`
- Create: `experiments_cli_v4/split_training_val/README.md`
- Create: `experiments_cli_v4/split_training_val/HANDOFF20260808.md`

- [ ] **Step 1: 写 `extract_sequences_v4.py`**

```python
#!/usr/bin/env python3
"""Extract per-chain sequences from cli_v4 training data for MMseqs2 clustering.

Produces:
  - sequences.fasta          : one entry per chain, header = ``>{case_id}|{chain_id}``
  - chain_index.json         : mapping from seq_id to (dataset, case_id, chain_id, seq_len)
  - case_chains.json         : mapping from case_id to list of chain_ids

All outputs land in --output-dir (default: experiments_cli_v4/split_training_val/).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

DATASET_DIRS = {
    "afdb": Path("/101045/jinyu/PPI/Prior_out/training/cli_v4/AFDB-10k-1"),
    "qbiolip": Path("/101045/jinyu/PPI/Prior_out/training/cli_v4/Q-BioLiP-10k-new"),
}


def _write_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def extract_sequences(
    dataset_dirs: dict[str, Path],
    output_dir: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / "sequences.fasta"
    chain_index: dict[str, dict[str, str]] = {}
    case_chains: dict[str, list[str]] = {}

    total_cases = 0
    total_chains = 0

    with fasta_path.open("w", encoding="ascii", newline="\n") as fh:
        for dataset_name, dataset_dir in sorted(dataset_dirs.items()):
            chunk_dir = dataset_dir / "chunk"
            if not chunk_dir.is_dir():
                print(f"WARNING: {chunk_dir} not found, skipping {dataset_name}")
                continue

            for chunk_path in sorted(chunk_dir.glob("chunk*")):
                seq_cache_path = chunk_path / "seq_cache.pkl"
                if not seq_cache_path.is_file():
                    print(f"WARNING: {seq_cache_path} not found")
                    continue

                with seq_cache_path.open("rb") as handle:
                    seq_cache: dict[str, dict[str, Any]] = pickle.load(handle)

                for case_id in sorted(seq_cache):
                    chains = seq_cache[case_id]
                    total_cases += 1
                    case_chains[case_id] = []

                    for chain_id in sorted(chains):
                        chain_data = chains[chain_id]
                        seq_1aa = chain_data.get("full_seq_1aa", "")
                        if not seq_1aa:
                            seq_3aa = chain_data.get("full_seq_3aa", [])
                            seq_1aa = "".join(
                                {
                                    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
                                    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
                                    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                                    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
                                    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
                                    "MSE": "M", "ASX": "B", "GLX": "Z", "UNK": "X",
                                }.get(aa, "X")
                                for aa in seq_3aa
                            )

                        seq_id = f"{case_id}|{chain_id}"
                        fh.write(f">{seq_id}\n{seq_1aa}\n")
                        chain_index[seq_id] = {
                            "dataset": dataset_name,
                            "case_id": case_id,
                            "chain_id": chain_id,
                            "seq_len": len(seq_1aa),
                        }
                        case_chains[case_id].append(chain_id)
                        total_chains += 1

    print(
        f"Extracted {total_chains} chains from {total_cases} cases "
        f"({len(chain_index)} unique chain IDs)"
    )
    if len(chain_index) != total_chains:
        raise ValueError(
            f"Duplicate chain IDs: {total_chains} entries but {len(chain_index)} unique"
        )

    _write_json(output_dir / "chain_index.json", chain_index)
    _write_json(output_dir / "case_chains.json", case_chains)
    return chain_index, case_chains


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory for FASTA and JSON index files",
    )
    parser.add_argument(
        "--dataset-dirs",
        type=Path,
        nargs=2,
        default=None,
        help="Override dataset directories: AFDB_DIR QBIOLIP_DIR",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_dirs = dict(DATASET_DIRS)
    if args.dataset_dirs is not None:
        dataset_dirs = {
            "afdb": args.dataset_dirs[0],
            "qbiolip": args.dataset_dirs[1],
        }
    extract_sequences(dataset_dirs, args.output_dir)
    print(f"Done. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写 `run_mmseqs_v4.py`**

```python
#!/usr/bin/env python3
"""Run MMseqs2 easy-cluster at 30% and 40% identity for cli_v4 split."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run_mmseqs(
    fasta_path: Path,
    output_prefix: Path,
    tmp_dir: Path,
    *,
    executable: str = "mmseqs",
    threads: int = 16,
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    cov_mode: int = 0,
    cluster_mode: int = 2,
) -> Path:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for stale in output_prefix.parent.glob(output_prefix.name + "*"):
        if stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink()

    cmd = [
        executable,
        "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        str(tmp_dir),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", str(cov_mode),
        "--cluster-mode", str(cluster_mode),
        "--threads", str(threads),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    cluster_tsv = output_prefix.with_name(output_prefix.name + "_cluster.tsv")
    if not cluster_tsv.is_file():
        raise RuntimeError(f"MMseqs2 did not produce {cluster_tsv}")
    return cluster_tsv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--mmseqs",
        type=str,
        default=os.environ.get("MMSEQS", "mmseqs"),
    )
    parser.add_argument("--threads", type=int, default=min(32, (os.cpu_count() or 1)))
    parser.add_argument("--coverage", type=float, default=0.8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for min_seq_id, label in [(0.3, "30"), (0.4, "40")]:
        out_prefix = args.output_dir / f"mmseqs_{label}" / "clusters"
        tmp_dir = args.output_dir / f"mmseqs_{label}" / "tmp"
        print(f"\n=== Clustering at {min_seq_id * 100:.0f}% identity ===")
        cluster_tsv = run_mmseqs(
            fasta_path=args.fasta,
            output_prefix=out_prefix,
            tmp_dir=tmp_dir,
            executable=args.mmseqs,
            threads=args.threads,
            min_seq_id=min_seq_id,
            coverage=args.coverage,
        )
        print(f"  -> {cluster_tsv}")
    print(f"\nDone. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 写 `build_split_v4.py`**

```python
#!/usr/bin/env python3
"""Build chain-level homology-aware train/val split for cli_v4 (AF-Multimer protocol).

Algorithm is identical to v3:
1. Parse MMseqs2 30% cluster TSV -> per-chain membership.
2. Build case-level homology graph (shared 30% cluster => edge).
3. Find connected components; greedy largest-first into val until ~20%.
4. Parse 40% clusters -> per-case sampling weights (1/max cluster size, mean=1).
5. Write split_manifest.json, sampling_weights.pt, train/val IDs, split_report.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _write_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def parse_cluster_tsv(
    cluster_tsv: Path,
    chain_index: dict[str, dict[str, str]],
) -> dict[str, str]:
    valid_ids = set(chain_index)
    memberships: dict[str, str] = {}
    with cluster_tsv.open("r", encoding="ascii") as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            if rep not in valid_ids:
                raise ValueError(f"unknown representative in cluster TSV: {rep}")
            if member not in valid_ids:
                raise ValueError(f"unknown member in cluster TSV: {member}")
            if member in memberships:
                raise ValueError(f"chain assigned to multiple clusters: {member}")
            memberships[member] = rep

    clustered = set(memberships)
    for seq_id in sorted(valid_ids - clustered):
        memberships[seq_id] = seq_id
    for rep in set(memberships.values()):
        if rep not in clustered:
            memberships[rep] = rep
    return memberships


def build_homology_components(
    chain_memberships_30: dict[str, str],
    case_chains: dict[str, list[str]],
) -> list[set[str]]:
    cluster_to_cases: dict[str, set[str]] = defaultdict(set)
    for case_id, chain_ids in case_chains.items():
        for chain_id in chain_ids:
            seq_id = f"{case_id}|{chain_id}"
            cluster_rep = chain_memberships_30.get(seq_id)
            if cluster_rep is not None:
                cluster_to_cases[cluster_rep].add(case_id)

    parent: dict[str, str] = {}
    all_cases = set(case_chains)
    for case_id in all_cases:
        parent[case_id] = case_id

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for cluster_rep, cases in cluster_to_cases.items():
        cases_list = sorted(cases)
        if len(cases_list) < 2:
            continue
        first = cases_list[0]
        for other in cases_list[1:]:
            union(first, other)

    components: dict[str, set[str]] = defaultdict(set)
    for case_id in all_cases:
        components[find(case_id)].add(case_id)
    return sorted(components.values(), key=len, reverse=True)


def split_components(
    components: list[set[str]],
    target_val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[set[str], set[str], dict[str, Any]]:
    import random

    total = sum(len(c) for c in components)
    target_val = int(total * target_val_ratio)

    by_size: dict[int, list[set[str]]] = defaultdict(list)
    for comp in components:
        by_size[len(comp)].append(comp)

    rng = random.Random(seed)
    ordered: list[set[str]] = []
    for size in sorted(by_size, reverse=True):
        group = by_size[size]
        rng.shuffle(group)
        ordered.extend(group)

    val_cases: set[str] = set()
    val_count = 0
    for comp in ordered:
        if val_count + len(comp) <= target_val:
            val_cases.update(comp)
            val_count += len(comp)

    train_cases = set()
    for comp in components:
        if not (comp & val_cases):
            train_cases.update(comp)

    actual_val_ratio = len(val_cases) / total if total else 0.0
    stats = {
        "total_cases": total,
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "train_ratio": len(train_cases) / total if total else 0.0,
        "val_ratio": actual_val_ratio,
        "connected_components": len(components),
        "components_in_val": sum(1 for c in components if c & val_cases),
        "components_in_train": sum(1 for c in components if c & train_cases),
        "largest_component": max((len(c) for c in components), default=0),
        "singleton_components": sum(1 for c in components if len(c) == 1),
    }
    return train_cases, val_cases, stats


def calculate_sampling_weights(
    chain_memberships_40: dict[str, str],
    case_chains: dict[str, list[str]],
    train_cases: set[str],
) -> dict[str, float]:
    cluster_sizes: dict[str, int] = defaultdict(int)
    for seq_id, rep in chain_memberships_40.items():
        cluster_sizes[rep] += 1

    raw_weights: dict[str, float] = {}
    for case_id, chain_ids in case_chains.items():
        max_size = 1
        for chain_id in chain_ids:
            seq_id = f"{case_id}|{chain_id}"
            rep = chain_memberships_40.get(seq_id, seq_id)
            sz = cluster_sizes.get(rep, 1)
            if sz > max_size:
                max_size = sz
        raw_weights[case_id] = 1.0 / max_size

    train_raw = {cid: raw_weights[cid] for cid in train_cases if cid in raw_weights}
    if not train_raw:
        return {}
    mean_raw = sum(train_raw.values()) / len(train_raw)
    return {cid: raw / mean_raw for cid, raw in train_raw.items()}


def build_split(
    output_dir: Path,
    final_dir: Path,
    *,
    seed: int = 42,
    target_val_ratio: float = 0.2,
    mmseqs_params: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    chain_index = _load_json(output_dir / "chain_index.json")
    case_chains = _load_json(output_dir / "case_chains.json")
    case_chains = {k: list(v) for k, v in case_chains.items()}
    print(f"Loaded {len(case_chains)} cases, {len(chain_index)} chains")

    cluster_tsv_30 = output_dir / "mmseqs_30" / "clusters_cluster.tsv"
    if not cluster_tsv_30.is_file():
        raise FileNotFoundError(f"30% cluster TSV not found: {cluster_tsv_30}")
    chain_memberships_30 = parse_cluster_tsv(cluster_tsv_30, chain_index)
    print(
        f"30% clusters: {len(set(chain_memberships_30.values()))} clusters "
        f"for {len(chain_memberships_30)} chains"
    )

    cluster_tsv_40 = output_dir / "mmseqs_40" / "clusters_cluster.tsv"
    if not cluster_tsv_40.is_file():
        raise FileNotFoundError(f"40% cluster TSV not found: {cluster_tsv_40}")
    chain_memberships_40 = parse_cluster_tsv(cluster_tsv_40, chain_index)
    print(
        f"40% clusters: {len(set(chain_memberships_40.values()))} clusters "
        f"for {len(chain_memberships_40)} chains"
    )

    components = build_homology_components(chain_memberships_30, case_chains)
    print(
        f"Connected components: {len(components)} "
        f"(largest={len(components[0]) if components else 0}, "
        f"singletons={sum(1 for c in components if len(c) == 1)})"
    )

    train_cases, val_cases, split_stats = split_components(
        components, target_val_ratio=target_val_ratio, seed=seed
    )
    print(
        f"Split: train={len(train_cases)} ({split_stats['train_ratio']:.1%}), "
        f"val={len(val_cases)} ({split_stats['val_ratio']:.1%})"
    )

    sampling_weights = calculate_sampling_weights(
        chain_memberships_40, case_chains, train_cases
    )
    print(
        f"Sampling weights for {len(sampling_weights)} training cases "
        f"(min={min(sampling_weights.values()):.4f}, "
        f"max={max(sampling_weights.values()):.4f})"
    )

    dataset_cases: dict[str, list[str]] = defaultdict(list)
    for case_id in case_chains:
        for seq_id, info in chain_index.items():
            if info["case_id"] == case_id:
                dataset_cases[info["dataset"]].append(case_id)
                break

    dataset_splits: dict[str, dict[str, list[str]]] = {}
    for dataset_name in sorted(dataset_cases):
        all_ds = set(dataset_cases[dataset_name])
        dataset_splits[dataset_name] = {
            "train_ids": sorted(all_ds & train_cases),
            "val_ids": sorted(all_ds & val_cases),
        }
        print(
            f"  {dataset_name}: train={len(dataset_splits[dataset_name]['train_ids'])}, "
            f"val={len(dataset_splits[dataset_name]['val_ids'])}"
        )

    chain_clusters_30: dict[str, dict[str, object]] = {}
    cluster_sizes_30: dict[str, int] = defaultdict(int)
    for seq_id, rep in chain_memberships_30.items():
        cluster_sizes_30[rep] += 1
    for seq_id, rep in chain_memberships_30.items():
        chain_clusters_30[seq_id] = {"rep": rep, "size": cluster_sizes_30[rep]}

    chain_clusters_40: dict[str, dict[str, object]] = {}
    cluster_sizes_40: dict[str, int] = defaultdict(int)
    for seq_id, rep in chain_memberships_40.items():
        cluster_sizes_40[rep] += 1
    for seq_id, rep in chain_memberships_40.items():
        chain_clusters_40[seq_id] = {"rep": rep, "size": cluster_sizes_40[rep]}

    case_clusters_30: dict[str, list[str]] = {}
    for case_id, chain_ids in case_chains.items():
        reps: list[str] = []
        for chain_id in chain_ids:
            seq_id = f"{case_id}|{chain_id}"
            rep = chain_memberships_30.get(seq_id, seq_id)
            if rep not in reps:
                reps.append(rep)
        case_clusters_30[case_id] = reps

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": dataset_splits,
    }
    _write_json(output_dir / "split_manifest.json", manifest)
    _write_json(final_dir / "split_manifest.json", manifest)

    _write_json(output_dir / "train_ids.json", sorted(train_cases))
    _write_json(output_dir / "val_ids.json", sorted(val_cases))

    mmseqs_params_filled = mmseqs_params or {
        "30": {"min_seq_id": 0.3, "coverage": 0.8, "cov_mode": 0, "cluster_mode": 2},
        "40": {"min_seq_id": 0.4, "coverage": 0.8, "cov_mode": 0, "cluster_mode": 2},
    }
    pt_payload = {
        "schema": "physppi_identity_split_v1",
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": dataset_splits,
        "sampling_weights": sampling_weights,
        "chain_clusters_30": chain_clusters_30,
        "chain_clusters_40": chain_clusters_40,
        "case_clusters_30": case_clusters_30,
        "mmseqs_params": mmseqs_params_filled,
        "split_stats": split_stats,
    }
    pt_path = output_dir / "sampling_weights.pt"
    torch.save(pt_payload, pt_path)
    final_pt_path = final_dir / "sampling_weights.pt"
    torch.save(pt_payload, final_pt_path)
    print(f"Saved sampling_weights.pt -> {final_pt_path}")

    report = {
        **split_stats,
        "per_dataset": {
            ds: {
                "train": len(info["train_ids"]),
                "val": len(info["val_ids"]),
            }
            for ds, info in dataset_splits.items()
        },
        "cluster_stats_30": {
            "n_clusters": len(set(chain_memberships_30.values())),
            "max_size": max(cluster_sizes_30.values()) if cluster_sizes_30 else 0,
            "mean_size": (
                sum(cluster_sizes_30.values())
                / len(set(chain_memberships_30.values()))
                if cluster_sizes_30
                else 0
            ),
        },
        "cluster_stats_40": {
            "n_clusters": len(set(chain_memberships_40.values())),
            "max_size": max(cluster_sizes_40.values()) if cluster_sizes_40 else 0,
            "mean_size": (
                sum(cluster_sizes_40.values())
                / len(set(chain_memberships_40.values()))
                if cluster_sizes_40
                else 0
            ),
        },
        "weight_stats": {
            "min": min(sampling_weights.values()) if sampling_weights else 0,
            "max": max(sampling_weights.values()) if sampling_weights else 0,
            "mean": (
                sum(sampling_weights.values()) / len(sampling_weights)
                if sampling_weights
                else 0
            ),
        },
    }
    _write_json(output_dir / "split_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=Path("/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_split(
        output_dir=args.output_dir,
        final_dir=args.final_dir,
        seed=args.seed,
        target_val_ratio=args.val_ratio,
    )
    print("\n=== Split Report ===")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 写 `run_all_v4.py`**

```python
#!/usr/bin/env python3
"""Run the full identity-aware train/val split pipeline for cli_v4."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_step(step_name: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 70}")
    print(f"  STEP: {step_name}")
    print(f"  CMD:  {' '.join(str(c) for c in cmd)}")
    print(f"{'=' * 70}\n")
    subprocess.run(cmd, check=True, cwd=str(HERE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=min(32, (os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--mmseqs", type=str, default=os.environ.get("MMSEQS", "mmseqs"))
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=Path("/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4"),
    )
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-cluster", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    args = parser.parse_args()

    fasta_path = HERE / "sequences.fasta"
    chain_index_path = HERE / "chain_index.json"
    cluster_tsv_30 = HERE / "mmseqs_30" / "clusters_cluster.tsv"
    cluster_tsv_40 = HERE / "mmseqs_40" / "clusters_cluster.tsv"

    if not args.skip_extract:
        if chain_index_path.is_file() and fasta_path.is_file():
            print("[skip] Step 1: sequences already extracted")
        else:
            run_step("Step 1: Extract chain sequences", [
                sys.executable,
                str(HERE / "extract_sequences_v4.py"),
                "--output-dir", str(HERE),
            ])

    if not args.skip_cluster:
        if cluster_tsv_30.is_file() and cluster_tsv_40.is_file():
            print("[skip] Step 2: clusters already exist")
        else:
            run_step("Step 2: MMseqs2 clustering (30% + 40%)", [
                sys.executable,
                str(HERE / "run_mmseqs_v4.py"),
                "--fasta", str(fasta_path),
                "--output-dir", str(HERE),
                "--mmseqs", args.mmseqs,
                "--threads", str(args.threads),
            ])

    if not args.skip_split:
        print("[info] Step 3 will always re-compute the split (idempotent, fast)")
        run_step("Step 3: Build homology-aware split", [
            sys.executable,
            str(HERE / "build_split_v4.py"),
            "--output-dir", str(HERE),
            "--final-dir", str(args.final_dir),
            "--seed", str(args.seed),
            "--val-ratio", str(args.val_ratio),
        ])

    print(f"\n{'=' * 70}")
    print("  Pipeline complete.")
    print(f"  Output:  {HERE}/split_manifest.json")
    print(f"  Output:  {HERE}/sampling_weights.pt")
    print(f"  Output:  {HERE}/split_report.json")
    print(f"  Copied:  {args.final_dir}/split_manifest.json")
    print(f"  Copied:  {args.final_dir}/sampling_weights.pt")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 语法检查（4 个文件）**

Run:
```bash
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m py_compile \
  experiments_cli_v4/split_training_val/extract_sequences_v4.py \
  experiments_cli_v4/split_training_val/run_mmseqs_v4.py \
  experiments_cli_v4/split_training_val/build_split_v4.py \
  experiments_cli_v4/split_training_val/run_all_v4.py
```

Expected: 退出码 0，无输出。

- [ ] **Step 6: 写 `README.md`**

```markdown
# split_training_val (v4) — identity-aware train/val split

完全沿用 v3 协议：

1. `extract_sequences_v4.py`：从 v4 `seq_cache.pkl` 提取 chain 序列。
2. `run_mmseqs_v4.py`：MMseqs2 `easy-cluster` @ 30% / 40% identity。
3. `build_split_v4.py`：chain-level 同源图 → 连通分量 → 贪心分配（9:1，
   seed 42）→ `split_manifest.json` + `sampling_weights.pt`。

一键运行：

```bash
python run_all_v4.py --threads 32
```

输出复制到 `/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4/`。

注意：v4 case id 带 `_G1` 等分组后缀，必须重新聚类，不能复用 v3 的 split。
```

- [ ] **Step 7: 写 `HANDOFF20260808.md`**

```markdown
# Train/Val Split 交接文档（v4）— 2026-08-08

## 状态

- 4 个脚本已写好，`py_compile` 通过。
- **未运行** extract / MMseqs2 / build_split。

## 与 v3 的差异

1. 数据源改为 `training/cli_v4/{dataset}/chunk/`，case id 带分组后缀
   （`_G1` 等），chain 集合来自 v4 分组后的 `seq_cache.pkl`。
2. 最终复制目录改为 `precomputed_mesh/cli_v4/`。
3. 协议完全不变（30%/40% easy-cluster、9:1、seed 42）。

## 待办 / 提醒

- Q-BioLiP v4 数据本地不完整，建议等数据齐再跑全量。
- v3 的 inventory reconciliation（从 manifest 移除 reader 找不到的 case）
  是后续独立步骤，本版脚本未内置；等预计算 .pt 生成后需要补一次 reconcile。
- MMseqs2 结果不可跨版本复用：v3 聚类基于 91,143 条 chain，v4 的分组 chain
  集合完全不同。
```

- [ ] **Step 8: Commit**

```bash
git add experiments_cli_v4/split_training_val/
git commit -m "feat(cli_v4): add split_training_val pipeline + docs"
```

---

### Task 6: mesh_prior HANDOFF

**Files:**
- Create: `experiments_cli_v4/mesh_prior/HANDOFF20260808.md`

不要修改 `generate_mesh_tensors_v4.py` 和 `README.md`（师姐已写）。

- [ ] **Step 1: 写 `HANDOFF20260808.md`**

```markdown
# Mesh Prior 交接文档（v4）— 2026-08-08

## 状态

- `generate_mesh_tensors_v4.py` 和 `README.md` 由师姐实现，本次**未修改**。
- 师姐已在 `precomputed_mesh/cli_v4/AFDB-10k-1-old/` 跑过 50 chunk 试点
  （9990 个 `_mesh.pt`），本任务只补交接文档。

## 转换方法（摘自 README / 脚本）

1. 顶点归属：优先读 v4 PKL 的 `{case}_point_to_residue`（全重原子 KDTree
   结果，逐顶点一个 PDB resseq）；缺失时用 `chunk/input/{case}.pdb` 的
   全重原子重算 KDTree（已验证两种方式逐顶点一致）。
2. 每残基顶点按“到最近重原子距离”升序，截断到 K=64。
3. 顶点 10 列 → 输出 9 列（drop hbond）：`x,y,z,charge,hphob,nx,ny,nz,si`。
4. 行顺序 = input PDB 每链残基首次出现顺序（与 dmap / point_to_residue 的
   resseq 一致）。

## 输出格式

```python
{
    "data":   {chain_id: torch.Tensor[N, K, 9]},
    "resseq": {chain_id: [resseq_0, ...]},
}
```

## 与 v3 的差异

- mesh 是全链全残基，不再是 interface-only。
- 不再用 `bb_input` backbone 做 KDTree；使用 v4 自带的
  `point_to_residue`（全重原子映射）。
- case id 带 `_G1` 等分组后缀。
- CLI/输出：`--dataset-dir` / `--output-root` / `--dataset-name` /
  `--chunks` / `--max-samples` / `--num-workers` / `--force`；
  summary 为 `{output_root}/{dataset_name}/generation_summary.json`。

## 待办

- 正式输出目录是 `precomputed_mesh/cli_v4/AFDB-10k-1/` 和
  `Q-BioLiP-10k-new/`（当前为空）；用 `--dataset-name AFDB-10k-1` 重新跑即可
  （`AFDB-10k-1-old` 保留不动）。
```

- [ ] **Step 2: 语法检查师姐脚本（只读确认，不修改）**

Run:
```bash
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m py_compile experiments_cli_v4/mesh_prior/generate_mesh_tensors_v4.py
```

Expected: 退出码 0，无输出。

- [ ] **Step 3: Commit**

```bash
git add experiments_cli_v4/mesh_prior/HANDOFF20260808.md
git commit -m "docs(cli_v4): add mesh_prior v4 handoff"
```

---

### Task 7: 全量验证与最终提交

**Files:**
- Run: `py_compile` 全部新脚本
- Commit: `docs/superpowers/plans/2026-08-08-cli-v4-precompute-scripts.md` + 所有新文件

- [ ] **Step 1: 全部语法检查**

Run:
```bash
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m py_compile \
  experiments_cli_v4/mesh_prior/generate_mesh_tensors_v4.py \
  experiments_cli_v4/msa/generate_msa_tensors_v4.py \
  experiments_cli_v4/pdb_cache/generate_pdb_tensors_v4.py \
  experiments_cli_v4/residue_prior/generate_residue_tensors_v4.py \
  experiments_cli_v4/split_training_val/extract_sequences_v4.py \
  experiments_cli_v4/split_training_val/run_mmseqs_v4.py \
  experiments_cli_v4/split_training_val/build_split_v4.py \
  experiments_cli_v4/split_training_val/run_all_v4.py
```

Expected: 退出码 0，无输出。

- [ ] **Step 2: 只读确认新文件都在**

Run:
```bash
find experiments_cli_v4 -maxdepth 2 -type f | sort
```

Expected:
```text
experiments_cli_v4/mesh_prior/HANDOFF20260808.md
experiments_cli_v4/mesh_prior/README.md
experiments_cli_v4/mesh_prior/generate_mesh_tensors_v4.py
experiments_cli_v4/msa/HANDOFF20260808.md
experiments_cli_v4/msa/README.md
experiments_cli_v4/msa/generate_msa_tensors_v4.py
experiments_cli_v4/pdb_cache/HANDOFF20260808.md
experiments_cli_v4/pdb_cache/README.md
experiments_cli_v4/pdb_cache/generate_pdb_tensors_v4.py
experiments_cli_v4/residue_prior/HANDOFF20260808.md
experiments_cli_v4/residue_prior/README.md
experiments_cli_v4/residue_prior/generate_residue_tensors_v4.py
experiments_cli_v4/split_training_val/HANDOFF20260808.md
experiments_cli_v4/split_training_val/README.md
experiments_cli_v4/split_training_val/build_split_v4.py
experiments_cli_v4/split_training_val/extract_sequences_v4.py
experiments_cli_v4/split_training_val/run_all_v4.py
experiments_cli_v4/split_training_val/run_mmseqs_v4.py
```

- [ ] **Step 3: 确认未写入输出目录**

Run:
```bash
find /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4/AFDB-10k-1 /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4/Q-BioLiP-10k-new -type f | wc -l
```

Expected: `0`（不运行生成，不写任何 `.pt`）。

- [ ] **Step 4: 最终提交**

```bash
git add docs/superpowers/plans/2026-08-08-cli-v4-precompute-scripts.md experiments_cli_v4/
git commit -m "feat(cli_v4): add precompute generators and docs (msa/pdb/residue/split) + mesh handoff"
```

---

## Self-Review（实现前检查）

1. **Spec 覆盖**：spec 的 msa / pdb_cache / residue_prior / split_training_val
   各模块均有对应任务；mesh_prior 只补 HANDOFF（Task 6）；不写 adapter；
   不运行生成（Task 7 Step 3 校验）；输出布局统一。
2. **占位符**：所有代码块完整，无 TBD/TODO/“类似 Task N”。
3. **一致性**：四个 generate 脚本 CLI 参数一致（`--dataset-dir` /
   `--output-root` / `--dataset-name` / `--chunks` / `--max-samples` /
   `--num-workers` / `--force`）；summary 文件名互不冲突
   （`generation_summary.json` / `msa_generation_summary.json` /
   `pdb_generation_summary.json` / `residue_generation_summary.json`）。
