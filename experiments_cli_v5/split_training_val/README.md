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
