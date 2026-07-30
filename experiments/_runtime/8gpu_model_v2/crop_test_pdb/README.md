# 7v6c_1 crop 可视化输出

## 这次要看的三个 PDB

- `7v6c_1_gt_original.pdb`
  - 原始 GT PDB 的直接 copy。
  - 来源：`/101045/jinyu/PPI/Prior_out/training/Q-BioLiP-10k-new/chunk/chunk0033/gt/7v6c_1.pdb`

- `7v6c_1_old_crop_B_one_residue_4atoms.pdb`
  - 旧 generic crop 的失败形态模拟。
  - 关闭 `physppi.interface_preserving_crop`，使用 generic OF3 crop。
  - 固定 `np.random.seed=208` 时，旧 crop 自然抽到 `strategy=spatial`，retained 结果为 `A=767 residues, B=1 residue`。
  - 为了可视化你说的 failure case，B 链只写这个 residue 的四个 backbone atoms：`N/CA/C/O`。

- `7v6c_1_new_crop_interface_preserving.pdb`
  - 新 PhysPPI dmap-aware interface-preserving crop。
  - crop attempt seed：`11789218922213766488`。
  - retained 结果为 `A=508 residues, B=260 residues`。

- `crop_summary.json`
  - 本次生成的结构化统计摘要。

上一次生成的两个不够明确的 PDB 已移到 `superseded/`，主目录里只保留这次要看的版本。

## PDB 文本校验

| 文件 | chain A atoms | chain A residues | chain B atoms | chain B residues | TER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `7v6c_1_gt_original.pdb` | 12487 | 1540 | 2057 | 260 | 2 |
| `7v6c_1_old_crop_B_one_residue_4atoms.pdb` | 6121 | 767 | 4 | 1 | 2 |
| `7v6c_1_new_crop_interface_preserving.pdb` | 4076 | 508 | 2057 | 260 | 2 |

旧失败版的 chain B 四条记录是：

```text
ATOM  13339  N   SER B 136      84.156 103.287  89.306  1.00 68.85           N
ATOM  13340  CA  SER B 136      85.013 103.461  90.501  1.00 67.82           C
ATOM  13341  C   SER B 136      84.225 103.249  91.799  1.00 61.87           C
ATOM  13342  O   SER B 136      84.874 103.165  92.856  1.00 75.72           O
```

## dmap interface residue 保留情况

| crop | chain A | chain B |
| --- | ---: | ---: |
| old failure crop | 83 / 178 = 46.63% | 1 / 142 = 0.70% |
| new interface crop | 178 / 178 = 100% | 142 / 142 = 100% |

## 结论

- 旧 generic crop 的 seed `208` 可以自然产生 `B=1 residue` 的极端情况；这就是你想看的“chain B 只剩一个 residue 四个原子”的失败形态。
- 新 PhysPPI crop 在同样 `768` token budget 下保留了 chain B 的全部 `260` residues，并完整保留了本 case 的 dmap interface residues。
- 这次 PDB 是从原始 GT PDB 行按 crop residue selection 过滤写出的，保留原始 chain ID 列，并显式插入 `TER`，不再依赖 Biotite 默认 writer。
