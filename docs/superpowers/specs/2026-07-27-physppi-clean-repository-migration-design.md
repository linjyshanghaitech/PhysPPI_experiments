# PhysPPI 全新代码仓库迁移设计

## 目标

以干净的源代码树
`/101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3` 为基线，在
`/101045/jinyu/PhysPPI_jinyu/PhysPPI` 创建一个全新的 Git 仓库。

新仓库需要满足以下要求：

- 在 `main` 分支上建立全新的 Git 历史；
- 保留复现当前实现所需的完整源码、测试、环境定义和必要文档；
- 排除生成输出、缓存、checkpoint、运行状态和历史调试材料；
- 导入过程中不改变运行时行为；
- 将源提交 `4c1c97d13` 记录为代码来源。

本次迁移不配置 remote。

## 源仓库状态

源仓库在以下提交上保持 clean：

```text
4c1c97d13 docs(physppi): refresh canonical project handoff
```

该提交比上游 remote 领先 9 个文档提交。本次导入使用当前 `HEAD`，而不是
`origin/main`，因为当前根目录文档和仓库约定属于已选定的基线。

只有被该提交跟踪的文件才有资格导入。被忽略或未跟踪的文件系统内容绝不能作为
导入来源。

## 导入方式

从源提交使用 Git allowlist 导出。与直接复制文件系统相比，这种方式不会带入本地
缓存或未跟踪文件；与先完整导出再按 blacklist 删除相比，这种方式能够明确限定
最终仓库边界。

导入过程不得复制源仓库的 `.git` 目录或任何源 remote 配置。

## 保留内容

当以下路径存在于源提交中时，将其导入。

### 运行时代码和测试

- `openfold3/`
- `scripts/`
- `examples/`

完整保留 `openfold3/tests/` 及其测试数据。测试 fixture 是由源码管理的复现输入，
不是生成输出。

### 构建和环境定义

- `pyproject.toml`
- `pixi.toml`
- `pixi.lock`
- `docker/`
- `environments/`
- `deepspeed_configs/`
- `devtools/`

### 仓库元数据

- `.dockerignore`
- `.gitattributes`
- `.git-blame-ignore-revs`
- `.github/`
- `.readthedocs.yaml`
- `CITATION.cff`
- `CONTRIBUTING`
- `LICENSE`
- `README.md`

导入的 `.gitignore` 将按照下文规则加固。

### 文档

保留 OpenFold3 通用文档：

- `docs/Makefile`
- `docs/README.md`
- `docs/environment.yml`
- `docs/imgs/`
- `docs/source/`

仅保留下列 PhysPPI 文档：

- `docs/physppi/baseline_inference.md`
- `docs/physppi/full_msa_source_and_file_structure.md`
- `docs/physppi/of3_training_data_contract.md`
- `docs/physppi/qbiolip_manifest_contract.md`
- `docs/physppi/refactor/module-map.md`
- `docs/physppi/verification.md`

这些文件描述稳定接口或当前模块结构。历史 handoff、运行报告、review 和恢复记录
全部排除。

## 排除内容

不得导入：

- `.git/`；
- `.pixi/`、`.venv/`、`__pycache__/`、`.pytest_cache/`、`.ruff_cache/`；
- `output/`、`outputs/`、`logs/`、`runs/`、`remote/`、`scratch/`、`work/`；
- `*.ckpt`、`*.pt`、`*.pth` 等 checkpoint 和模型产物；
- `state/` 和 `.superpowers/`；
- `CURRENT_STATE.md`、`HANDOFF.md`、`INDEX.md` 和 `RUNBOOK.md`；
- 源仓库的 `docs/superpowers/` 历史；
- 带日期的 PhysPPI 状态报告、review、handoff、恢复文档、会议报告和历史训练记录；
- 根目录 `assets/`。其中包含不参与构建、测试或根 README 的论文和宣传媒体。

源仓库当前在 `output/` 下跟踪了大约 71 MB 文件。即使这些文件已被源仓库的 Git
跟踪，也必须排除。

目标仓库中的迁移设计文档
`docs/superpowers/specs/2026-07-27-physppi-clean-repository-migration-design.md`
是排除源文档规则的唯一例外。它记录新仓库的来源，并非从旧仓库导入。

## 忽略规则

新 `.gitignore` 必须保留有效的上游规则，并额外排除：

```gitignore
# 本地环境和缓存
.pixi/
.venv/
.pytest_cache/
.ruff_cache/
**/__pycache__/

# 运行数据和生成产物
output/
outputs/
logs/
runs/
remote/
scratch/
state/
work/

# 模型产物和本地配置
*.ckpt
*.pt
*.pth
*.log
.env
.env.*
```

不得全局忽略已被跟踪的 `.npz`、`.cif` 和 alignment 文件等测试 fixture。

## 排列对齐审阅结论

本次迁移不删除链级 permutation alignment。

此前建议删除它的依据包含一个正确但不充分的观察：prediction token 和
ground-truth token 来自同一个已处理结构，因此一定存在按源顺序对应的 identity
mapping。但这并不能推出该 identity mapping 就是模型输出在几何上的正确分配。

对于相同链，PDB chain name 是任意标签。mini-rollout 可能把某个相同链 token block
对应的坐标生成在另一个相同链所占据的物理位置上。`token_index` 只能标识 token
block，不能约束生成坐标最终占据哪个对称等价的物理位置。

AlphaFold3 Supplementary Methods 4.2 明确包含这一过程：在 confidence 和
diffusion supervision 之前，使用 mini-rollout prediction 对对称的 ground-truth
链和 ligand 进行 permutation。因此，它并不是一个未经重新审视、原样遗留的
AlphaFold2 组件。

当前源代码已经在训练路径中用 protected matching 替代了脆弱的 legacy greedy
assignment：

- sequence 和 molecule type compatibility 是硬约束；
- 使用确定性的 Hungarian assignment 代替无界或不安全的 matching；
- candidate permutation 只有在 RMSD 优于 identity baseline 时才会被接受；
- typed matcher failure 使用 sequence-compatible identity fallback；
- 保留配置中的 loss weight；
- 保留 Kabsch alignment 和 component 内部的 atom symmetry resolution。

因此，后续工作的正确基线是：

1. 保留 protected chain matching；
2. 保留 Kabsch coordinate alignment；
3. 保留 residue 和 ligand 的 atom-level symmetry resolution；
4. 在证据表明 `mol_*` label 构造仍有缺陷时，独立调查该问题；
5. 只有存在明确 consumer 或 diagnostic contract 时，才给 GT 添加 `asym_id`、
   `entity_id` 或 `sym_id`，不能用它们代替 symmetry-aware chain matching。

以后如需简化 alignment，必须先提供能够失败的 regression fixture，并与当前
protected identity baseline 进行比较。

## 行为保持

初始导入不改写现有源码和文档中针对当前机器的默认值或操作路径。虽然这些路径降低
了代码在当前环境以外的可移植性，但在仓库导入时修改它们会把行为变更和来源基线的
建立混在一起。

路径可移植性、README 通用化和运行时配置清理将在新仓库中作为后续独立变更处理，
并分别测试。

## 验证要求

导入提交只有在满足以下条件后才能视为完成：

1. 确认源仓库仍然 clean，且位于 `4c1c97d13`。
2. 确认目标仓库没有 Git remote，也没有源仓库历史。
3. 确认目标仓库中的所有文件都位于已批准的 allowlist 内。
4. 确认被排除的产物和状态路径不存在。
5. 确认没有被跟踪的文件超过 GitHub 单文件 100 MB 限制。
6. 确认没有跟踪 checkpoint 或包含环境 secret 的文件。
7. 确认可以从目标代码树导入 `openfold3` package。
8. 运行聚焦于 permutation 和 protected-chain matching 的测试。
9. 运行一组有代表性的仓库测试；这些测试不得依赖外部数据集、checkpoint、GPU
   或网络服务。
10. 在导入提交后确认 `git status --short --branch` clean。

如果验证失败揭示的是源代码行为缺陷，而不是导入缺陷，应记录该问题，不得在迁移
提交中静默修改运行时行为。
