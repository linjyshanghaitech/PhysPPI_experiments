# PhysPPI 全新代码仓库迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 将源仓库提交 `4c1c97d13f3fbda1184f003e9147f7b122c7c5a4`
按已批准的 allowlist 导入全新 PhysPPI 仓库，验证后推送到
`https://github.com/linjyshanghaitech/PhysPPI_experiments.git`。

**架构：** 通过 `git archive` 只导出源提交中被明确允许的路径，从机制上排除
未跟踪文件、缓存和旧 Git 历史。迁移只调整仓库边界和忽略规则，不修改运行时行为；
验证使用文件清单检查、GitHub 大小限制检查、包导入和聚焦 CPU 测试。

**技术栈：** Git、Bash、Python 3、Pixi、Pytest、Ruff、OpenFold3。

---

## 文件结构

### 新增或导入

- `openfold3/`：运行时代码及完整测试。
- `scripts/`：数据处理、训练和运维脚本。
- `examples/`：可复现配置示例。
- `docker/`、`environments/`、`deepspeed_configs/`、`devtools/`：环境和构建支持。
- `.github/`：GitHub 元数据和 CI 定义。
- `docs/source/`、`docs/imgs/`：OpenFold3 通用文档。
- `docs/physppi/`：规格中批准的稳定 PhysPPI 文档。
- `SOURCE_PROVENANCE.md`：源提交、导入范围和行为边界。

### 修改

- `.gitignore`：保留上游规则并增加运行产物、缓存、checkpoint 和本地配置排除项。

### 保留

- `docs/superpowers/specs/2026-07-27-physppi-clean-repository-migration-design.md`
- `docs/superpowers/plans/2026-07-27-physppi-clean-repository-migration.md`

## Task 1：锁定源和目标状态

**文件：**

- 不修改文件。

- [ ] **Step 1：验证源提交和工作树**

运行：

```bash
test "$(git -C /101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3 rev-parse HEAD)" \
  = "4c1c97d13f3fbda1184f003e9147f7b122c7c5a4"
test -z "$(git -C /101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3 status --porcelain)"
```

预期：两个命令退出码均为 0，且没有输出。

- [ ] **Step 2：验证目标只包含设计和计划历史**

运行：

```bash
git status --short --branch
git remote -v
git log --oneline --decorate
```

预期：分支为 `main`，除本实施计划尚未跟踪外没有其他工作树变更，没有 remote；
历史仅包含设计及其中文化提交。

- [ ] **Step 3：提交实施计划**

```bash
git add docs/superpowers/plans/2026-07-27-physppi-clean-repository-migration.md
git commit -m "docs: add clean repository migration plan"
```

预期：生成只包含实施计划的提交。

## Task 2：按 allowlist 导入源代码

**文件：**

- 创建：规格“保留内容”中列出的所有路径。
- 不导入：规格“排除内容”中列出的所有路径。

- [ ] **Step 1：从固定源提交导出运行时代码和仓库元数据**

在目标仓库运行：

```bash
git -C /101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3 archive \
  4c1c97d13f3fbda1184f003e9147f7b122c7c5a4 \
  .dockerignore .gitattributes .git-blame-ignore-revs .github \
  .gitignore .readthedocs.yaml CITATION.cff CONTRIBUTING LICENSE README.md \
  deepspeed_configs devtools docker environments examples openfold3 \
  pixi.lock pixi.toml pyproject.toml scripts |
tar -x -C /101045/jinyu/PhysPPI_jinyu/PhysPPI
```

预期：只创建命令中列出的 tracked 路径，不覆盖目标 `.git/`。

- [ ] **Step 2：导入通用文档和精选 PhysPPI 文档**

```bash
git -C /101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3 archive \
  4c1c97d13f3fbda1184f003e9147f7b122c7c5a4 \
  docs/Makefile docs/README.md docs/environment.yml docs/imgs docs/source \
  docs/physppi/baseline_inference.md \
  docs/physppi/full_msa_source_and_file_structure.md \
  docs/physppi/of3_training_data_contract.md \
  docs/physppi/qbiolip_manifest_contract.md \
  docs/physppi/refactor/module-map.md \
  docs/physppi/verification.md |
tar -x -C /101045/jinyu/PhysPPI_jinyu/PhysPPI
```

预期：目标 `docs/superpowers/` 中已有的新仓库设计和计划不受影响，源仓库的旧
`docs/superpowers/` 不会导入。

- [ ] **Step 3：检查未暂存边界**

运行：

```bash
git status --short
```

预期：只显示 allowlist 路径和随后需要修改的 `.gitignore`。

## Task 3：加固仓库卫生规则并记录来源

**文件：**

- 修改：`.gitignore`
- 创建：`SOURCE_PROVENANCE.md`

- [ ] **Step 1：在 `.gitignore` 增加仓库级排除规则**

保留导入的上游内容，并追加：

```gitignore

# Local environments and caches
.pixi/
.venv/
.pytest_cache/
.ruff_cache/
**/__pycache__/

# Runtime data and generated artifacts
/output/
/outputs/
/logs/
/runs/
/remote/
/scratch/
/state/
/work/

# Model artifacts and local configuration
*.ckpt
*.pt
*.pth
*.log
.env
.env.*
```

- [ ] **Step 2：创建来源记录**

创建 `SOURCE_PROVENANCE.md`，内容为：

```markdown
# Source Provenance

本仓库的初始运行时代码从以下 clean Git 提交按 allowlist 导入：

- 源目录：`/101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3`
- 源提交：`4c1c97d13f3fbda1184f003e9147f7b122c7c5a4`
- 导入日期：`2026-07-27`

未导入源仓库的 Git 历史、remote、运行输出、缓存、checkpoint、运行状态和历史
调试文档。初始导入不改变运行时行为；机器路径可移植性和模型行为修改将在新仓库
中作为独立变更处理。
```

- [ ] **Step 3：验证忽略规则**

运行：

```bash
mkdir -p output .pytest_cache scratch
touch output/probe.ckpt .pytest_cache/probe scratch/probe.log
git status --short --ignored output .pytest_cache scratch
rm -rf output .pytest_cache scratch
```

预期：三个 probe 均显示 `!!`，删除后不残留文件。

## Task 4：验证导入内容

**文件：**

- 不修改运行时代码。

- [ ] **Step 1：验证 allowlist 和 denylist**

生成目标顶层清单并确认没有禁止路径：

```bash
find . -maxdepth 1 -mindepth 1 -printf '%f\n' | sort
test ! -e output
test ! -e assets
test ! -e state
test ! -e CURRENT_STATE.md
test ! -e HANDOFF.md
test ! -e INDEX.md
test ! -e RUNBOOK.md
```

预期：所有 `test` 均退出 0。

- [ ] **Step 2：检查 GitHub 文件大小和禁止扩展名**

运行：

```bash
find . -path ./.git -prune -o -type f -size +99M -print
find . -path ./.git -prune -o -type f \
  \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' \) -print
```

预期：两个命令都没有输出。

- [ ] **Step 3：检查常见 secret 文件**

运行：

```bash
find . -path ./.git -prune -o -type f \
  \( -name '.env' -o -name '.env.*' -o -name '*.pem' -o -name 'id_rsa*' \) \
  -print
```

预期：没有输出。

- [ ] **Step 4：验证导入文件与源提交 blob 一致**

除 `.gitignore` 和目标仓库新建文档外，逐文件比较源提交 blob：

```bash
SOURCE=/101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3
SHA=4c1c97d13f3fbda1184f003e9147f7b122c7c5a4
while IFS= read -r path; do
  case "$path" in
    .gitignore|SOURCE_PROVENANCE.md|docs/superpowers/*) continue ;;
  esac
  source_oid=$(git -C "$SOURCE" rev-parse "$SHA:$path")
  target_oid=$(git hash-object "$path")
  test "$source_oid" = "$target_oid"
done < <({ git ls-files; git ls-files --others --exclude-standard; } | sort -u)
```

预期：所有比较均通过，退出码为 0。

- [ ] **Step 5：验证包导入**

使用源仓库已有的 CPU 环境执行目标代码：

```bash
PYTHONPATH=/101045/jinyu/PhysPPI_jinyu/PhysPPI \
  /101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3/.pixi/envs/openfold3-cpu/bin/python \
  -c 'import openfold3; print(openfold3.__file__)'
```

预期：输出路径位于目标仓库的 `openfold3/__init__.py`。

- [ ] **Step 6：运行排列对齐聚焦测试**

```bash
PYTHONPATH=/101045/jinyu/PhysPPI_jinyu/PhysPPI \
  /101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3/.pixi/envs/openfold3-cpu/bin/python \
  -m pytest -q \
  openfold3/tests/core/utils/test_permutation_alignment.py \
  openfold3/tests/projects/of3_all_atom/physppi/data/test_protected_chain_matching.py
```

预期：全部通过。

- [ ] **Step 7：运行代表性仓库测试和 Ruff**

```bash
PYTHONPATH=/101045/jinyu/PhysPPI_jinyu/PhysPPI \
  /101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3/.pixi/envs/openfold3-cpu/bin/python \
  -m pytest -q \
  openfold3/tests/core/data/framework/test_stochastic_sampler_dataset.py \
  openfold3/tests/projects/of3_all_atom/physppi/integration/test_end_to_end_prior_smoke.py

/101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3/.pixi/envs/openfold3-cpu/bin/ruff \
  check openfold3 scripts
```

预期：测试全部通过。Ruff 在
`openfold3/tests/core/data/tools/test_rscb.py:19` 报告一个源提交中可复现的未使用
`pytest` import（F401）；在 `SOURCE_PROVENANCE.md` 中记录该基线问题，不在迁移
提交中修改模型行为。

## Task 5：提交并推送

**文件：**

- 提交所有批准导入和新增的文件。

- [ ] **Step 1：检查 staged 内容**

```bash
git add .
git diff --cached --check -- \
  .gitignore SOURCE_PROVENANCE.md docs/superpowers
git status --short
```

预期：本次手工编辑的文件没有 whitespace error；staged 内容不包含 denylist
路径。完整导入内容不运行全量 `git diff --cached --check`，因为新仓库会将源提交中
既有的文档和 CIF 尾随空格全部显示为新增行错误；这些文件改用逐 blob 来源一致性
检查。

- [ ] **Step 2：创建 clean baseline 提交**

```bash
git commit -m "chore: import clean PhysPPI source baseline"
```

预期：生成单一源码迁移提交。

- [ ] **Step 3：确认提交后状态**

```bash
git status --short --branch
git log --oneline --decorate -5
git remote -v
```

预期：工作树 clean，仍无 remote。

- [ ] **Step 4：配置并推送 GitHub remote**

```bash
git remote add origin https://github.com/linjyshanghaitech/PhysPPI_experiments.git
git push -u origin main
```

预期：GitHub 空仓库接受 `main`，本地 `main` 跟踪 `origin/main`。

- [ ] **Step 5：核对远端提交**

```bash
test "$(git rev-parse HEAD)" = "$(git ls-remote origin refs/heads/main | cut -f1)"
git status --short --branch
```

预期：本地和远端 `main` 指向同一提交，工作树 clean。
