# PhysPPI 实验隔离与 TOML/Slurm 提交设计

日期：2026-07-27
状态：已批准，等待实施计划

## 1. 背景

当前仓库继承了 OpenFold3 的目录布局，同时在 `scripts/dev/` 中积累了大量
PhysPPI 调试、数据审计、训练启动和结果收集脚本。该目录目前包含 59 个文件，
其中临时实验、接近生产的命令和集群提交逻辑混放，存在以下问题：

- 无法从目录位置判断代码是临时实验还是主框架能力；
- shell 提交脚本重复保存资源、路径和命令参数；
- `/101045/...` 等机器路径容易进入可提交源码；
- 实验成功后缺少明确的代码晋升流程；
- 新实验继续进入 `scripts/dev/` 会扩大历史负担。

本设计借鉴 IntPEMP 的“配置、实验实现、核心包、输出”分层思路，但不复制其
未纳入 Git 管理的运行目录和历史脚本。PhysPPI 采用渐进式迁移，不一次性整理全部
`scripts/dev/`。

## 2. 目标

1. 在仓库根目录建立独立的 `experiments/` 调试实验区。
2. 使用 TOML 描述实验运行、Slurm 资源和提交设置。
3. 使用一个 Python CLI 统一校验配置、生成 `sbatch` 文件和提交任务。
4. 默认只 dry-run，只有显式 `--submit` 才能提交。
5. 保留 OpenFold3 现有 YAML 训练配置体系；TOML 只引用 YAML，不复制其内部参数。
6. 支持可提交基础配置与 Git 忽略的本机覆盖。
7. 定义实验代码晋升到 `openfold3/` 的门禁。
8. 首批只迁移排列对齐相关的三个调试脚本。

## 3. 非目标

- 不在本阶段将 OpenFold3 的模型、数据集或训练 YAML 改写为 TOML。
- 不一次性移动或重命名全部 `scripts/dev/` 文件。
- 不把 Slurm 调度逻辑放入 `openfold3/` 核心包。
- 不在 TOML 中提供任意 shell 片段或 shell 插值能力。
- 不改变排列对齐、数据审计或特征构建算法。
- 不提交运行输出、日志、本机路径配置或集群凭据。

## 4. 方案选择

### 4.1 采用的方案

采用根目录实验隔离区：

```text
configs/
  experiments/
    permutation_alignment/
      audit_protected_chain_matching.toml
      build_minimal_alignment_arrays.toml
      audit_v1_v32_data_errors.toml
  slurm/
    cpu_debug.toml
    single_gpu_debug.toml
  local/
    README.md

experiments/
  README.md
  __init__.py
  permutation_alignment/
    README.md
    __init__.py
    audit_protected_chain_matching.py
    build_minimal_alignment_arrays.py
    audit_v1_v32_data_errors.py

scripts/physppi/
  experiment_config.py
  experiment_slurm.py
  submit_experiment.py

tests/
  experiments/
  scripts/physppi/
```

`submit_experiment.py` 是唯一面向用户的提交入口。
`experiment_config.py` 负责严格配置模型、加载、合并和环境变量展开；
`experiment_slurm.py` 负责确定性脚本生成、状态记录和 `sbatch` 调用。

### 4.2 未采用的方案

将调试代码放入 `openfold3/projects/.../experiments/` 会使未验证实现进入发行包，
不符合“实验成功后再同步到主框架”的要求。

只在 `scripts/dev/` 上增加 TOML launcher 虽然改动较小，但不会解决临时脚本与
稳定命令混放的问题。

## 5. 依赖边界

依赖方向固定为：

```text
experiments  -> openfold3
scripts      -> experiments / openfold3
openfold3    -X-> experiments
```

- `experiments/` 可以导入和调用 `openfold3/`。
- `scripts/physppi/` 可以调用实验模块或主框架。
- `openfold3/` 不得导入 `experiments/` 或 `scripts/`。
- `experiments/` 不加入 `pyproject.toml` 的 setuptools package include，不随
  OpenFold3 wheel 发布。
- 测试通过仓库根目录运行实验模块，不要求实验代码成为安装包的一部分。

## 6. 配置分层

### 6.1 实验 TOML

`configs/experiments/<topic>/<run>.toml` 是可提交的实验运行描述。它保存：

- schema 版本；
- 实验名称、run 名称和 Python 模块；
- 参数数组；
- 可选的 OpenFold3 runner YAML 路径；
- Slurm profile 路径及本实验覆盖；
- 输出目录；
- 必需环境变量和允许导出的环境变量。
- 需要在提交前确认存在的输入文件和目录。

它不得复制 runner YAML 内部的模型、数据集、优化器、batch size、学习率或训练
周期等字段。YAML 仍是这些参数的唯一来源。

示例：

```toml
schema_version = 1

[experiment]
name = "permutation-alignment"
run_name = "audit-protected-chain-matching-v1"
module = "experiments.permutation_alignment.audit_protected_chain_matching"
args = [
  "--runtime-ledger", "${PHYSPPI_FORMAL_ROOT}/manifests/runtime-failure-ledger.jsonl",
  "--qbiolip-root", "${PHYSPPI_QBIOLIP_ROOT}",
  "--afdb-root", "${PHYSPPI_AFDB_ROOT}",
  "--formal-root", "${PHYSPPI_FORMAL_ROOT}",
  "--components-cif", "${PHYSPPI_COMPONENTS_CIF}",
  "--output", "${PHYSPPI_OUTPUT_ROOT}/experiments/permutation_alignment/audit-v1/audit.jsonl",
]

[openfold]
runner_yaml = "examples/training_yamls/physppi_training.yml"

[slurm]
profile = "configs/slurm/cpu_debug.toml"
job_name = "ppi-align-audit"

[artifacts]
output_dir = "${PHYSPPI_OUTPUT_ROOT}/experiments/permutation_alignment/audit-v1"

[environment]
required = [
  "PHYSPPI_QBIOLIP_ROOT",
  "PHYSPPI_AFDB_ROOT",
  "PHYSPPI_FORMAL_ROOT",
  "PHYSPPI_COMPONENTS_CIF",
  "PHYSPPI_OUTPUT_ROOT",
]

[validation]
required_files = [
  "${PHYSPPI_FORMAL_ROOT}/manifests/runtime-failure-ledger.jsonl",
  "${PHYSPPI_COMPONENTS_CIF}",
]
required_directories = [
  "${PHYSPPI_QBIOLIP_ROOT}",
  "${PHYSPPI_AFDB_ROOT}",
  "${PHYSPPI_FORMAL_ROOT}",
]
```

`[openfold]` 整段可省略。不使用 runner YAML 的三个首批实验只保留实际需要的
输入；示例中的该段用于说明未来训练实验的引用方式。

### 6.2 Slurm profile

`configs/slurm/*.toml` 保存可复用、可提交的资源模板，例如：

```toml
schema_version = 1

[slurm]
nodes = 1
ntasks_per_node = 1
cpus_per_task = 8
memory = "64G"
time = "04:00:00"
```

GPU profile 可以额外声明 `gpus_per_node`。`partition`、`account` 和 `qos` 为可选
字段；没有可移植默认值时由本机覆盖提供。

launcher 只接受 schema 明确列出的 Slurm 字段，不提供 `raw_directives`。

### 6.3 输入路径声明

launcher 无法从通用 argv 判断某个路径是输入还是输出，因此实验 TOML 使用可选的
`[validation]` 显式声明：

- `required_files`：提交前必须存在的普通文件；
- `required_directories`：提交前必须存在的目录。

这些列表只承担 preflight 校验职责，不参与命令构造。实验模块仍需在打开输入时
执行自己的业务校验。路径在环境变量展开后、生成 run 目录前检查。

### 6.4 本机覆盖

`configs/local/*.toml` 被 Git 忽略，只能通过
`--local-config configs/local/<name>.toml` 显式加载，不自动探测。

本机覆盖只允许修改：

- `[slurm]` 中的资源、partition、account 和 qos；
- `[environment]` 中的必需变量列表与导出变量。

它不能修改实验名称、run 名称、Python 模块、参数数组或 runner YAML，因此同一
实验的行为不会被本机文件静默替换。

### 6.5 合并规则

优先级固定为：

```text
Slurm profile < 实验 TOML < 显式本机覆盖
```

合并采用以下确定性规则：

- mapping 递归合并；
- scalar 由高优先级值替换；
- list 整体替换，不做追加；
- 字段类型改变时报错；
- 未知字段时报错；
- `schema_version` 不参与覆盖。

## 7. launcher 接口

默认 dry-run：

```bash
python scripts/physppi/submit_experiment.py \
  configs/experiments/permutation_alignment/audit_protected_chain_matching.toml
```

使用本机覆盖：

```bash
python scripts/physppi/submit_experiment.py \
  configs/experiments/permutation_alignment/audit_protected_chain_matching.toml \
  --local-config configs/local/cluster.toml
```

真实提交：

```bash
python scripts/physppi/submit_experiment.py \
  configs/experiments/permutation_alignment/audit_protected_chain_matching.toml \
  --local-config configs/local/cluster.toml \
  --submit
```

不提供隐式提交模式。`--submit` 不得通过 TOML 或环境变量开启。

## 8. launcher 数据流

1. 从仓库根目录解析实验 TOML。
2. 严格校验 schema 和字段类型。
3. 加载实验引用的 Slurm profile。
4. 加载显式指定的本机覆盖。
5. 按固定优先级合并配置。
6. 检查所有 `[environment].required` 变量。
7. 展开字符串中的 `${NAME}`；不支持 shell 默认值、命令替换或嵌套表达式。
8. 验证实验 module、runner YAML 和 `[validation]` 声明的输入路径。
9. 确认输出目录不是文件系统根目录且尚不存在。
10. 使用当前 Python 解释器构造：

   ```text
   sys.executable -m <experiment.module> <experiment.args...>
   ```

11. 通过字段 allowlist 生成确定性的 `job.sbatch`。
12. 写入 prepared 运行记录。
13. dry-run 到此结束并返回成功。
14. `--submit` 模式调用 `sbatch --parsable job.sbatch`。
15. 只有解析到合法 job ID 后才写入 submitted 运行记录。

命令始终由 argv 数组构造，并使用 `shlex.join()` 进行可显示、可执行的 shell 引号
转义。Python 代码调用 `subprocess` 时不使用 `shell=True`。

## 9. 运行产物

每次运行使用配置中唯一的 `artifacts.output_dir`，已存在时直接失败，不覆盖旧
实验。目录结构为：

```text
outputs/experiments/<experiment>/<run_name>/
  experiment.toml
  slurm-profile.toml
  local-overrides.redacted.toml
  runner-yaml.sha256
  command.json
  job.sbatch
  prepared.json
  submitted.json
  stdout.log
  stderr.log
```

细节：

- `runner-yaml.sha256` 仅在配置引用 runner YAML 时生成。
- `local-overrides.redacted.toml` 保留表、键和容器结构，但递归遇到的所有 scalar
  叶子值均写成 `<redacted>`，不保存本机路径、账户或环境变量值。
- `prepared.json` 包含 schema 版本、UTC 时间、Git commit、Git dirty 状态、
  配置文件哈希、最终 argv 和生成脚本哈希。
- `submitted.json` 只在 `sbatch --parsable` 成功并返回合法 job ID 后生成。
- stdout/stderr 路径写入 `job.sbatch`，由 Slurm 作业创建。
- 整个 `outputs/` 已由根 `.gitignore` 排除。

dry-run 会生成 prepared 产物，但不会要求本机安装 Slurm，也不会创建
`submitted.json`。

## 10. 失败处理

- 缺少环境变量、未知字段、错误类型、无效 module、缺失输入或 runner YAML 时，
  在生成 Slurm 脚本前失败。
- run 目录已存在时失败，不进行部分覆盖。
- 生成 prepared 产物时使用临时目录，全部写入成功后再原子重命名为最终目录。
- `sbatch` 不存在、返回非零或输出不是合法 job ID 时，命令返回非零。
- 提交失败保留 prepared 目录，并将经过清理的 stderr 写入
  `submission-error.json`；不得写入 `submitted.json`。
- job 脚本使用 `set -euo pipefail`，并显式切换到记录的仓库根目录。
- launcher 不判断作业训练是否成功；作业完成状态属于后续监控功能，不在本阶段
  范围内。

## 11. Python 版本

OpenFold3 支持 Python 3.10 及以上：

- Python 3.11+ 使用标准库 `tomllib`；
- Python 3.10 使用条件依赖
  `tomli>=2; python_version < "3.11"`。

除该兼容依赖外，不引入新的配置框架。

## 12. 首批脚本迁移

以下文件通过保留 Git 历史的移动操作进入
`experiments/permutation_alignment/`：

```text
scripts/dev/physppi_audit_protected_chain_matching.py
scripts/dev/physppi_build_minimal_alignment_arrays.py
scripts/dev/physppi_audit_v1_v32_data_errors.py
```

迁移规则：

- 删除 `scripts/dev/` 下的原路径，不保留兼容 wrapper；
- 保持审计、扫描和数组构建算法不变；
- 保持每个模块的 `main()` 和直接运行能力；
- 为每个模块创建一个实验 TOML；
- 更新文档和任何仓库内引用；
- 新入口使用 `python -m experiments.permutation_alignment.<module>`。

`physppi_audit_protected_chain_matching.py` 当前包含 `/101045/...` 硬编码。迁移时
增加显式 CLI 参数：

- `--qbiolip-root`
- `--afdb-root`
- `--formal-root`
- `--components-cif`
- 可选的 `--reference-molecules`

索引缓存、manifest 和 runtime ledger 可以从 `formal-root` 推导，也可以通过已有
明确参数覆盖。上述路径由 TOML 的参数数组和 `${PHYSPPI_*}` 环境变量提供。该变化
只替换路径来源，不改变 matcher 或 fallback 逻辑。

## 13. 渐进迁移规则

- 本阶段不整理其余 `scripts/dev/`。
- `scripts/dev/` README 增加说明：新的 PhysPPI 调试实验必须进入根
  `experiments/`。
- 后续一次只迁移一个主题目录，并为其单独编写设计和实施计划。
- 稳定的操作命令进入 `scripts/physppi/`。
- 可复用业务逻辑进入 `openfold3/`。
- 一次性诊断、假设验证和尚未稳定的代码留在 `experiments/`。

## 14. 实验晋升

实验代码晋升到主框架需要同时满足：

1. 能稳定复现目标问题；
2. 修复在目标数据及回归样本上通过；
3. 不依赖硬编码机器路径；
4. 可复用逻辑具有正式单元测试；
5. Ruff 和相关 pytest 通过；
6. 实验 README 记录结论、配置和对应核心提交。

晋升时移动实现，不复制实现：

- 可复用逻辑移入合适的 `openfold3/` 模块；
- 正式测试移入 `openfold3/tests/`；
- `experiments/` 保留薄的复现入口、TOML 和结论文档；
- 薄入口只能调用已晋升 API，不保存分叉实现。

## 15. 测试设计

### 15.1 launcher 单元测试

- TOML schema 成功解析；
- 未知字段和错误类型被拒绝；
- profile、实验配置、本机覆盖的优先级；
- list 整体替换与类型改变拒绝；
- 缺失环境变量报错；
- 只展开 `${NAME}`；
- module、输入路径和 runner YAML 校验；
- argv 构造和 shell 转义；
- 确定性 `job.sbatch` 快照；
- 已存在 run 目录拒绝；
- 本机覆盖归档完全脱敏。

### 15.2 提交行为测试

- 默认 dry-run 不调用 `sbatch`；
- dry-run 在没有 Slurm 的环境中成功；
- 假 `sbatch` 返回合法 job ID 时生成 `submitted.json`；
- 非零退出、空输出和非法 job ID 均返回失败；
- 提交失败不生成 `submitted.json`；
- prepared 目录和错误记录在失败后可用于诊断。

### 15.3 实验回归测试

- protected matcher audit 的 ledger 解析、终态记录和原子输出；
- minimal alignment arrays 的 FASTA 生成、NPZ 复制和缺失输入失败；
- v1-v32 audit 的 PDB 字段扫描和分类聚合；
- 移除硬编码路径前后使用同一 fixture 得到相同业务结果。

根 `tests/` 加入 pytest testpaths；主框架测试继续保留在
`openfold3/tests/`。

## 16. 验收标准

实施完成必须满足：

1. 新目录结构与第 4 节一致；
2. 三个旧脚本已从 `scripts/dev/` 移出；
3. 三个实验均有可提交 TOML；
4. TOML 不复制 runner YAML 内部训练字段；
5. 可执行源码、TOML 和 shell 文件中不含 `/101045/...` 硬编码；设计文档允许将
   该路径作为需要消除的历史示例；
6. launcher 默认 dry-run，只有 `--submit` 调用 `sbatch`；
7. dry-run 生成完整、可审计的 prepared 产物；
8. 所有新增测试、相关排列对齐测试和 Ruff 检查通过；
9. README 说明新实验的创建与晋升流程；
10. 实施提交推送到 `origin/main` 前，本地与远端目标状态再次核对。
