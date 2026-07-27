# PhysPPI Experiment Isolation and TOML/Slurm Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立根目录实验隔离区、严格 TOML 配置加载器和默认 dry-run 的统一 Slurm launcher，并将三个排列对齐调试脚本迁入首个实验。

**Architecture:** `configs/` 只保存声明式实验与 Slurm 配置，`experiments/` 保存不随 wheel 发布的调试实现，`scripts/physppi/` 保存稳定的配置解析和提交入口。launcher 使用 Pydantic 严格校验、argv 数组和原子运行目录；`openfold3/` 不依赖实验或脚本层。

**Tech Stack:** Python 3.10+、Pydantic 2、`tomllib`/`tomli`、pytest、Ruff、Slurm `sbatch --parsable`

---

## 实施约束

- 设计规格：
  `docs/superpowers/specs/2026-07-27-physppi-experiment-isolation-and-toml-slurm-design.md`
- 所有测试命令从仓库根目录运行。
- 推荐命令前缀：`pixi run -e openfold3-cpu`。
- 每项实现先写失败测试，再写最小实现。
- 不修改排列对齐、matcher、fallback 或 PDB 扫描算法。
- 不保留 `scripts/dev/` 下三个旧入口的兼容 wrapper。
- 不在可执行源码、TOML 或 shell 文件中加入 `/101045/...`。

## 文件责任映射

### 新增

- `scripts/physppi/experiment_config.py`
  - TOML 读取、Pydantic schema、profile/local 合并、环境变量展开和路径 preflight。
- `scripts/physppi/experiment_slurm.py`
  - argv 构造、确定性 sbatch 渲染、原子 prepared 目录和提交状态记录。
- `scripts/physppi/submit_experiment.py`
  - 唯一用户 CLI；默认 dry-run，显式 `--submit` 才提交。
- `experiments/README.md`
  - 实验隔离和晋升规则。
- `experiments/permutation_alignment/README.md`
  - 首个实验目的、配置和结论记录方式。
- `configs/slurm/cpu_debug.toml`
  - CPU 调试资源模板。
- `configs/slurm/single_gpu_debug.toml`
  - 单 GPU 调试资源模板。
- `configs/local/README.md`
  - 本机覆盖格式及 Git 忽略规则。
- `configs/experiments/permutation_alignment/*.toml`
  - 三个首批实验配置。
- `tests/scripts/physppi/test_experiment_config.py`
  - 配置 schema、合并、展开和 preflight 测试。
- `tests/scripts/physppi/test_experiment_slurm.py`
  - 渲染、prepared 状态、提交成功和失败测试。
- `tests/scripts/physppi/test_submit_experiment.py`
  - CLI 默认 dry-run 和显式提交测试。
- `tests/experiments/permutation_alignment/*.py`
  - 三个迁移脚本的行为回归测试。

### 移动

- `scripts/dev/physppi_audit_protected_chain_matching.py`
  -> `experiments/permutation_alignment/audit_protected_chain_matching.py`
- `scripts/dev/physppi_build_minimal_alignment_arrays.py`
  -> `experiments/permutation_alignment/build_minimal_alignment_arrays.py`
- `scripts/dev/physppi_audit_v1_v32_data_errors.py`
  -> `experiments/permutation_alignment/audit_v1_v32_data_errors.py`

### 修改

- `pyproject.toml`
  - Python 3.10 的 `tomli` 条件依赖；pytest 收集根 `tests/`。
- `.gitignore`
  - 忽略 `configs/local/*.toml`，保留说明文档。
- `scripts/dev/README.md`
  - 声明旧目录冻结，不再接收新的 PhysPPI 实验。
- `scripts/physppi/README.md`
  - 记录统一 launcher 使用方式。

---

### Task 1: 建立严格 TOML 配置加载器

**Files:**
- Create: `scripts/physppi/experiment_config.py`
- Create: `tests/scripts/__init__.py`
- Create: `tests/scripts/physppi/__init__.py`
- Create: `tests/scripts/physppi/test_experiment_config.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: 先让 pytest 收集根测试并添加 Python 3.10 TOML 兼容依赖**

在 `pyproject.toml` 的运行依赖中加入：

```toml
"tomli>=2; python_version < '3.11'",
```

将 pytest 配置改为：

```toml
testpaths = ["openfold3/tests", "tests"]
```

- [ ] **Step 2: 写配置 loader 的失败测试**

`tests/scripts/physppi/test_experiment_config.py` 至少覆盖以下接口：

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.physppi.experiment_config import load_experiment


def _write_minimal_config(root: Path) -> Path:
    profile = root / "profile.toml"
    profile.write_text(
        'schema_version = 1\n'
        '[slurm]\n'
        'nodes = 1\n'
        'cpus_per_task = 4\n'
        'memory = "16G"\n'
        'time = "01:00:00"\n'
    )
    config = root / "experiment.toml"
    config.write_text(
        'schema_version = 1\n'
        '[experiment]\n'
        'name = "alignment"\n'
        'run_name = "audit-v1"\n'
        'module = "tests.fixtures.fake_experiment"\n'
        'args = ["--input", "${DATA_ROOT}/input.json"]\n'
        '[slurm]\n'
        f'profile = "{profile}"\n'
        'job_name = "align-audit"\n'
        '[artifacts]\n'
        'output_dir = "${OUTPUT_ROOT}/audit-v1"\n'
        '[environment]\n'
        'required = ["DATA_ROOT", "OUTPUT_ROOT"]\n'
        '[validation]\n'
        'required_files = ["${DATA_ROOT}/input.json"]\n'
    )
    return config


def test_load_experiment_merges_and_expands(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "input.json").write_text("{}\n")
    config = _write_minimal_config(tmp_path)

    loaded = load_experiment(
        config,
        repo_root=tmp_path,
        environ={"DATA_ROOT": str(data), "OUTPUT_ROOT": str(tmp_path / "out")},
        validate_module=False,
    )

    assert loaded.config.slurm.nodes == 1
    assert loaded.config.slurm.job_name == "align-audit"
    assert loaded.config.experiment.args[-1] == str(data / "input.json")
    assert loaded.config.artifacts.output_dir == tmp_path / "out/audit-v1"


def test_unknown_experiment_field_is_rejected(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path)
    config.write_text(config.read_text() + '\n[unknown]\nvalue = 1\n')

    with pytest.raises(ValidationError):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ={"DATA_ROOT": str(tmp_path), "OUTPUT_ROOT": str(tmp_path)},
            validate_module=False,
        )


def test_missing_required_environment_variable_is_rejected(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path)

    with pytest.raises(ValueError, match="DATA_ROOT"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ={"OUTPUT_ROOT": str(tmp_path)},
            validate_module=False,
        )
```

再添加测试验证：

- local 覆盖能替换 `partition`，不能包含 `[experiment]`；
- list 被整体替换；
- scalar/list 类型变化失败；
- 缺失 `required_files` 和 `required_directories` 分别失败；
- 未定义 `${NAME}` 失败；
- `$NAME` 保持普通字符串，不进行 shell 展开；
- runner YAML 存在时解析为仓库绝对路径；
- `importlib.util.find_spec()` 找不到 module 时失败。

- [ ] **Step 3: 运行测试确认 RED**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/scripts/physppi/test_experiment_config.py -q
```

Expected: collection fails with
`ModuleNotFoundError: scripts.physppi.experiment_config`。

- [ ] **Step 4: 实现严格 schema 和 loader**

`scripts/physppi/experiment_config.py` 定义以下公开结构：

TOML parser 会返回字符串路径和 list。strict model 在校验前只执行两种明确规范化：

```python
PathValue = Annotated[
    Path,
    BeforeValidator(lambda value: Path(value) if isinstance(value, str) else value),
]
StringTuple = Annotated[tuple[str, ...], BeforeValidator(tuple)]
PathTuple = Annotated[tuple[PathValue, ...], BeforeValidator(tuple)]
```

配置结构：

```python
class ExperimentSection(StrictModel):
    name: str
    run_name: str
    module: str
    args: StringTuple = ()


class OpenFoldSection(StrictModel):
    runner_yaml: PathValue


class SlurmResources(StrictModel):
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    nodes: int | None = Field(default=None, ge=1)
    ntasks_per_node: int | None = Field(default=None, ge=1)
    cpus_per_task: int | None = Field(default=None, ge=1)
    gpus_per_node: int | None = Field(default=None, ge=0)
    memory: str | None = None
    time: str | None = None


class ExperimentSlurmSection(SlurmResources):
    profile: PathValue
    job_name: str


class ArtifactsSection(StrictModel):
    output_dir: PathValue


class EnvironmentSection(StrictModel):
    required: StringTuple = ()
    export: dict[str, str] = Field(default_factory=dict)


class ValidationSection(StrictModel):
    required_files: PathTuple = ()
    required_directories: PathTuple = ()


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    experiment: ExperimentSection
    openfold: OpenFoldSection | None = None
    slurm: ExperimentSlurmSection
    artifacts: ArtifactsSection
    environment: EnvironmentSection = Field(default_factory=EnvironmentSection)
    validation: ValidationSection = Field(default_factory=ValidationSection)


class LoadedExperiment(NamedTuple):
    config: ExperimentConfig
    experiment_path: Path
    profile_path: Path
    local_path: Path | None
```

另外定义两个文件级 schema：

```python
class SlurmProfileFile(StrictModel):
    schema_version: Literal[1]
    slurm: SlurmResources


class LocalOverrideFile(StrictModel):
    schema_version: Literal[1]
    slurm: SlurmResources = Field(default_factory=SlurmResources)
    environment: EnvironmentSection = Field(default_factory=EnvironmentSection)
```

因此 profile 不能递归引用 profile，本机覆盖也不能出现 `profile`、`job_name`、
`experiment`、`openfold`、`artifacts` 或 `validation`。

所有 Pydantic model 使用：

```python
model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
```

提供唯一主入口
`load_experiment(experiment_path: Path, *, repo_root: Path, local_path:
Path | None = None, environ: Mapping[str, str] | None = None,
validate_module: bool = True) -> LoadedExperiment`。

实现要求：

- Python 3.11+ `import tomllib`，3.10 fallback 到 `tomli`；
- profile 路径相对仓库根解析；
- mapping 递归合并、list 整体替换、类型变化时报错；
- local schema 只允许 `schema_version`、资源型 `slurm` 和 `environment`；
- 只用正则展开 `${[A-Z_][A-Z0-9_]*}`；
- 相对 runner YAML、validation path 和 output path 相对仓库根解析；
- output path 不得等于 `/`，且加载时不得已存在；
- preflight 使用 `is_file()`、`is_dir()` 和 `find_spec()`；
- 不执行实验 module。

- [ ] **Step 5: 运行 loader 测试确认 GREEN**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/scripts/physppi/test_experiment_config.py -q
```

Expected: all tests pass。

- [ ] **Step 6: 提交配置 loader**

```bash
git add pyproject.toml scripts/physppi/experiment_config.py tests/scripts
git commit -m "feat: add strict experiment TOML loader"
```

---

### Task 2: 生成原子 prepared 目录和确定性 sbatch

**Files:**
- Create: `scripts/physppi/experiment_slurm.py`
- Create: `tests/scripts/physppi/test_experiment_slurm.py`

- [ ] **Step 1: 写渲染与 prepared 状态失败测试**

测试通过 Task 1 的 `_write_minimal_config()` 构造 `LoadedExperiment`，覆盖：

```python
def test_render_sbatch_is_deterministic(loaded, tmp_path: Path) -> None:
    first = render_sbatch(
        loaded.config,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
    )
    second = render_sbatch(
        loaded.config,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
    )
    assert first == second
    assert "#SBATCH --nodes=1" in first
    assert "set -euo pipefail" in first
    assert "shell=True" not in first


def test_prepare_run_writes_prepared_but_not_submitted(loaded, tmp_path: Path) -> None:
    prepared = prepare_run(
        loaded,
        repo_root=tmp_path,
        python_executable=Path("/usr/bin/python3"),
        git_commit="a" * 40,
        git_dirty=False,
    )
    assert prepared.run_dir.is_dir()
    assert (prepared.run_dir / "prepared.json").is_file()
    assert not (prepared.run_dir / "submitted.json").exists()
```

再添加测试验证：

- argv 使用 `sys.executable -m module`；
- 所有 Slurm 字段按固定顺序渲染；
- 环境变量 export 使用安全 shell quoting；
- existing output 拒绝；
- local TOML 的每个 scalar 叶子在归档中变为 `<redacted>`；
- 原始 experiment/profile 被复制；
- runner YAML 只记录 SHA256；
- prepared JSON 包含 commit、dirty、config hash、argv 和 sbatch hash。
- `git_state()` 使用 `git rev-parse HEAD` 和 `git status --porcelain` 返回精确状态。

- [ ] **Step 2: 运行测试确认 RED**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/scripts/physppi/test_experiment_slurm.py -q
```

Expected: collection fails with
`ModuleNotFoundError: scripts.physppi.experiment_slurm`。

- [ ] **Step 3: 实现 argv、渲染和 prepared 状态**

`scripts/physppi/experiment_slurm.py` 提供：

```python
@dataclass(frozen=True)
class PreparedRun:
    run_dir: Path
    script_path: Path
    command: tuple[str, ...]


def build_command(
    config: ExperimentConfig,
    *,
    python_executable: Path,
) -> tuple[str, ...]:
    return (
        str(python_executable),
        "-m",
        config.experiment.module,
        *config.experiment.args,
    )
```

同时实现以下精确接口：

- `render_sbatch(config: ExperimentConfig, *, repo_root: Path,
  python_executable: Path) -> str`
- `prepare_run(loaded: LoadedExperiment, *, repo_root: Path,
  python_executable: Path, git_commit: str, git_dirty: bool) -> PreparedRun`
- `git_state(repo_root: Path) -> tuple[str, bool]`

渲染顺序固定为：

```text
job-name, partition, account, qos, nodes, ntasks-per-node,
cpus-per-task, gpus-per-node, mem, time, output, error
```

脚本主体固定包含：

```bash
set -euo pipefail
cd <quoted-repository-root>
export <NAME>=<quoted-value>
exec <shlex.join(command)>
```

`prepare_run()` 在 output 的同级目录创建临时目录，写完所有文件并 `fsync` 后使用
`os.replace()` 发布最终目录。任何失败都清理临时目录，不留下半成品。

本机配置脱敏不引入 TOML writer。由于 `LocalOverrideFile` 只允许固定字段，提供
一个确定性 renderer：

```python
def redact_scalar_tree(value: object) -> object:
    if isinstance(value, dict):
        return {key: redact_scalar_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_scalar_tree(item) for item in value]
    return "<redacted>"


```

另实现
`render_redacted_local(raw: Mapping[str, object]) -> str`。renderer 按
`schema_version`、`[slurm]`、`[environment]`、
`[environment.export]` 固定顺序输出合法 TOML；数组保留元素数量，每个元素写成
`"<redacted>"`。没有 local 文件时不创建 `local-overrides.redacted.toml`。

`git_state()` 的 subprocess 调用固定使用 argv、`check=True` 和
`capture_output=True`，不使用 shell。

- [ ] **Step 4: 运行渲染测试确认 GREEN**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/scripts/physppi/test_experiment_slurm.py -q
```

Expected: prepared/render tests pass。

- [ ] **Step 5: 提交 prepared 实现**

```bash
git add scripts/physppi/experiment_slurm.py \
  tests/scripts/physppi/test_experiment_slurm.py
git commit -m "feat: render auditable Slurm experiment runs"
```

---

### Task 3: 添加显式 Slurm 提交状态和统一 CLI

**Files:**
- Modify: `scripts/physppi/experiment_slurm.py`
- Create: `scripts/physppi/submit_experiment.py`
- Create: `tests/scripts/physppi/test_submit_experiment.py`
- Modify: `tests/scripts/physppi/test_experiment_slurm.py`

- [ ] **Step 1: 写提交成功与失败测试**

使用可注入的 runner 测试：

```python
def test_submit_run_records_valid_job_id(prepared: PreparedRun) -> None:
    runner = Mock(
        return_value=subprocess.CompletedProcess(
            args=["sbatch"],
            returncode=0,
            stdout="12345;cluster\n",
            stderr="",
        )
    )
    job_id = submit_run(prepared, runner=runner)
    assert job_id == "12345"
    assert json.loads((prepared.run_dir / "submitted.json").read_text())[
        "job_id"
    ] == "12345"


@pytest.mark.parametrize("stdout", ["", "not-a-job", "123 bad"])
def test_submit_run_rejects_invalid_job_id(
    prepared: PreparedRun,
    stdout: str,
) -> None:
    runner = Mock(
        return_value=subprocess.CompletedProcess(
            args=["sbatch"], returncode=0, stdout=stdout, stderr=""
        )
    )
    with pytest.raises(RuntimeError):
        submit_run(prepared, runner=runner)
    assert not (prepared.run_dir / "submitted.json").exists()
    assert (prepared.run_dir / "submission-error.json").is_file()
```

另测 `sbatch` 不存在和非零退出。

- [ ] **Step 2: 写 CLI 默认 dry-run 测试**

`tests/scripts/physppi/test_submit_experiment.py`：

```python
def test_cli_defaults_to_dry_run(monkeypatch, minimal_config: Path) -> None:
    submit = Mock(side_effect=AssertionError("must not submit"))
    monkeypatch.setattr(
        "scripts.physppi.submit_experiment.submit_run",
        submit,
    )
    result = main([str(minimal_config)])
    assert result == 0
    submit.assert_not_called()


def test_cli_submits_only_with_explicit_flag(
    monkeypatch,
    minimal_config: Path,
) -> None:
    submit = Mock(return_value="12345")
    monkeypatch.setattr(
        "scripts.physppi.submit_experiment.submit_run",
        submit,
    )
    assert main([str(minimal_config), "--submit"]) == 0
    submit.assert_called_once()
```

- [ ] **Step 3: 运行测试确认 RED**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/scripts/physppi/test_experiment_slurm.py \
  tests/scripts/physppi/test_submit_experiment.py -q
```

Expected: `submit_run` 和 CLI module 尚不存在。

- [ ] **Step 4: 实现 `submit_run()`**

接口为 `submit_run(prepared: PreparedRun, *, runner:
Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> str`。

固定调用：

```python
runner(
    ["sbatch", "--parsable", str(prepared.script_path)],
    check=False,
    capture_output=True,
    text=True,
)
```

合法 stdout 为 `DIGITS` 或 `DIGITS;cluster`。只有合法 job ID 原子写入
`submitted.json`；其他情况写入清理后的 `submission-error.json` 并抛出
`RuntimeError`。错误清理固定执行：将 local override 和
`environment.export` 中所有非空 scalar 值替换为 `<redacted>`，然后只保留
stderr 最后 8192 个字符。状态 JSON 不记录完整 process environment。

- [ ] **Step 5: 实现唯一 CLI**

`scripts/physppi/submit_experiment.py` 首先复用仓库现有 direct-execution bootstrap：

```python
# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from scripts.physppi._bootstrap import ensure_repository_root
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _bootstrap import ensure_repository_root

ensure_repository_root()
```

随后定义 CLI：

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--local-config", type=Path)
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    loaded = load_experiment(
        args.config,
        repo_root=repo_root,
        local_path=args.local_config,
    )
    git_commit, git_dirty = git_state(repo_root)
    prepared = prepare_run(
        loaded,
        repo_root=repo_root,
        python_executable=Path(sys.executable),
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    print(prepared.script_path)
    if args.submit:
        print(submit_run(prepared))
    return 0
```

增加 `if __name__ == "__main__": raise SystemExit(main())`。

- [ ] **Step 6: 运行提交与 CLI 测试确认 GREEN**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/scripts/physppi/test_experiment_slurm.py \
  tests/scripts/physppi/test_submit_experiment.py -q
```

Expected: all tests pass。

- [ ] **Step 7: 提交 launcher**

```bash
git add scripts/physppi tests/scripts/physppi
git commit -m "feat: add explicit PhysPPI Slurm submission CLI"
```

---

### Task 4: 迁移 minimal alignment arrays 实验

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/permutation_alignment/__init__.py`
- Move: `scripts/dev/physppi_build_minimal_alignment_arrays.py`
  -> `experiments/permutation_alignment/build_minimal_alignment_arrays.py`
- Create: `tests/experiments/__init__.py`
- Create: `tests/experiments/permutation_alignment/__init__.py`
- Create: `tests/experiments/permutation_alignment/test_build_minimal_alignment_arrays.py`

- [ ] **Step 1: 写现有行为回归测试**

```python
def test_build_minimal_alignment_arrays(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping"
    main = tmp_path / "main"
    output = tmp_path / "output"
    mapping.mkdir()
    main.mkdir()
    (mapping / "rep_id_to_seq.json").write_text(
        '{"rep-b": "BBBB", "rep-a": "AAAA"}\n'
    )
    (main / "rep-a.npz").write_bytes(b"a")
    (main / "rep-b.npz").write_bytes(b"b")

    reps = load_rep_sequences(mapping)
    write_alignment_representatives_fasta(reps, output / "reps.fasta")
    copied = copy_colabfold_main_npz(reps, main, output / "arrays")

    assert (output / "reps.fasta").read_text() == (
        ">rep-a\nAAAA\n>rep-b\nBBBB\n"
    )
    assert sorted(copied) == ["rep-a", "rep-b"]
    assert (output / "arrays/rep-a.npz").read_bytes() == b"a"
```

另测空 mapping 和缺失 NPZ 抛出明确错误。

- [ ] **Step 2: 运行测试确认 RED**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/experiments/permutation_alignment/test_build_minimal_alignment_arrays.py -q
```

Expected: experiment module 不存在。

- [ ] **Step 3: 使用 Git move 迁移脚本**

```bash
mkdir -p experiments/permutation_alignment
git mv scripts/dev/physppi_build_minimal_alignment_arrays.py \
  experiments/permutation_alignment/build_minimal_alignment_arrays.py
```

使用 `apply_patch` 创建两个 `__init__.py`。保持函数和 CLI 行为不变。

- [ ] **Step 4: 运行测试确认 GREEN**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/experiments/permutation_alignment/test_build_minimal_alignment_arrays.py -q
```

Expected: all tests pass。

- [ ] **Step 5: 提交第一个迁移脚本**

```bash
git add experiments tests/experiments scripts/dev
git commit -m "refactor: isolate alignment array experiment"
```

---

### Task 5: 迁移 v1-v32 数据错误审计实验

**Files:**
- Move: `scripts/dev/physppi_audit_v1_v32_data_errors.py`
  -> `experiments/permutation_alignment/audit_v1_v32_data_errors.py`
- Create: `tests/experiments/permutation_alignment/test_audit_v1_v32_data_errors.py`

- [ ] **Step 1: 写 fixed-width PDB 扫描回归测试**

测试 helper 构造恰好 80 列的 ATOM 行：

```python
def _atom_line(*, serial: str, residue: str, charge: str) -> str:
    line = list(" " * 80)
    line[0:6] = "ATOM  "
    line[6:11] = serial
    line[12:16] = " CA "
    line[17:20] = residue
    line[21] = "A"
    line[22:26] = "   1"
    line[78:80] = charge
    return "".join(line) + "\n"


def test_scan_pdb_reports_historical_data_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "fixture.pdb"
    path.write_text(
        _atom_line(serial="    2", residue="ALA", charge="2+")
        + _atom_line(serial="    1", residue="GLY", charge="XX")
    )

    result = scan_pdb(path, surface_neighbors=1)

    assert result["atom_record_count"] == 2
    assert result["invalid_charge_field_count"] == 1
    assert result["nonincreasing_atom_id_count"] == 1
    assert result["microheterogeneous_residue_count"] == 1
    assert result["surface_neighbor_axis_collision"] is True
```

另测 `category_payload()` 去重、排序和计数。

- [ ] **Step 2: 运行测试确认 RED**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/experiments/permutation_alignment/test_audit_v1_v32_data_errors.py -q
```

Expected: experiment module 不存在。

- [ ] **Step 3: 使用 Git move 迁移且不改算法**

```bash
git mv scripts/dev/physppi_audit_v1_v32_data_errors.py \
  experiments/permutation_alignment/audit_v1_v32_data_errors.py
```

只更新 module docstring 和文档中的入口路径；不修改 `scan_pdb()`、
`category_payload()` 或 `run_scan()`。

- [ ] **Step 4: 运行回归测试确认 GREEN**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/experiments/permutation_alignment/test_audit_v1_v32_data_errors.py -q
```

Expected: all tests pass。

- [ ] **Step 5: 提交第二个迁移脚本**

```bash
git add experiments/permutation_alignment tests/experiments scripts/dev
git commit -m "refactor: isolate v1-v32 data audit experiment"
```

---

### Task 6: 迁移 protected matcher audit 并移除机器路径

**Files:**
- Move: `scripts/dev/physppi_audit_protected_chain_matching.py`
  -> `experiments/permutation_alignment/audit_protected_chain_matching.py`
- Create: `tests/experiments/permutation_alignment/test_audit_protected_chain_matching.py`

- [ ] **Step 1: 写 ledger 与原子 audit 回归测试**

```python
def test_run_protected_match_audit_writes_one_terminal_row_per_id(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "runtime.jsonl"
    ledger.write_text(
        '{"dataset":"qbiolip","sample_id":"q-1"}\n'
        '{"dataset":"afdb","sample_id":"AF-2"}\n'
    )
    output = tmp_path / "audit.jsonl"

    records = run_protected_match_audit(
        ledger,
        output,
        lambda dataset, sample_id: {
            "accepted": False,
            "reason": f"fixture:{dataset.value}:{sample_id}",
            "identity_rmsd": 2.0,
            "candidate_rmsd": 2.0,
            "matched_coverage": 4,
        },
    )

    assert len(records) == 2
    assert len(output.read_text().splitlines()) == 2
```

另测重复 ledger ID、非法 JSON、已有 output 和 non-finite RMSD。

- [ ] **Step 2: 写生产路径派生测试**

目标接口：

```python
def test_production_paths_derive_all_formal_inputs(tmp_path: Path) -> None:
    paths = ProductionPaths(
        qbiolip_root=tmp_path / "qbiolip",
        afdb_root=tmp_path / "afdb",
        formal_root=tmp_path / "formal",
        components_cif=tmp_path / "components.cif",
        reference_molecules=tmp_path / "reference",
    )
    options = paths.screening_options(tmp_path / "work")
    assert options.runtime_ledger == (
        tmp_path / "formal/manifests/runtime-failure-ledger.jsonl"
    )
    assert options.qbiolip_root == tmp_path / "qbiolip"
    assert options.afdb_root == tmp_path / "afdb"
```

- [ ] **Step 3: 运行测试确认 RED**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/experiments/permutation_alignment/test_audit_protected_chain_matching.py -q
```

Expected: experiment module 和 `ProductionPaths` 不存在。

- [ ] **Step 4: Git move 后将全局机器路径改为不可变配置**

```bash
git mv scripts/dev/physppi_audit_protected_chain_matching.py \
  experiments/permutation_alignment/audit_protected_chain_matching.py
```

新增：

```python
@dataclass(frozen=True)
class ProductionPaths:
    qbiolip_root: Path
    afdb_root: Path
    formal_root: Path
    components_cif: Path
    reference_molecules: Path

    def screening_options(self, output_root: Path) -> ScreeningOptions:
        manifests = self.formal_root / "manifests"
        return ScreeningOptions(
            qbiolip_root=self.qbiolip_root,
            afdb_root=self.afdb_root,
            qbiolip_index_cache=(
                self.formal_root / "reader-index-cache/qbiolip10k-v1.pkl"
            ),
            afdb_index_cache=(
                self.formal_root / "reader-index-cache/afdb10k-v1.pkl"
            ),
            qbiolip_canonical_report=manifests / "qbiolip-audit.json",
            afdb_canonical_report=manifests / "afdb-audit.json",
            history_runs_root=self.formal_root,
            manifests_dir=manifests,
            runtime_ledger=manifests / "runtime-failure-ledger.jsonl",
            prior_audit=(
                self.formal_root
                / "audits/v1-v32-full-20k-data-errors/data-error-ids.json"
            ),
            components_cif=self.components_cif,
            reference_molecule_directory=self.reference_molecules,
            output_root=output_root,
            workers=1,
        )
```

`_dataset_for()` 和 `production_match_evaluator()` 显式接收 `ProductionPaths`。
cache key 包含 `(ProductionPaths, DatasetName)`，防止不同配置共享 dataset。

parser 增加：

```python
parser.add_argument("--qbiolip-root", type=Path)
parser.add_argument("--afdb-root", type=Path)
parser.add_argument("--formal-root", type=Path)
parser.add_argument("--components-cif", type=Path)
parser.add_argument(
    "--reference-molecules",
    type=Path,
    default=_REPOSITORY_ROOT
    / "openfold3/core/data/resources/canonical_protein_data/reference_mols",
)
```

仅当未指定 `--evaluator` 时要求前四个路径参数；自定义 evaluator 保持原调用方式。

- [ ] **Step 5: 确认路径与 matcher 回归 GREEN**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/experiments/permutation_alignment/test_audit_protected_chain_matching.py \
  openfold3/tests/core/utils/test_permutation_alignment.py \
  openfold3/tests/projects/of3_all_atom/physppi/data/test_protected_chain_matching.py \
  -q
```

Expected: experiment tests pass；排列对齐测试保持 `46 passed, 1 skipped`。

- [ ] **Step 6: 扫描机器路径并提交**

Run:

```bash
! rg -n '/101045/' experiments scripts/physppi \
  --glob '*.py' --glob '*.sh'
```

Expected: no matches。

```bash
git add experiments/permutation_alignment tests/experiments scripts/dev
git commit -m "refactor: isolate protected chain matching audit"
```

---

### Task 7: 添加可提交配置、资源模板和使用文档

**Files:**
- Create: `configs/slurm/cpu_debug.toml`
- Create: `configs/slurm/single_gpu_debug.toml`
- Create: `configs/experiments/permutation_alignment/audit_protected_chain_matching.toml`
- Create: `configs/experiments/permutation_alignment/build_minimal_alignment_arrays.toml`
- Create: `configs/experiments/permutation_alignment/audit_v1_v32_data_errors.toml`
- Create: `configs/local/README.md`
- Create: `experiments/README.md`
- Create: `experiments/permutation_alignment/README.md`
- Create: `scripts/dev/README.md`
- Modify: `.gitignore`
- Modify: `scripts/physppi/README.md`
- Modify: `tests/scripts/physppi/test_experiment_config.py`

- [ ] **Step 1: 写仓库真实 TOML schema 测试**

参数化读取三个配置，使用 `tmp_path` 创建它们声明的必需文件/目录，并传入测试环境
变量。断言：

```python
assert loaded.config.experiment.module.startswith(
    "experiments.permutation_alignment."
)
assert loaded.config.openfold is None
assert loaded.config.slurm.profile is not None
assert "/101045/" not in config_path.read_text()
```

另断言所有 `configs/slurm/*.toml` 能被 profile schema 严格解析。

- [ ] **Step 2: 运行真实配置测试确认 RED**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/scripts/physppi/test_experiment_config.py -q
```

Expected: configs 尚不存在。

- [ ] **Step 3: 创建 Slurm profiles**

`cpu_debug.toml` 使用设计规格中的 1 node、1 task、8 CPU、64G、4 小时。
`single_gpu_debug.toml` 在相同字段上增加：

```toml
gpus_per_node = 1
```

不写不可移植的 partition、account 或 qos。

- [ ] **Step 4: 创建三个实验 TOML**

所有 TOML：

- `schema_version = 1`；
- module 指向迁移后的模块；
- 使用 `configs/slurm/cpu_debug.toml`；
- 使用 `${PHYSPPI_*}` 路径；
- 用 `[validation]` 列出必需输入；
- 不包含 `[openfold]`，因为三个审计不读取 runner YAML；
- output 进入 `${PHYSPPI_OUTPUT_ROOT}/experiments/permutation_alignment/...`。

protected audit 使用：

```text
PHYSPPI_QBIOLIP_ROOT
PHYSPPI_AFDB_ROOT
PHYSPPI_FORMAL_ROOT
PHYSPPI_COMPONENTS_CIF
PHYSPPI_OUTPUT_ROOT
```

minimal arrays 使用：

```text
PHYSPPI_MSA_CACHE
PHYSPPI_OUTPUT_ROOT
```

v1-v32 audit 使用前一组数据根目录，并从 formal root 引用 index cache、canonical
report、training manifest 和 runtime ledger。

- [ ] **Step 5: 配置本机忽略规则**

`.gitignore` 增加：

```gitignore
/configs/local/*.toml
!/configs/local/README.md
```

`configs/local/README.md` 给出不含真实值的示例，并明确通过 `--local-config` 显式
加载。

- [ ] **Step 6: 写实验与晋升文档**

`experiments/README.md` 记录依赖方向和六项晋升门禁。
`experiments/permutation_alignment/README.md` 记录三个入口、配置位置、输出位置和
“实验结论必须引用核心提交”的规则。
`scripts/dev/README.md` 明确该目录仅保留遗留上游开发脚本，新的 PhysPPI debug
代码进入 `experiments/`。
`scripts/physppi/README.md` 增加 dry-run、local override 和 `--submit` 示例。

- [ ] **Step 7: 运行配置测试和 Ruff**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/scripts/physppi/test_experiment_config.py -q
pixi run -e openfold3-cpu ruff check \
  configs experiments scripts tests
pixi run -e openfold3-cpu ruff format --check \
  experiments scripts tests
```

Expected: all checks pass。

- [ ] **Step 8: 提交配置和文档**

```bash
git add .gitignore configs experiments/README.md \
  experiments/permutation_alignment/README.md scripts/dev/README.md \
  scripts/physppi/README.md tests/scripts/physppi/test_experiment_config.py
git commit -m "feat: add reproducible permutation alignment experiments"
```

---

### Task 8: 全面验证、真实 dry-run 和推送

**Files:**
- Verify only; modify files only if a check exposes a scoped defect.

- [ ] **Step 1: 运行新增测试**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  tests/scripts/physppi \
  tests/experiments/permutation_alignment \
  -q
```

Expected: all pass。

- [ ] **Step 2: 运行排列对齐与代表性回归**

Run:

```bash
pixi run -e openfold3-cpu pytest \
  openfold3/tests/core/utils/test_permutation_alignment.py \
  openfold3/tests/projects/of3_all_atom/physppi/data/test_protected_chain_matching.py \
  openfold3/tests/core/data/framework/test_stochastic_sampler_dataset.py \
  openfold3/tests/projects/of3_all_atom/physppi/integration/test_end_to_end_prior_smoke.py \
  -q
```

Expected: no regression；CUDA-only test may remain skipped。

- [ ] **Step 3: 使用真实 minimal config 做无 Slurm dry-run**

Run:

```bash
tmp="$(mktemp -d)"
mkdir -p \
  "$tmp/msa/colabfold_msas/mappings" \
  "$tmp/msa/colabfold_msas/main"
printf '{}\n' > "$tmp/msa/colabfold_msas/mappings/rep_id_to_seq.json"
PHYSPPI_MSA_CACHE="$tmp/msa" \
PHYSPPI_OUTPUT_ROOT="$tmp/output" \
pixi run -e openfold3-cpu python \
  scripts/physppi/submit_experiment.py \
  configs/experiments/permutation_alignment/build_minimal_alignment_arrays.toml
test -f \
  "$tmp/output/experiments/permutation_alignment/build-minimal-arrays-v1/prepared.json"
test ! -e \
  "$tmp/output/experiments/permutation_alignment/build-minimal-arrays-v1/submitted.json"
rm -rf "$tmp"
```

Expected: exit 0，存在 prepared，不存在 submitted。

- [ ] **Step 4: 运行代码质量和仓库卫生检查**

Run:

```bash
pixi run -e openfold3-cpu ruff check openfold3 scripts experiments tests
pixi run -e openfold3-cpu ruff format --check openfold3 scripts experiments tests
git diff --check
! rg -n '/101045/' configs experiments scripts \
  --glob '*.toml' --glob '*.py' --glob '*.sh'
test -z "$(git ls-files configs/local | rg '\.toml$' || true)"
git status --short --branch
```

Expected: Ruff 和 diff check 通过；无机器路径；无 local TOML 被跟踪；工作树干净或
只包含本任务已解释的修改。

- [ ] **Step 5: 核对提交链**

Run:

```bash
git log --oneline --decorate -12
git status --short --branch
```

Expected: 每个 Task 有独立提交，`main` 仅领先 `origin/main` 本任务提交。

- [ ] **Step 6: 推送并核对远端哈希**

Run:

```bash
git push origin main
local_head="$(git rev-parse HEAD)"
remote_head="$(git ls-remote origin refs/heads/main | cut -f1)"
test "$local_head" = "$remote_head"
git status --short --branch
```

Expected: 本地与远端哈希一致，状态为 `main...origin/main`。
