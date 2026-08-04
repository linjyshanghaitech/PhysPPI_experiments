"""Contracts for the canonical PhysPPI-first root README."""

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
README_PATH = REPOSITORY_ROOT / "README.md"


def read_readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def test_readme_is_physppi_first_and_preserves_upstream_attribution() -> None:
    text = read_readme()
    assert text.startswith("# PhysPPI")
    assert "基于 OpenFold3" in text
    assert "https://github.com/aqlaboratory/openfold-3" in text
    assert "https://www.nature.com/articles/s41586-024-07487-w" in text
    assert "CITATION.cff" in text
    assert "LICENSE" in text


def test_readme_has_the_approved_top_level_information_groups() -> None:
    text = read_readme()
    required = (
        "## 项目定位",
        "## 完整训练流程",
        "## 环境配置",
        "## 网络架构",
        "## 数据集与特征契约",
        "## 数据筛查",
        "## 训练脚本",
        "## 从头启动训练",
        "## 重要参数",
        "## 监控、错误处理与恢复",
        "## 输出与证据",
        "## OpenFold3 上游、许可与引用",
        "## 备份与恢复索引",
    )
    missing = [heading for heading in required if heading not in text]
    assert not missing, missing


def test_readme_omits_unapproved_status_and_placeholder_content() -> None:
    text = read_readme()
    assert "## 当前训练状态" not in text
    assert "stage1-afdb-v31" not in text
    assert "stage1-afdb-v32" not in text
    placeholder_pattern = re.compile(
        r"(?:T\x42D|T\x4fDO|implement\x20later|fill\x20in\x20details|"
        r"\x3c\s*(?:placeholder|fill\s+me|replace\s+me)\s*\x3e)",
        flags=re.IGNORECASE,
    )
    assert placeholder_pattern.search(text) is None


def test_readme_uses_the_verified_cluster_assets() -> None:
    text = read_readme()
    required_paths = (
        "/101045/jinyu/PhysPPI/openfold-3",
        "/101045/jinyu/PhysPPI/openfold-3/.pixi/envs/openfold3-cuda12",
        "/101045/jinyu/.openfold3/of3-p2-155k.pt",
        "/101045/jinyu/PPI/Prior_out/training/Q-BioLiP-10k-new",
        "/101045/jinyu/PPI/Prior_out/training/AFDB-10k-1",
        "/101045/jinyu/PhysPPI/openfold-3/output/physppi_msa_cache/components.cif",
        "/101045/jinyu/PhysPPI/runs/physppi_formal_8gpu/reader-index-cache",
        "/101045/jinyu/PhysPPI/runs",
    )
    missing = [path for path in required_paths if path not in text]
    assert not missing, missing


def test_readme_contains_exactly_two_physppi_mermaid_diagrams() -> None:
    text = read_readme()
    assert text.count("```mermaid") == 2
    assert text.count("```") % 2 == 0
    required_symbols = (
        "InterfaceResidueEncoder",
        "InteractionPairEncoder",
        "InteractionDistanceHead",
        "SurfaceTokenEncoder",
        "TMTopologyEncoder",
        "DirectedMembranePlaneEstimator",
        "s_init",
        "z_init",
        "delta_s",
        "delta_z",
    )
    missing = [symbol for symbol in required_symbols if symbol not in text]
    assert not missing, missing


def test_readme_states_the_label_and_residual_boundaries() -> None:
    text = read_readme()
    required = (
        "continuous dmap 不进入网络",
        "interface residue、surface 和 TM 只生成 single residual",
        "surface 和 TM 只生成 single residual",
        "精确零初始化",
        "只在 input embedding 阶段注入一次",
        "pDockQ 是唯一 PhysPPI-specific assessment",
        "normal 和反向 normal 不等价",
        "pEnergy",
        "Delta-G",
    )
    missing = [fragment for fragment in required if fragment not in text]
    assert not missing, missing


def test_readme_documents_every_supported_physppi_operation() -> None:
    text = read_readme()
    scripts = (
        "scripts/physppi/screen_full_dataset.py",
        "scripts/physppi/prepare_semantic_50epoch.py",
        "scripts/physppi/launch_semantic_training.py",
        "scripts/physppi/monitor_semantic_training.py",
        "scripts/physppi/validate_semantic_checkpoint.py",
    )
    missing = [script for script in scripts if script not in text]
    assert not missing, missing


def test_readme_full_screen_command_uses_the_production_contract() -> None:
    text = read_readme()
    flags = (
        "--qbiolip-root",
        "--afdb-root",
        "--qbiolip-index-cache",
        "--afdb-index-cache",
        "--qbiolip-canonical-report",
        "--afdb-canonical-report",
        "--history-runs-root",
        "--manifests-dir",
        "--runtime-ledger",
        "--prior-audit",
        "--components-cif",
        "--reference-molecule-directory",
        "--output-root",
        "--workers 8",
        "--sample-timeout-seconds 600",
        "--resume",
    )
    missing = [flag for flag in flags if flag not in text]
    assert not missing, missing
    assert "screened-training-manifest.json" in text
    assert "data-error-ids.json" in text
    assert "COMPLETE" in text
    assert "SHA256SUMS" in text


def test_readme_marks_the_safe_launcher_as_the_only_supported_entry() -> None:
    text = read_readme()
    assert "唯一受支持的训练启动入口" in text
    assert text.count("唯一受支持的训练启动入口") == 1


def test_readme_explains_the_canonical_epoch_arithmetic() -> None:
    text = read_readme()
    required = (
        "4096 / 8 ranks / 2 accumulation = 256 optimizer updates",
        "512 micro-steps/rank",
        "48 diffusion replicas",
        "12,800 optimizer updates",
        "50 epochs",
    )
    missing = [fragment for fragment in required if fragment not in text]
    assert not missing, missing


def test_readme_records_the_canonical_training_parameters() -> None:
    text = read_readme()
    required = (
        "BF16 mixed",
        "token budget",
        "768",
        "surface neighbors",
        "128",
        "PDE",
        "1e-4",
        "experimentally-resolved",
        "0.0",
        "Adam",
        "5e-5",
        "(0.9, 0.95)",
        "512 optimizer updates",
        "EMA",
        "0.999",
        "gradient norm clip",
        "10.0",
        "save_top_k=-1",
        "last.ckpt",
    )
    missing = [fragment for fragment in required if fragment not in text]
    assert not missing, missing


def test_readme_documents_smoke_before_formal_and_checkpoint_validation() -> None:
    text = read_readme()
    smoke = text.index("### 八卡 smoke")
    formal = text.index("### 50-epoch formal")
    assert smoke < formal
    assert "--verify-prefix-against" in text
    assert "--prefix-micro-steps 64" in text
    assert "--expected-global-step 32" in text
    assert "--manifest-sha256" in text


def test_raw_run_openfold_command_is_explanation_only() -> None:
    text = read_readme()
    assert text.index("scripts/physppi/launch_semantic_training.py") < text.index(
        "run_openfold train"
    )
    raw_index = text.index("run_openfold train")
    context = text[max(0, raw_index - 240) : raw_index + 240]
    assert "不要直接执行" in context


def test_readme_documents_owned_process_and_exact_id_recovery() -> None:
    text = read_readme()
    required = (
        "PID、PGID、cwd、command 和 run directory",
        "TERM",
        "等待 60 秒",
        "KILL",
        "exact ID",
        "same-manifest resume",
        "manifest-changing quarantine",
        "optimizer、scheduler、EMA",
    )
    missing = [fragment for fragment in required if fragment not in text]
    assert not missing, missing


def test_readme_preserves_every_recovery_identity() -> None:
    text = read_readme()
    required = (
        "backup/physppi-qbiolip-refactor-dirty-20260724",
        "86bb689a37ccfcf7a69e83e0251d3554a98f1262",
        "29d9159b53e6146ff56d304353e02a8d2dd70b28",
        "backup/physppi-canonical-checkout-dirty-20260724",
        "506622c0cbe0bc4c59bf8eb6ad2ae8b47d10ff86",
        "2ed3fc099f34b916246af398e4cf5c72e6423518",
        "backup/origin-main-before-physppi-refactor-20260724",
        "archive/origin-main-before-physppi-refactor-20260724",
        "336b3a9177427e108a0b563147133c86dfcb3815",
        "67d26777422a6bd54bfdf4ee46f8a49c3855f932",
        "/101045/jinyu/PhysPPI/openfold-3.pre-refactor-20260724",
        "/101045/jinyu/.config/superpowers/worktrees/openfold-3/physppi-preservation-20260724",
        "origin/backup/physppi-qbiolip-refactor-dirty-20260724",
        "origin/backup/physppi-canonical-checkout-dirty-20260724",
        "origin/backup/origin-main-before-physppi-refactor-20260724",
    )
    missing = [fragment for fragment in required if fragment not in text]
    assert not missing, missing


def test_readme_relative_links_resolve() -> None:
    text = read_readme()
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    missing: list[str] = []
    for target in links:
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = (REPOSITORY_ROOT / target.split("#", 1)[0]).resolve()
        if not path.exists():
            missing.append(target)
    assert not missing, missing


def test_readme_has_no_embedded_secret() -> None:
    text = read_readme()
    secret_pattern = re.compile(r"(?:ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})")
    assert secret_pattern.search(text) is None
