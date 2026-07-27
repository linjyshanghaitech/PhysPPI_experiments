# Source Provenance

本仓库的初始运行时代码从以下 clean Git 提交按 allowlist 导入：

- 源目录：`/101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3`
- 源提交：`4c1c97d13f3fbda1184f003e9147f7b122c7c5a4`
- 导入日期：`2026-07-27`

未导入源仓库的 Git 历史、remote、运行输出、缓存、checkpoint、运行状态和历史
调试文档。初始导入不改变运行时行为；机器路径可移植性和模型行为修改将在新仓库
中作为独立变更处理。

## 迁移验证说明

迁移时的聚焦排列对齐测试和代表性 CPU 集成测试均通过。全量
`ruff check openfold3 scripts` 在
`openfold3/tests/core/data/tools/test_rscb.py:19` 报告一个未使用的 `pytest`
import（F401）；相同问题已在源提交中复现，因此保留为源基线问题，没有在迁移提交
中修改。源码基线提交之后另建独立提交删除该未使用 import，以确保导入的 GitHub
Ruff workflow 的 lint 阶段可以通过；该清理不改变运行时或测试行为。

同一 GitHub workflow 的 `ruff format --check` 阶段在源提交中还发现 47 个未格式化
文件。完整源 blob 已保存在源码基线提交 `44ce89f`；其后使用同一锁定环境中的 Ruff
对这些文件进行机械格式化，并在独立提交中保存，以使新仓库的格式检查通过。
