# QBioLiP `_msa.pt` 残基覆盖缺口 — 问题与修复方案

状态:**已实施**(2026-08-19),实施细节见第 6 节。

## 1. 问题现象

- 抽样 100 个 QBioLiP `_msa.pt`:共 509 条链,**467 条** MSA 列数 = GT 残基数,
  **42 条(≈8%)** MSA 列数 < GT 残基数;AFDB 抽样 10 个全部对齐。
- 训练影响:列数不足的链在运行时 `_project_msa_to_query_tokens` 返回 `None`,
  该链退化为 query-only MSA 兜底(不崩溃,但没有真实 MSA 信息)。
- 典型例子 `4wz7_1_G2` 链 B:`_gt.pt` 438 个残基,`_msa.pt` 只有 312 列。

## 2. 根因(证据链)

1. **源数据固有 UNK 残基**:`training/cli_v4/Q-BioLiP-10k-new/raw/4wz7_1.pdb`
   链 B 共 438 个残基,其中 126 个是 `UNK`(残基 26–42 等区段,有坐标但序列身份未知);
   `seq_cache` 链 B 对应为 126 个 `X`。这是源数据本身的属性,不是预处理引入。
2. **源 MSA 其实覆盖全部 438 列**:
   `chunk/chunk0013/msa_local/main/4wz7_1_G2/4wz7_1_G2_B.npz` 的
   `colabfold_main` 形状为 `(145305, 438)`,首行在 26–42 位置就是 `X`——
   即 MSA 是按 438 长度序列(带 X 占位)搜索的,数据没有缺列。
3. **msa 生成器把 X 列裁掉了(真正的缺陷)**:
   `experiments_cli_v5/msa/generate_msa_tensors_v4.py` 中
   `_parse_gt_chain_data` 构造投影目标时**只保留标准残基**(UNK 被跳过,
   目标序列=312),`_project_msa_to_gt_residues` 把 438 列 MSA 投影到 312 列
   标准序列 → 126 个 X 列被丢弃,写出 312 列 `_msa.pt`。
4. **运行时把问题放大成整链退化**:
   `openfold3/.../adapters/_runtime.py` 的 `_project_msa_to_query_tokens` 要求
   源 MSA query 长度 ≥ 结构 query(438);312 < 438 → 返回 `None` → 整链
   query-only 兜底。即源 MSA 明明有 438 列数据,却因生成器丢 X 列,
   导致这些链完全没有真实 MSA。

## 3. 修复方案(未实施)

### 3.1 修改 msa 生成器

`experiments_cli_v5/msa/generate_msa_tensors_v4.py`:

- `_parse_gt_chain_data`:非标准残基(UNK 等)在目标序列中**保留为 `X`**,
  不再丢弃;链过滤条件从"全部是标准残基"改为"全部是标准残基或 X"。
- `_project_msa_to_gt_residues`:匹配规则支持 **X↔X** 推进
  (当前 `allow_unknown=True` 只允许"目标 X 匹配任意源残基";
  需要源 X 列遇到目标 X 时也能被选中并推进;源 X 遇到目标标准残基时不推进,
  维持现有语义)。
- 校验:投影后列数 == 目标序列长度(438),首行 X 数量与目标一致。
- 兼容性:无 UNK 的链行为不变;含 UNK 链输出列数 = GT 残基数。

### 3.2 重生成受影响 case

- 判定:链的 resseq 数量 ≠ `_gt.pt` 残基数(即含被丢弃的非标准残基)。
- 执行:对受影响 case 用 `--force` 重跑 msa 生成(msa 生成器目前按 chunk 全量
  扫描,建议写一个只处理受影响 case 的驱动脚本,避免全量重算)。
- 预期:重生成后 `_msa.pt` 列数 == `_gt.pt` 残基数,运行时 438↔438 投影成功,
  链恢复真实 MSA。

### 3.3 回归验证

- 抽样重生成文件:每链 `main_msa/profile/deletion_mean/paired_msa` 列数
  == `_gt.pt` 残基数;`resseq` == `_gt.pt` res_id;`chain_ids` == `_seq.pt`。
- 训练侧确认这些链不再走 query-only 兜底(可选:加诊断日志,涉及正式代码,
  需另行评估)。

## 4. 影响面与备选

- 影响:QBioLiP 约 8% 链(抽样估计);AFDB 无此问题;v3 数据同属此设计,
  但 v4 数据中 UNK 链占比更高。
- 备选 A:不修,接受 query-only 兜底(与 v3 行为一致)。
- 备选 B:修生成器 + 重生成受影响 case(**推荐**,恢复真实 MSA)。
- 备选 C:连运行时 adapter 一起改(允许部分投影+缺失列置 X),
  涉及正式代码,违背隔离约束,不推荐。

## 5. 决策

- 2026-08-18:暂不实施,先记录问题与方案;如后续决定修复,按第 3 节执行,
  并把修复脚本/日志放到对应实验 runtime 目录。

## 6. 实施记录(2026-08-19)

### 6.1 代码修改

- `experiments_cli_v5/msa/generate_msa_tensors_v4.py`:
  - `_parse_gt_chain_data`:CA 残基中 `UNK` 现在保留为目标序列里的 `X`
    (不再跳过);其余非标准残基(FME/TPO/RNA/配体等)维持原跳过行为
    ——因为源 MSA 对它们没有对应列,映射成 X 反而会导致投影失败。
  - `_project_msa_to_gt_residues`:确认并注释 X↔X 等值匹配与
    `allow_unknown`(目标 X 匹配任意源残基)语义,无需改逻辑。

### 6.2 再生成

- 受影响判定:gt PDB 中含 UNK CA 残基的 case。
- 扫描结果:QBioLiP **265** 个 case(AFDB 0)。
- 驱动脚本:`experiments_cli_v4/_runtime/8gpu_model_v4_fixed_crop_pdockq_auxlog_20260806_smoke250_fix_template/scripts/regenerate_affected_msa_cases.py`
  (fork 多进程,`--force` 只重写受影响 case,其余文件不动)。
- 执行:16 worker,**265/265 全部成功**,报告在
  `.../logs/msa_regenerate_report.json`。

### 6.3 验证结果(265 个 case,1461 条链)

- **1,327 条链**:真实 MSA(>1 行)、列数 = GT 残基数、resseq 与 `_gt.pt` 完全一致;
  运行时 `_project_msa_to_query_tokens` 抽样 87/88 成功。示例 `4wz7_1_G2` 链 B:
  312→**438** 列(含 126 个 X),行数 16,384。
- **122 条链**:query-only 兜底(1 行、全长)。原因是源 MSA query 与结构序列本身不匹配
  (如 `2odr_1_G1` 链 A:源 544 vs 目标 491,第 98 位起序列完全不同),属源数据质量问题,
  修复前后运行时行为一致(都兜底),无回归。
- **12 条链**:off-by-one 残差(如 `3tgu_1_G1` 链 C 379 vs 380),由插入码/非 CA 残基
  解析边界导致;修复前同样更短,无回归,暂不处理(如需修复需改 `_parse_gt_chain_data`
  的 (resseq, ins_code) 键处理,风险较大)。

### 6.4 结论

- fix.md 目标达成:含 UNK 的链不再被截断,真实 MSA 覆盖全部 GT 残基,运行时投影成功。
- 剩余两类(122 源不匹配、12 off-by-one)与本次修复无关,行为不劣于修复前。

## 7. 第二次修复:非 UNK 非标准残基纳入目标序列(2026-08-20)

### 7.1 问题

`logs/pt_problem.md` 问题 5 复查发现:msa 生成器目标仍跳过非 UNK 非标准残基
(FME/L2C/BCR/UNL 等),导致 `_msa.pt` 列数 < gt.pt(如 `3tgu_1_G1` 链 C
379 vs 380,首残基 FME 被跳过;全数据集无插入码 case,12 条 off-by-one 全部由
非标准残基导致)。

### 7.2 根因与修复

- 根因:`_parse_gt_chain_data` 只保留"标准残基 + UNK(X)",其余非标准残基跳过。
- 修复(与 pdb_cache canonize 口径对齐):
  - 按 `(chain, resseq, ins_code)` 分组统计蛋白聚合物 ATOM 残基(任意重原子,
    不只 CA);
  - 标准残基映射 AA1,MSE→M;其余(UNK/FME/L2C/UNL 等)全部映射 `X`;
  - 不再丢弃任何蛋白聚合物残基。

### 7.3 再生成

- 受影响判定:gt PDB 含非 UNK 非标准 ATOM 残基 **或含无 CA 的 ATOM 残基**
  的 case(5,569 个,QBioLiP;AFDB 0——后者是 7.5 全量校验补发现的)。
- 驱动脚本:更新 `regenerate_affected_msa_cases.py`(支持 `--resume-since`
  断点续跑);64 worker + 单线程 BLAS,5,565 个有 `_msa.pt` 的目标全部成功
  (0 失败),4 个无源 npz 跳过。
- 报告:`.../logs/msa_regenerate_report.json`(5,565 条,含续跑跳过项)。

### 7.4 验证

- `scripts/verify_msa_gt_columns.py` 逐链校验**全部 29,252 个** QBioLiP
  `_msa.pt`(7.5 后复跑):
  - 无任何"列数 < gt.pt"的链(7.5 前有 481 case / 802 链,全部为无 CA 残基
    导致);
  - 232 个 case 的 383 条链无源 `msa_local/main` npz(源覆盖缺口,运行时
    `_project_msa_to_query_tokens` 走 query-only 兜底,与修复前一致)。
- AFDB 9,990 个 `_msa.pt` 全量核对:0 问题。
- 全量审计:问题 3(r2s)808→0;问题 4(mesh)3,642→48 行(48 行为修复前就存在
  的源 mesh<gt 差异,文档问题 7/8);dmap 残留 28,012 行/159 case 全部为源
  dmap 引用结构中不存在的残基。

### 7.5 补充修复:无 CA 残基(2026-08-20,全量校验发现)

- 现象:全量校验发现 481 case / 802 链 `_msa.pt` 列数仍 < gt.pt(如
  `1brd_1_G1` 链 A gt=177、msa=170,7 个标准残基只有侧链原子、无 CA)。
- 根因:旧生成器只统计 CA 残基;第一轮受影响扫描只查非标准残基名,漏掉
  "标准残基名但无 CA"的 case,故未纳入再生成集合。
- 修复:受影响判定增加"存在无 CA 的 ATOM 残基";新增 249 个 case 全部重生成
  成功(0 失败),全量复跑后列数偏短清零。
