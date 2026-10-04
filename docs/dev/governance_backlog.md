# 治理待办：架构治理方案的剩余阶段

> **来源**：外部治理方案《PU Learning Toolbox 架构治理与代码表达重构方案》
> （原文件在仓库外、随时可能被清理；本文件转抄自 2026-10-03 最后修改的那一版）。
> 本文件只收录其中**尚未完成**的部分——阶段 6
> 与阶段 7 的未完项。方案其余部分（阶段 0–5 与阶段 3 的全部批次）已完成，逐批记录
> 回填在 [`architecture_principles.md`](architecture_principles.md) §5（权威），治理机制
> 与决策见 [ADR-0001](../adr/0001-architecture-governance.md)。
>
> **本文件的用法**：只列未完项；某项完成后从本表删除该条，不在此处留「已完成」的历史
> （历史归 §5）。故本文件只应缩短，不应增厚——要新增治理事项，先确认它确实属于原方案的
> 阶段 6/7。
>
> **任务与出口条件为方案原文的转抄**（仅按本仓库惯例把中文引号改为「」），改动方案措辞
> 前须回查原文。

## 阶段 6：Survey 与生产工作流隔离复核

**状态**：未启动（2026-10-04 核）。

**目标**：在完成基础治理后复核实验层是否把实验协议泄漏到通用工作流。

**任务**：

1. 明确 `experiment/` 只承担研究协议、版本化制品和实验留痕；
2. 明确 `workflows/` 只承担用户端到端训练和报告；
3. 明确 CLI 只做薄封装，算法选择由 registry/advisor/model_selection 负责；
4. 检查 Survey runner 不改变通用 estimator 公共契约；
5. 将通用算法测试和 Survey 协议测试分开；
6. 对 `method_ledger` 的镜像字段继续保留契约测试，不重复扩展成第二套 registry；
7. 对 P2/P3 未发布事项逐项核验真实代码状态，不以旧文档状态直接判定完成。

**出口条件**：

- Survey 变更不会默认触发通用工作流全量回归；
- 通用包基础导入不加载可选实验依赖；
- 实验制品身份和状态闭集测试保持通过；
- 架构文档中的「未发布」项与真实代码状态一致。

**开工前须由项目所有者定**：

1. 任务 7 的「P2/P3 未发布事项」以哪份文档为权威清单（`docs/research/pu_survey/` 的交付与
   复核文档，还是协议本身）——不定范围，该项会退化成无边界的全库扫描；
2. 出口条件第 1、2 条是**可测但现状未知**的命题，按「先勘察实测、再定 spec」处理，还是
   先出一版判据草案供审。

## 阶段 7：收口、发布与持续治理

方案规定 7 项，其中 4 项已落地：批次交付记录与文档同步见 §5 各行；复跑触发规则见
[`architecture_principles.md`](architecture_principles.md) 末行；PR 模板的注释门禁检查项见
`../../.github/pull_request_template.md` 的「验证」块。剩余 3 项：

- [ ] **注释门禁从 advisory 升级为分区 strict**。现状：`scripts/check_comment_quality.py`
  默认把非工具类行尾注释报为 advisory（`--strict-inline` 是局部迁移辅助，尚未对任何目录
  启用）；分区迁移策略见 [`comment_governance.md`](comment_governance.md) §4。
- [ ] **对稳定模块冻结重构范围**。现状：[`architecture_principles.md`](architecture_principles.md)
  §4 已有「稳定模块标识」原则（长期无提交视为稳定、不迁移不重构），但没有稳定模块清单，
  冻结范围未落到具体模块。
- [ ] **复核门禁本身：能否失败、是否误报、是否覆盖新增文件**。勘察线索（2026-10-04 实测，
  **均未裁定**，不等于缺陷判定）：
  - `scripts/check_api_docs.py` 不在 `README.md` 的门禁块、也不在
    `../../.github/workflows/tests.yml` 的步骤里，但被 [`process_checklist.md`](process_checklist.md)
    的「质量门禁」一行计入；
  - `scripts/check_survey_recipe_registry.py` 已登记 README 并自陈未接入 CI（属已声明的范围，
    不是漏登记）；
  - `scripts/generate_structure.py` 的生成范围只含 `pu_toolbox/`、`tests/`、`scripts/`
    （`GENERATABLE_ROOTS`，且只收 `.py`），`project_structure.md` §5 的 `docs/` 树块不在其中。
    该块曾与磁盘脱节（`dev/` 缺 4 篇、`user/` 缺 5 篇、`research/` 缺 2 篇方法卡；树内无
    过期名），已按「如实写明范围」的方式修掉：这几处补全到与磁盘一致，`research/pu_survey/`
    与 `adr/` 保持分组并指向各自索引，范围写进 §5。**仍开口**：`docs/` 是否也纳入生成器——
    生成器当前按后缀收文件（`.py`），要覆盖 `docs/` 须先泛化这一口径。
