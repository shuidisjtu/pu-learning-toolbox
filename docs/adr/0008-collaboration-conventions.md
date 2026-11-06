# ADR-0008:协作与文档惯例

- 状态:已接受
- 触发复审:协作方式变化(如外部贡献者增多),或文档导航混乱时

## 背景

双人 + agent 协作项目,需约定分支、文档组织与任务分工,避免冲突与
导航混乱。

## 决策

1. **分支规范**:代码改动开 `feature/<name>` 或 `fix/<name>`,提 PR 合并到
   main;不在 main 直接开发。
2. **docs 受众分层**:docs/ 按 user/(旅程化)/dev/(开发者)/research/(方法卡)/
   project_management/(过程)分层,参考 scikit-learn 受众分离;docs 全中文、
   根 README 双语。
   > 2026-08-13 修订:project_management/ 层解散并入 dev/,见 ADR-0013。
3. **论文分工**:shuidisjtu 负责基础 6 篇(Elkan-Noto/uPU/nnPU/PNU/Centroid/
   LLSVM),HENG958 负责扩展 6+3 篇(penL1/ReCPE/Dist-PU/PUSB/LBE/Self-PU
   + InfoMax/WConPU/DGPU)。
4. **Phase 重整**:核心 PU 风险估计优先(Elkan-Noto → uPU → nnPU → ReCPE),
   经典分类器包装器后移;阶段定义以 `process_checklist.md` 为准。
5. **Survey 实验协作（2026-11-06 修订）**：实验分工与结果审核任务由研究团队在本地任务材料中维护。
   工具箱能力与当前实验状态见 [Survey README](../research/pu_survey/README.md)，
   科学规则见[实验协议](../research/pu_survey/pu_survey_protocol.md)。

## 备选方案

- **main 直接提交**:main 稳定性无保障。否决。
- **文档平铺**:22 篇平铺索引无法区分受众(已实际重构)。否决。

## 后果

- 协作并行无冲突;文档导航按受众可预期。
- `docs/adr/` 目录是第 2 项(受众分层)的补充,不推翻受众分层。
