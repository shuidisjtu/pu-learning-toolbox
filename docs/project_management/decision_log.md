# 决策日志

| 日期 | 决策 | 理由 | 决策人 |
|---|---|---|---|
| 2026-08-09 | v1.0.0 版本升级(0.1.0.dev0→1.0.0) | roadmap 0.1→0.6 功能全部完成(17 算法、6 门禁、738 测试);从未发布正式版,直接 1.0.0 首版 | shuidisjtu |
| 2026-08-09 | 新增第 6 道质量门禁 check_format.py(ruff check + format --check,CI 与本地单一入口) | 2026-08-09 CI 曾因本地漏跑 `ruff format --check` 失败,软约束需转硬门禁 | shuidisjtu |
| 2026-08-08 | PUSB Table 2 采用严格子集策略:manifest 锁定 6 数据集 sha256/形状/类别计数,fidelity 降级项显式声明 | 复现基准需可审计可复跑,数据漂移检测机制化 | shuidisjtu |
| 2026-08-08 | PUSBKernelClassifier 独立注册(非 LDCE 别名) | official_compatibility 有 0.5·reg 分歧,独立注册保证元数据诚实 | shuidisjtu |
| 2026-08-06 | pu-workflow 通用 skill(开放规范/双目录 SKILL.md + 中文解读指南) | 把论文复现工作流沉淀为可复用流程,双目录由 check_skill_sync 门禁保证一致 | shuidisjtu |
| 2026-08-06 | Deep PU 算法接入 Pipeline/CLI（MLP/CNN 架构选择） | PUPipeline/CLI 支持 WConPU + InfoMax PU，两级参数 `architecture`（mlp/cnn）+ `backbone`（cnn13/resnet18/resnet50）；`--data` 接受 .npy 4D NCHW 图像；auto 行为不变；DGPU/Self-PU 不接入（无单骨架插拔概念）。spec 见历史 `deep_pipeline_design.md`（已蒸馏删除） | shuidisjtu |
| 2026-07-10 | 代码改动走 feature/fix 分支，提 PR 合并到 main | main 保持稳定可运行 | shuidisjtu |
| 2026-07-16 | Phase 1/2 重整：实际执行中优先实现核心 PU 风险估计方法（Elkan-Noto → uPU → nnPU → ReCPE），经典分类器包装器（PU Bagging 等）后移至 Phase 2。阶段定义以 `process_checklist.md` 为准。 | 风险估计方法是工具箱核心差异化能力，且是后续深度方法（Dist-PU, Self-PU 等）的前置依赖；经典包装器价值在于与传统 sklearn 对比，优先级可适当降低 | shuidisjtu |
| 2026-07-11 | 论文按基础/扩展分工：shuidisjtu 负责 Elkan-Noto/uPU/nnPU/PNU/Centroid/LLSVM（6篇），HENG958 负责 penL1/ReCPE/Dist-PU/PUSB/LBE/Self-PU（6篇），剩余3篇（InfoMax/WConPU/DGPU）由 HENG958 负责。 | 基础方法先做能为扩展方法提供参考，无前置依赖，可并行 | shuidisjtu |
| 2026-08-01 | Python 支持收敛为 3.10–3.12；CI 显式锁定 matrix interpreter，分离测试、静态门禁和 wheel 安装冒烟 | 避免 `.python-version` 覆盖 CI matrix、未验证却声明 3.13，以及源码可用但发布包缺文件 | Codex / HENG958 |
| 2026-08-01 | `pyproject.toml` 作为依赖权威来源；`uv.lock` 不入库；`requirements.txt` 仅作开发环境快照 | library 需在 CI 持续验证声明范围内的可解析依赖，同时保留特定环境问题的复查依据 | Codex / HENG958 |
| 2026-08-05 | CLI 采用 argparse 单命令薄封装（run / list-methods / list-priors / make-demo-data），所有逻辑在 PUPipeline，CLI 只做参数解析/CSV IO/错误映射；辅助命令从 registry 实时读取 | 零新增依赖；与项目既有 argparse 惯例一致；新算法注册后 CLI 自动可见，无需维护 | shuidisjtu |
| 2026-08-05 | `run` 默认 auto 模式引入推荐器训练成本维度（第 7 维）+ LLSVM 收敛早停（默认开），默认 run 实测 30s → 2s | auto 推荐器原本选中固定 3000 epoch 的 LLSVM 导致默认路径 ~30s；成本维度使其 rank 6，早停使其显式调用也降至 ~7s | shuidisjtu |
| 2026-08-05 | 文档体系重构：docs/ 按受众分层（user/ 旅程化 + dev/ 开发者文档 + project_management/ 原位），参考 scikit-learn 受众分离与 Claude Code 页面串联机制；docs 全中文、根 README 双语 | 原 22 篇平铺索引无法区分受众，用户文档各自孤立、部分选型文档泛泛空话；重构后用户旅程（快速开始→概念→操作→API 参考）显式可导航，参数契约集中去重 | shuidisjtu |
