# 决策日志

| 日期 | 决策 | 理由 | 决策人 |
|---|---|---|---|
| 2026-07-10 | 代码改动走 feature/fix 分支，提 PR 合并到 main | main 保持稳定可运行 | shuidisjtu |
| 2026-07-16 | Phase 1/2 重整：实际执行中优先实现核心 PU 风险估计方法（Elkan-Noto → uPU → nnPU → ReCPE），经典分类器包装器（PU Bagging 等）后移至 Phase 2。阶段定义以 `process_checklist.md` 为准。 | 风险估计方法是工具箱核心差异化能力，且是后续深度方法（Dist-PU, Self-PU 等）的前置依赖；经典包装器价值在于与传统 sklearn 对比，优先级可适当降低 | shuidisjtu |
| 2026-07-11 | 论文按基础/扩展分工：shuidisjtu 负责 Elkan-Noto/uPU/nnPU/PNU/Centroid/LLSVM（6篇），HENG958 负责 penL1/ReCPE/Dist-PU/PUSB/LBE/Self-PU（6篇），剩余3篇（InfoMax/WConPU/DGPU）由 HENG958 负责。详见 `division.txt`。 | 基础方法先做能为扩展方法提供参考，无前置依赖，可并行 | shuidisjtu |
| 2026-08-01 | Python 支持收敛为 3.10–3.12；CI 显式锁定 matrix interpreter，分离测试、静态门禁和 wheel 安装冒烟 | 避免 `.python-version` 覆盖 CI matrix、未验证却声明 3.13，以及源码可用但发布包缺文件 | Codex / HENG958 |
| 2026-08-01 | `pyproject.toml` 作为依赖权威来源；`uv.lock` 不入库；`requirements.txt` 仅作开发环境快照 | library 需在 CI 持续验证声明范围内的可解析依赖，同时保留特定环境问题的复查依据 | Codex / HENG958 |
