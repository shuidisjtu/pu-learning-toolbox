# 分布漂移感知 PU 补充清单

> 本清单是 [`distribution_shift_aware_pu.md`](distribution_shift_aware_pu.md) 的执行账本。
> `[x]` 表示代码、测试和文档均已有证据；不能以“接口已预留”代替完成。

## A. 设计与契约

- [x] 核对 AISTATS 2025 原论文的联合密度比、相对权重和动态训练边界
- [x] 明确第一版只对边际 (p_t(x)/p_s(x)) 提供协变量漂移加权保证
- [x] 定义域 AUC、ESS、裁剪率、严重度和 `adaptation_ready` 语义
- [x] 定义 JSON、Markdown、CSV 三类固定产物
- [ ] 为稳定公开 API 增加版本变更记录

## B. 漂移审计与权重

- [ ] 新增源/目标域共同输入校验
- [ ] 新增 OOF 域分类 AUC，避免训练内评估
- [ ] 新增有界相对密度比和均值归一化权重
- [ ] 新增 ESS、分位数、上界触及率与覆盖警告
- [ ] 新增源/目标域 PU 标签比例对比（目标标签可选）
- [ ] 新增严格 JSON/Markdown/CSV 报告保存
- [ ] 新增相同分布、均值漂移、极端权重和错误输入测试

## C. 工作流接入

- [ ] `PUPipeline.fit_evaluate` 接受并逐折切分 `sample_weight`
- [ ] CV 训练折和最终 refit 传入权重，验证折不使用训练权重
- [ ] 新增 `ShiftAwarePUPipeline` 与组合报告
- [ ] 仅允许 `SampleWeightSupport.SUPPORTED` 分类器启动适配
- [ ] 目标域无 PU 标签时保持审计可用但 `adaptation_ready=false`
- [ ] 目标域有 `y_true` 时提供明确标为 oracle 的评估
- [ ] 新增工作流集成测试和权重传递探针测试

## D. CLI 与用户体验

- [ ] 新增 `pu-toolbox shift-audit` 子命令
- [ ] 支持 CSV 表格和 `.npy` 特征，与现有 CLI 输入约定一致
- [ ] 输出 `shift_report.json`、`shift_report.md`、`source_importance_weights.csv`
- [ ] 在 CLI 帮助和用户文档中解释“检测不等于适配”
- [ ] 新增成功旅程、缺失文件、特征不一致和非法参数测试
- [ ] 增加最小可运行示例

## E. 发布门禁

- [ ] 单元、数学、集成和 E2E 测试通过
- [ ] `ruff check` 与 `ruff format --check` 通过
- [ ] 文档链接、数学渲染、项目元数据、测试质量、skill 同步门禁通过
- [ ] 构建 wheel/sdist 并执行包检查
- [ ] 更新目录结构权威文档
- [ ] 核对本地 `main` 与 `origin/main` 差异并推送

## F. 论文级联合漂移扩展（后续研究，不属于第一版）

- [ ] 实现 (m(x,y)) 类别条件相对密度比模型和论文式 PU 风险修正
- [ ] 实现共享特征提取器下权重估计/分类器交替优化
- [ ] 分别估计并传播源域与目标域类先验及其不确定性
- [ ] 增加概念漂移、支持漂移和类先验漂移合成协议
- [ ] 与 `trPU`、`tePU`、fine-tune、MMD 和两步加权基线比较
- [ ] 在至少一个公开表格数据集上完成多随机种子 paper-like benchmark
- [ ] 通过公式金标准、退化情形和消融测试后再声明“联合漂移适配”
