# ADR-0021：研究设置的语义维度与共享边界

- 日期：2026-11-06
- 状态：已采用语义与职责边界；运行时接口按工作包逐项实施
- 关联：[ADR-0018](0018-experiment-layer-public-api.md)、
  [ADR-0020](0020-main-input-label-semantics.md)

## 背景

Survey 形成了可复用的采样、风险池和选模知识。`case_control`、`os_or_ts` 和
`class_prior` 在不同入口分别承担数据构造、损失角色和不同分布参数的含义。
统计设计通过采样来源、角色和参数目标共同描述，兼容迁移保留已有实验身份。
[当前语义基线](../dev/pu_research_semantics.md)记录已确认的差异与复现。

## 决策

1. 公共知识分别描述输入采样、标记机制、抽样政策、风险池、先验目标/来源、
   验证监督、表征及评测分布。OS/TS 输入与历史 `os_or_ts` 风险适配分别解释。
2. 沿用 `core.training_views` 的中立角色与来源索引；通用采样构造放在 preprocessing，
   同义先验换算在 prior 收敛。确有两个调用方共享相同语义时才提取实现。
   方法能力事实由 estimator/registry 维护，Survey 配方和校准准入留在 experiment。
3. 先验值先确定目标与估计池，再按消费者契约处理。OS 的质量恒等式可用于合法目标的换算，
   SAR 估计依据所选估计器的可辨识条件，第三方估计依据显式目标声明。
   ReCPE 等重组估计器专项核对目标回换。
4. `PUPipeline` 保留日常画像、推荐、先验处理与内部 CV；`ExperimentRunner` 保留四角色、
   候选轨迹、独立选模与最终测试。通用层沿既有依赖方向使用下层接口。
5. 当前接口与拟议能力分开登记。旧入口保持已定义行为；新能力用准确名称，后续改变通过
   回归测试、兼容边界和迁移文档实施。历史 manifest 按原记录解释。

同池定额标记使用 `make_fixed_count_labels`，独立 TS 高斯构造使用
`make_gaussian_case_control_data`。旧 `case_control` 名字保留原输出与随机轨迹，
以 `FutureWarning` 提示迁移，计划在 2.0 移除。

先验估计器通过 `estimate_target`、`estimation_pool` 声明目标与估计池；分类器通过
`class_prior_target` 及可选的 `risk_class_prior_target` 声明消费目标。
`prior.semantics` 统一 OS/TS 质量换算。Pipeline 的显式 `class_prior` 和 `PriorInfo.value`
采用总体目标，每折与最终重训按分类器默认风险视图换算。
`PriorInfo` 保存输入值/目标、估计池/细节、采样设计、换算及 `training_consumption`；
Pipeline JSON schema 为 1.1。ReCPE 保留完整 U，并将所选 U 复制到正类 component，
以 regrouping 近似记录估计结果。

## 备选方案

- 将历史 `ts` 统一为独立 TS：需要为已有 OS→TS-compatible 风险池建立新的兼容名称。
- 对所有自动估计值做 U→总体换算：前置条件是各估计器均声明原生 U 目标，迁移时逐项确认。
- 理论和能力集中在 Survey：公共使用路径通过 Survey 入口查阅研究知识，通用复用路径需要额外桥接。

## 后果

公共概念从[用户入口](../user/README.md)提供研究阅读路径；
代码、文档和后续迁移以[语义基线](../dev/pu_research_semantics.md)作为审查依据。
采样/Gaussian/先验链修复分别验证，随后再收敛共享原语和组合示例。
阶段与验证结果由[进度清单](../dev/process_checklist.md)维护。
