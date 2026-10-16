# ADR 0020：按声明检查主输入标签语义

- 日期：2026-10-15
- 状态：已实现，2026-10-16 用户接受技术复核证据，RV2 完成
- 依据：[标签语义设计](../dev/label_semantics_plan.md)、[Issue #41](https://github.com/shuidisjtu/pu-learning-toolbox/issues/41)

## 背景

PU的0表示未标记，PN的0表示真实负类；两套标签可以数值完全相同。
不能用取值或与 `y_true` 的值相等判断输入语义，c=1时合法PU标签甚至等于真值。
只登记字段而没有运行时读点，也不能防止错配训练。

## 决策

复用 `core/validation.py` 的 `validate_label_semantics`，不建立第二套判据。
估计器声明其主 `fit` 输入为单值pu、pn或pnu；未声明第三方估计器保持PU默认。
ExperimentRunner继续从生成视图推导预期语义，PU公共工作流固定要求pu。
PUPipeline在显式类/实例解析、运行入口及实际clone/构造后检查。
比较、调优、CLI run和UI执行复用pipeline；CLI sensitivity的直接fit单独调用同一守卫。
普通PU推荐器和UI选项不推荐PN/PNU分类器，CLI仍展示其实际声明。

不改变PNU估计器直接接受三值标签的API，不把PU工作流扩展成PNU工作流。
Self-PU辅助验证集的真值语义不改变主输入PU声明，也不赋予clean主视图资格。
`guard_fit_labels`仍负责对象身份/共享内存防泄漏，两者不合并、不互相替代。

## 备选与后果

按标签数值猜测语义无法区分PU/PN，因此不采用；只靠命名、文档或只读元数据缺少强制读点，
也不采用。参数搜索保留候选失败隔离，但语义不符的候选在fit之前失败。
PU比较器对显式PNU候选现在在构造预检拒绝，不再先训练其他方法再记录PNU取值错误。
这是有意的提前失败行为变更，不改合法PU算法的目标函数或训练预算。

守卫不能证明调用者标签的真实含义，也不能识别有意误声明。用户必须正确声明输入，
不能修改既有方法的声明来绕过门禁。本决策不批准新Survey配方、不回填历史manifest，
不代替RV2独立复核或任何人工签署。
