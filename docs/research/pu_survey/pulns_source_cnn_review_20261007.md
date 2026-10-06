# PULNS：来源复核与原生 CNN 工程路径

状态：**implemented engineering path / pending_method_owner_review**。
本项承接[后续优先级](survey_execution_plan.md)的缺失算法训练路径，未修改冻结协议、
比较矩阵、历史结果或本人签署。能力增加不意味着正式准入或 PU-only PA 资格。

## 1. 原始来源与不能推断的部分

主来源：[AAAI 2021 正式 PDF](https://cdn.aaai.org/ojs/17064/17064-13-20558-1-2-20210518.pdf)，
DOI 10.1609/aaai.v35i10.17064；[作者机构页面](https://www.microsoft.com/en-us/research/publication/pulns-positive-unlabeled-learning-with-effective-negative-sample-selector/)。
2026-10-07 复核两处来源及 PULNS+github/source-code 关键词后，仍未确认作者代码仓库。
这是检索状态，不是证明源码不存在；不将 PULSNAR 或第三方 survey 仓库登记为作者实现。

印刷页8786 的 Classifier/State 要求分类器提供特征表示，最后隐层可作状态；
页8789 的 Classifier Specification 明确 CIFAR-10 用 CNN，其余用100隐层 MLP。
页8787 的 Reward 需要含真正正/负标签的小验证集；它参与训练奖励，不仅外部选模。
页8788–8789 按给定 U 正例比例 γ 抽样，不等于本项目标签频率 c。
Algorithm 1 的更新后重新选负例及最佳分类器分离保留。
公开文字未确定完整作者 CNN 层/增强/初始化、probe 复制和优化器恢复细节，不能由通用
CNN13/ResNet18 接线推导出源配方数值复现。

## 2. 实现边界与执行行为

`PULNSClassifier(encoder=None, ...)` 默认 MLP不变；二维可注入编码器，4-D NCHW 必须显式注入，
不将图像 flatten 成旧 MLP。编码器必须输出有限二维特征并保留 batch；support 的完整样本形状
必须匹配训练输入，仍须同时含两类真值标签。

每次 fit deepcopy/解冻编码器；CNN→hidden_dim ReLU→一个 logit，最后隐层作状态。
预训练、reward probe、更新后的实际分类器都通过 BCE 端到端更新编码器；probe 是独立网络，
不共享参数/BN，不以 probe 的最高分替换最终最佳实际分类器。
每 episode 从当前实际分类器 eval 提取状态，特征/奖励 detach；selector 的 REINFORCE
不反传 CNN。更新后仍重新按政策选 N，空 N 不强制捏造负例或进行 P-only更新。
singleton BN 用运行统计而不丢尾部行，预测 eval 后恢复各层原模式（包括失败路径）。
可信 pickle 可恢复预测，未新增 epoch callback、episode checkpoint 或续训保证。

CNN 原图与 support 数组留 CPU，仅批次上设备；hidden 状态分批提取、在 CPU 缓存。
逐 U 的 selector 状态才上设备。默认 MLP 保留原全数组状态 forward，避免数值路径静默变化。
CNN CPU缓存仍需 n_train×hidden_dim；policy 激活图仍随 U增长，另有实际网络/probe/最佳权重、
优化器和奖励 forwards，不宣称全 episode 常数显存/内存，旧二维探针不外推新图像配额。

## 3. 预算、校准与隔离不因 CNN 改变

`stage_optimizer_steps_` 分 pretrain/reward_probe/policy/classifier；`optimizer_steps_`
与 `history_["optimizer_steps"]` 累计实际执行更新。返回较早最佳模型不减已付成本；
selector 更新计一次，classifier/probe 各计实际 minibatch；空 N 的跳过计零。
支持集评估前向及状态提取前向没有反向更新，但也不能从正式资源预算中忽略。

`requires_clean_support=true`、`pa_eligible=false`、`run_view=os-compatible`、
`calibration_applied=false` / `calibration_hooked=false` 保留，显式 TS fail-loud。
源文的 calibrated negative set 不等于本项目 TS-OS 风险校准。
support 不是 classifier 训练行/状态质心数据，但真值仍通过奖励和内部 best 选择影响训练。
双方 IDs 只验证 train/support 互斥；selection/test/prior/support 全角色互斥仍待新协议。
没有 IDs 时保持 unverified_caller_responsibility，不从数组不同推断来源隔离。

公共 Pipeline/CLI/Survey 暂无独立 reward-support 路由；CNN能力登记不批准其标准 fit_evaluate/run。
须直接调用 estimator 并显式提供支持集；测试确认外部 y_true 不会被隐式转为 support。
自动 PU-only 推荐继续排除它，旧 Survey不能直接跑批。
UI catalog 仍披露 CNN能力和 requires_clean_support，但 ui_ready=false；手动/比较选项及
图像“当前支持”提示不列此方法。配置导入在修改 widget 状态前拒绝需要 support 的方法
（同时适用于 LaGAM），不会因换图像路径就偷偷借用真值。能力集合与可运行集合明确分开。

## 4. 同步和工程证据

方法台账、[方法卡](../method_cards/PULNS.md)、API、能力契约与五方法扩展草稿同步；
新增 CNN状态/顺序 policy 资源 blocker，旧阻断保留。正式候选/预算/协议指针仍 null，
admitted=false、负责人决定为空；来源未确认、标签预算、校准口径和资源验收不自动接受。

单元测试核 CNN三个训练阶段确有参数更新、模板/probe互不共享、训练行不含support、
hidden与reward行序、分批设备形态、尾部BN、最佳实际权重恢复、全成本与 seed/refit/pickle。
集成测试通过公开 build_encoder 与 estimator 接口验证 CNN13/ResNet18 的独立副本训练，
并拒绝公共流程没有 reward support 的调用；并非正式协议或作者 CNN实验。
默认 MLP与上一提交 `bc992e2` 的可信代码在 seeds 0/1/7 下比较：分数、所选负例、
既有 history字段和实际更新数逐 bit一致；新分阶段计数之和等于全部实际成本。
本轮CPU测试使用 Python 3.11.12 / torch 2.13.0+cpu，无GPU验收记录。
最终相关组合为 **80 passed / 4 skipped**（6.46s）；跳过包含2项CUDA案例与
2个依赖Streamlit的集成测试文件，不宣称界面浏览器实跑已验收。
最终快层 `pytest tests/ -m 'not slow and not e2e'` 为
**3172 passed / 58 skipped / 36 deselected**（161 warnings，211.12s）；
新增 CNN单元12项CPU通过/1项CUDA跳过，内置backbone及工作流边界集成3项通过。
11项静态门禁通过；既有P3准入证据、公开对照及五方法选模预算检查通过，
formal_admission=false，本轮这些检查未重下载上游来源文件。
冻结协议、比较矩阵、既有 P1-1/P1-2 制品、uv.lock 和私有 Excel 未改。

下一步继续 GEN-PU 的双生成器/三判别器图像架构来源、PULNS新路径资源与独立支持集交接；
不以当前训练接线清空原始制品/负责人协议阻断。
