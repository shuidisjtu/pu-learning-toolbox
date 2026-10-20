# 候选工程续批与发布验收（2026-10-19）

用户安排跳过P3MIX并推进独立任务，后明确要求本轮整理推送。
起点main为 `c986e05`，保留原有用户数据/探测文件改动，不纳入本轮提交。
远程issue复查没有新任务；#69由合作者明确标为低优先且不影响Survey，本轮未改。

## 已完成的独立交付

1. Split-PU、GEN-PU/Holistic-PU/PULNS、LaGAM、TEDn、VPU/神经LBE的评分
   状态保护。正常/前向异常/eval转换异常恢复逐层模式，空输入不改状态；
   共享模式契约覆盖11方法，另测resident与神经LBE的propensity路径。
   内部评分也用于Split阶段、TEDn留出/排序及LBE E-step，已补相应时序/数值回归。
2. GEN-PU、Holistic-PU、LaGAM、PULNS、TEDn及VPU、神经LBE、PAN、CVIR、RP
   共10候选新增失败拟合guard，连同既有五方法组成15候选保护范围。
   参数绑定前和异常/取消后撤销公开拟合状态；不回滚权重/历史/外部checkpoint。
   成功路径、签名、clone、回调可见性、MLP/CNN及重训/pickle另行验证。
3. 新只读 `check_candidate_fit_contracts.py` 及15方法工程草稿，绑定源码、
   guard/检查器/审阅文档摘要、fit参数和台账校准口径，接入静态CI。
   专项27项通过；这是工程身份门禁，不是配方冻结或方法准入。
4. 全仓验收捕获PAN/RP实时来源SHA滞后：保留旧SHA历史，刷新实时字段和
   工程草稿引用。136项来源/机器门禁回归通过，不删断言、不重写上游来源摘要。
5. 临时LBE比较/诊断脚本溯源修复：Git始终从脚本检出根查询，不取调用方cwd
   或父仓库的提交；缺失Git记unknown而非clean。实际加载实现的路径/SHA与
   脚本Git分开登记，明确wheel不是脚本checkout。诊断锁文件也改为脚本根路径。
   共42项溯源及原比较/诊断专项通过，零跳过/失败/错误，3.438秒。

各批细节以同目录生命周期记录及 `../admission/` 的模式恢复记录为准。
旧验收计数保留为原批次记录，不回填成当前全量结果。

## 发布包与真实输入的工程回归

离线sdist→wheel构建通过；15候选模块、共享guard与台账共17文件与源码逐字节
一致。包未夹带用户根目录压缩包/Excel/私有索引、数据数组或checkpoint。
wheel解包至 `/tmp/pu-candidate-release-rZtu4g/installed`，从仓库外确认真实导入该副本。

初次从 `/tmp` 运行LBE临时脚本因错误cwd Git查询失败，只产生计划/失败记录、
未进入训练；未删除该失败记录。修复后使用已有Spambase split_0，
c=0.1/0.3/0.5、单线程CPU、terminal、`--smoke`，分别在仓库内/外运行同一wheel。
共12次短拟合、6对跨cwd运行；选择、标记身份和test结果一致，
24份本机自生成PA/OA选中模型的摘要/预测及阈值回放通过，原split文件身份未变。
仓库外6次拟合无重试、捕获warning为0；不外推全仓或正式预算零warning。

原始包、环境、计划、模型与日志仅在本机 `/tmp/pu-candidate-release-rZtu4g`，
不提交数据/权重；临时目录不保证长期保存。这是执行链验证，不是效果或论文数值证据。
短预算不能得出神经LBE优于线性LBE，不按test改变超参、方向、阈值或选择范围。
新环境说明中script checkout可dirty，wheel实现另有SHA，不伪称frozen-lock环境。

## 保留前置与后续顺序

- P3MIX等待用户补充全文，下轮再做来源/完整训练流程，不注册数学组件冒充完整方法。
- #119/#121/#122的共享BN/增强/初始化/学习率/阶段与JSD/prior/混合比例仍待负责人决定。
- 独立clean support真实来源、组身份及标签预算不因合成测试自动批准。
- 批准结构资源profile、正式registry/预算冻结、PN-CNN及多seed/GPU验收须满足原前置。
- 不主动重开已暂停的P2.2 review，不改旧645-run、冻结矩阵、上游许可证结论或签署。

## 最终验收

冻结本轮代码后的 `python -m pytest tests/ -m 'not slow and not e2e'`：
**4868 passed / 58 skipped / 0 failures / 0 errors**，共4926项，333.208秒；
本机JUnit `/tmp/candidate-wrapup-final-fast.xml`。跳过及未执行slow/e2e不计作通过，
既有代价/消融/solver警告仍存在，不宣称全仓零warning或CUDA验证。
16项静态/证据门禁通过（含本轮新增候选生命周期门禁）。

最终sdist→wheel再次离线构建通过，17个实现/guard/台账文件在两包中与源码一致，
sdist包含新溯源辅助脚本；最终wheel与上述仓库外回归实际导入包SHA一致。
包版本仍为1.11.0，不代表本轮已上传PyPI或发布release。

## 后续状态更正（2026-10-19）

本报告中“等待用户补充全文”的 P3MIX bullet 只代表形成报告时的状态。用户随后提供 14 页
PDF；P3Mix/P3Mix-E/P3Mix-C 原文核对、classifier、registry/ledger 与专项测试现已完成，
详见[原文交叉表](../admission/p3mix_paper_review_20261019.md)及[当前执行计划](../survey_execution_plan.md)。
RV8、正式配方/预算和资源验收仍待完成；该更正不改变正式准入状态。
最终包在本机 `/tmp/pu-candidate-release-rZtu4g/final`，构建后仅补写本验收文档。
以上为本地验证，远程Actions结果须以推送后实际运行结论为准。

## 2026-10-20 收尾与暂停

用户要求立即整理当前部分并提交推送，随后等待新任务，不再自动扩展算法工作。
本轮未改旧冻结协议、比较矩阵、645-run 结果、负责人签署或正式准入。

- `8c9c208`（已推送）：Dist-PU 增加显式 staged/histogram encoder 路径，支持 CPU-stream、
  CV 副本隔离、单行尾批及模型回放；默认 MLP 和冻结 Pilot 的 feature-adapter 不变。
  共享 checkpoint 评分输入隔离同时补齐，历史资源测量保留原源码身份。
- `d3f3ea7`（已推送）：再次核对 P3MIX PDF，修复 in-place 网络污染 raw/mixup/teacher/
  评分输入，补三变体真实卷积 CV、pickle、四角色 PA/OA checkpoint 与回收回归；
  台账和源码摘要同步。该截点完整快测 5,274 通过、59 跳过、0 失败/错误。
- 当前 CVIR 收尾：resident/CPU-stream 评分逐层模式恢复，覆盖部分 eval/前向失败和空
  输入；隔离 probe 与前向输入；提前校验学习率、selector α_U、回调及复数输入。
  [来源/技术记录](../admission/cvir_source_cnn_review_20261015.md)、方法卡、API、台账与
  当前候选源码摘要同步。36 组合法配置与修复前实现逐位一致。

CVIR 首轮全仓回归暴露旧 CNN 测试依赖全局随机初始化：某些模板可使 encoder 梯度为零，
权重变化断言受前序测试影响。现固定测试模板自身的 seed 并隔离 RNG；没有放宽权重更新
断言，也没有为测试更改算法目标。该失败记录保留，最终回归结果另行追加。

未完成事项继续保留：CVIR 的 TS-compatible 校准、各方法 RV8 复核、完整正式配方/
标签预算与数据来源、批准结构的资源 profile、多 seed/GPU 对照和正式准入。
本轮不启动新实验，不处理用户新增的协议草稿、PDF、Excel、压缩包或原始数据；
以上用户制品不随代码提交。

最终收尾完整快测 `not slow and not e2e`：**5,302 通过、59 条件跳过、0 失败/错误**，
共 5,361 项（本机 JUnit：`/tmp/cvir-final-fast-fixed-20261020.xml`）。静态质量门禁、
候选源码绑定与台账一致性通过；36 组兼容性对照为零偏差。未执行正式数据/GPU 跑批，
不把本地通过当作远程多平台 CI 已通过。本次推送后停止持续推进，等待用户新任务。

## 2026-10-20 复工：LaGAM 全文证据补齐

用户随后要求继续未完成工作，本批从 `892ecc4` 推进 P3.1 独立来源核查。
重新 fetch 确认 `main` 与 `origin/main` 无差异；gh 未登录，匿名 issue API 返回
403/rate-limit，公开网页为旧缓存，故不宣称已核对最新 issue 或无新任务。

已读取 LaGAM 正文 10 页/补充材料 7 页并绑定摘要，重读固定作者源码且摘要与旧收据一致。
[新交叉表](../admission/lagam_paper_review_20261020.md)与机器记录保留10-15的获取失败历史，
明确三项新来源差异：正文/补充的零梯度投影边界、β=0 与发布损失系数、100簇与README
CIFAR-10的5簇。普通均值/平衡BCE、当前hard-observed元步/原文软标签、完整LGA/
当前同簇项及support按原始类别取样的差异均补齐；未静默改变算法或解释为数值复现。
方法卡、台账、执行计划和文档索引同步，新收据保持 pending 且无代签。

新增15个展开用例通过，专项84通过/1条件跳过；完整快测 **5,317通过、59条件跳过、
0失败/错误**（5,376项、328.029秒）。本机XML为
`/tmp/lagam-paper-followup-final-20261020.xml`与`/tmp/lagam-paper-final-fast-20261020.xml`。
静态质量、候选fit身份、baseline与已物化draft registry门禁通过，不等于正式registry冻结
或远程CI/GPU验收。未执行slow/e2e，既有警告和审批/数据/预算前置不消失。

本批无算法默认值更改，不改冻结协议/比较矩阵、旧645-run结果、签署或正式准入。
用户原始PDF/Excel/压缩包/索引及根目录待审协议草案均未纳入提交。
剩余优先级：方法负责人裁决上述来源/配方差异及RV8；其余可独立的算法来源/行为核查；
正式CNN/共享特征规格、support预算、CVIR TS-compatible校准方案获批后，再做对应实现和
资源/多seed验收。根目录草案未被本轮视为已批准协议。
