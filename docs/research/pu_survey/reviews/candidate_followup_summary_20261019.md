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
最终包在本机 `/tmp/pu-candidate-release-rZtu4g/final`，构建后仅补写本验收文档。
以上为本地验证，远程Actions结果须以推送后实际运行结论为准。
