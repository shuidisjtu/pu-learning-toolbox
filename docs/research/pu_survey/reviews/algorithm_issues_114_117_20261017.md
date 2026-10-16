# 新增算法复核整改（2026-10-17）

基线 `92efb4d`，承接 GitHub #114–#117。工程整改不改变旧冻结协议、矩阵、
645-run、历史签署或正式准入；各项明确区分已验证行为和保留限制。

## #114：PUET TS bootstrap 风险归一化

TS 的 U 风险池包含原 U 与 P 的独立角色副本。原代码按该池大小取每行权重，
但 bootstrap 仅抽原 U 数量，导致根节点 U 总质量不足。现在从实际风险池
有放回抽取该池大小的行，P 仍独立抽 n_P 行，故每棵树 W_U=1、W_P=prior。
OS 抽样数量/RNG 调用和非 bootstrap 分支不变，默认 bootstrap=false 不受影响。

新增 2 views × 2 bootstrap × 3 P/U 数量 × 3 seeds 的36个组合，逐树检查
实际角色数量、风险质量、根风险0.96及恒定叶负票。修复前9个 TS-bootstrap
组合失败，修复后通过；不是只比较相同seed的重复结果。

## #115：LaGAM 发布采样定位

锁定作者 commit `362a7f41fcf3d4e4161fe99f4aab4e6128b6a5d8` 的
`utils/utils_data.py:182-193` 明确从 U 移除验证和已标记 P，因此将当前类与
registry 派生 scenario 统一为 SINGLE_TRAINING_SET、台账与准入草案快照为 os。
旧来源收据的初始 TS/OS 冲突保留，另补 resolution；不代签负责人。

默认 OS、calibration_applied=false，显式 TS 拒绝。这个判断只描述作者发布
实验，不断言算法理论只能 OS；独立 clean support、额外标签预算与 PA 限制不变。
