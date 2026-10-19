# CNN feature adapter 的推理模式恢复（2026-10-19）

## 范围

修复 `adapt_image_bundle_to_features` 的状态保护。此前只保存 encoder 根模块的
`training` 标志，异常后用递归 `train(was_training)` 恢复，可能覆盖调用方设置的
BatchNorm/Dropout 混合模式；而 `eval()` 位于 `try` 之前，eval 切换中途报错时无恢复保护。

现在在切换 eval 前保存所有模块各自的 `training` 状态，把 eval 和四角色特征提取都放入
`try`，并在 `finally` 逐模块直接恢复原标志。前向异常和 eval 转换异常原样传播，不重试。
不改变 state_dict/hash、特征值、输入角色、encoder 是否训练或 cache 语义。

## 回归与边界

`tests/unit/experiment/test_feature_adapter.py` 新增：

- 成功提取后恢复根模块、BN、Dropout 与 head 的混合 train/eval 状态；
- encoder 前向异常后恢复所有模块状态；
- eval 转换中子模块抛异常后恢复所有模块状态。

验收命令：

```bash
PYTHONPATH=. pytest -q tests/unit/experiment/test_feature_adapter.py \
  tests/unit/experiment/test_p3_feature_adapter_smoke.py \
  tests/unit/experiment/test_oracle_cnn.py
```

2026-10-19 本地结果：39 项中 34 通过、5 项 CUDA 测试跳过（环境无 CUDA），零失败/错误；
Ruff 与 `git diff --check` 通过。该测试不包含正式数据或 frozen-lock GPU 资源验收，
也不改变 P3.2 批准结构、实验配方、P3.3 资源前置或方法准入。
