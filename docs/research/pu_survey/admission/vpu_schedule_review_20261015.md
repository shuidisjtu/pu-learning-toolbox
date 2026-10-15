# VPU 源优化器调度与轮内步数复核（2026-10-15）

状态：独立工程复核 / pending_method_owner_review；不是正式候选或预算批准。
本次补齐[采样审计 §9](vpu_sampling_audit.md)记录的两项可实现来源差异，保留默认行为。
没有修改冻结协议、比较矩阵、旧结果、VPU 的 D21/D22 校准/选模裁决或人工签署。

## 来源与准确语义

独立检查本机干净的 `HC-Feynman/vpu` @ `603bcc1e628795f57a5ac87e5b0b977273b7cf91`，
HEAD 与原锁一致，工作区无改动；MIT 许可与 `vpu.py`、`run.py` 字节摘要见
[机器记录](../data/vpu_schedule_source_20261015.json)。仅阅读，未执行上游代码。

- [vpu.py:36–41](https://github.com/HC-Feynman/vpu/blob/603bcc1e628795f57a5ac87e5b0b977273b7cf91/vpu.py#L36)：
  条件是零基 `epoch % 20 == 19`，在该轮训练**之前**学习率减半并重新构造 Adam，betas=(0.5,0.99)。
  即 1-based 第20/40…轮使用新学习率；旧动量不保留。这不是普通 StepLR 的等价替换。
- [vpu.py:92](https://github.com/HC-Feynman/vpu/blob/603bcc1e628795f57a5ac87e5b0b977273b7cf91/vpu.py#L92)：
  每轮固定 `config.val_iterations` 次训练更新，不是调用验证集的次数。
- [run.py:8–14](https://github.com/HC-Feynman/vpu/blob/603bcc1e628795f57a5ac87e5b0b977273b7cf91/run.py#L8)：
  该数默认30；学习率3e-5、epochs50、batch500。

## 实现与默认兼容

新增两个 constructor 参数：

| 参数 | 默认 | 显式选择 |
|---|---|---|
| optimizer_schedule | constant | source_reset_halving：在上述轮次减半并重建 Adam |
| iterations_per_epoch | None（原数据量推导） | 正整数；30 对应作者的轮内训练更新数 |

其它默认参数不变。若要研究来源调度/步数切片，可显式配置：

```python
VPUClassifier(optimizer_schedule="source_reset_halving", iterations_per_epoch=30,
              learning_rate=3e-5, max_epochs=50, batch_size=500)
```

该示例不是获批 recipe，更不是完整作者重放：网络、loader 配对/抽样、数据规格和选择协议仍有差异。
没有改成作者的 val-var 选模；项目内该值依然只作 source diagnostic。
PA 资格、CNN/图像规格、公开数值、正式资源和配方依然需要独立审批。

history 新增 learning_rate/optimizer_generation，记录每轮实际率与 Adam 代数；
optimizer_resets_ 记录重建次数，optimizer_steps_ 累计实际更新数、不随重建清零。
callbacks/快照仍按 epoch 保存，不把30个内部更新误作30个权重快照。
新 history 字段是可观察的附加记录；不回填历史 manifest 或改写旧字段。
无效调度/预算须在训练前拒收；失败 refit 清除 fitted 标记，不能继续把旧模型当作成功重训。

## 台账、校准与证据

方法台账及原八方法非正式准入草稿的 VPU 快照同步来源能力与 review 引用，
admitted=false、candidate_pool/budget/protocol 仍为 null；不以台账同步替代批准。
TS 的边缘池 U∪P、OS 的消融、class_prior 不参与训练及 D22 验证诊断定位均不变。

独立 CPU 样例对照旧 `32762bc`：NumPy seed31、24×3、9P/15U，MLP8/depth1，
epochs3/batch5/random_state7；默认路径的 scores、max_log_phi 和全部既有 history 字段逐位相等。
这只证明该样例的兼容性，不外推完整数据/硬件或论文数字。
边界/Adam 新实例及空初始状态插桩、fixed/data-derived 步数、OS/TS、clone/pickle、
无效 refit 与确定性回归已补。新增 **16 项测试通过**；完整 CPU/Python 3.11 fast suite
为 **3765 passed / 59 skipped / 0 failures / 0 errors**（slow/e2e 未执行）。
格式、测试质量、文档链接/API、元数据、Skills、数学、baseline、注释、目录/层依赖及
materialized draft 门禁通过；原五方法 admission checker 保持 formal_admission=false，
其通过不冒充 VPU 来源批准。源码包与 wheel 离线构建成功，wheel 含新调度代码，未发布或上传。
不声称跨平台远程 CI、GPU/真实数据、论文数值或人工验收。
