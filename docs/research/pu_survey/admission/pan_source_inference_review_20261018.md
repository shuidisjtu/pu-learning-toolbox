# PAN 来源补证与逐层推理状态修复（2026-10-18）

承接 P3.1/P3.2，保留负责人在 #123 选定的论文目标与 OS 路径。
本项是代理工程补证，不是负责人签署、作者程序复现或正式准入。

## 具体交付

1. [机器复核包](../data/pan_source_review_20261018.json)将原来的单份 `main.py`
   摘要扩展为九文件：PU 采样/迭代器、CIFAR/MNIST loader、四套实际使用的 uPU
   配置及 README。锁定 `morning-dews/PAN@5be703639e4c1eb2c63d869ba150d84ebf0c116e`，
   清洁 checkout 的 commit 与九份原始字节摘要核对通过。
2. 补来源调用链：PAN 仍读取 `cfg/<dataset>/uPU`；epoch/batch 来自 iterator 配置，
   不是打印出的 CLI `n_epochs/batch_size`。记录 overlap 复制 P、实际角色比例、
   CIFAR/MNIST 映射与 /255、源码 error-rate/F1 日志口径及运行兼容性限制。
   `news/sddd` 只核配置，没有外推为文本/其余 loader 完整验收。
3. 保留四层区分：论文 Eq.(7)/Algorithm1、相关源码、工具箱、正式准入。源码的
   U_h/max-margin、eps、求和、共享旧图与论文目标不同；不把它们移入训练，也不声称
   期望与批量求和在有限 Adam 预算下完全等价。源码未执行，任何来源权重均未加载。
4. 新测试用独立解析式核 P/U 不等长均值和梯度；在**真实 Adam step 前**核查每个
   D/C 梯度、1:1 活动 batch、独立 U 重抽、更新后 D 的停止梯度奖励、非活动网络
   参数不变及全部更新成本。无先验进入目标，预测只使用 C。
5. 实际发现并修复 `_decision_function` 的状态副作用：旧代码只保存根层 `training`
   并递归 `.train(old_flag)`，会覆盖有意不同的 BN/dropout 子层模式。
   现在逐层保存、逐层恢复；正常返回、前向异常、取消均恢复原 flag。
   不扩展为任意自定义 `train()` 副作用的事务回滚承诺。

## 验证边界

修复前定向的 16 项混合模式测试 **全部失败**；修复后的测试覆盖 MLP/CNN、四个公开
预测入口、两种根层模式、第二批中断、空输入、非法形状和 pickle。
同时核查权重/BN buffers、D 的状态和 torch RNG 不被推理改变。

相对 `c62ada9`，`fit` AST 完全不变；用 seeds0/7 × MLP/CNN 共四组缩短合成训练，
C/D 全部权重与 buffers、历史、步数及默认推理得分逐 bit 相同，每组8步。
仅执行可信的本项目旧提交作差分，未执行作者程序。此为成功默认路径兼容性，
不抹平故意修复的混合模式副作用，更不是论文数值验收。

台账、[五方法扩展草稿](../data/p3_preintegration_extension_v1_draft.json)与来源/校准
引用同步；当前实现文本 SHA 绑定到机器复核包。旧探测的源码 SHA 仍对应旧提交，
不改写历史、不复用为当前源码资源重测。冻结协议/矩阵、旧结果、准入位与签名不变。
源代码许可仍为 unknown 观察，registry 的 needs_review 不自动放行。

复现（仓库根，CPU）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/unit/estimators/test_pan_source_contract.py \
  tests/unit/estimators/test_pan_prediction_modes.py \
  tests/unit/estimators/test_pan.py tests/unit/estimators/test_pan_cnn.py
python scripts/prepare_p3_preintegration_extension.py
```

本地已有锁定 checkout 时，可只读重验，不导入或执行作者代码：

```python
import json
from pathlib import Path
from scripts.check_p3_admission_evidence import validate_source_checkout

review = json.loads(Path(
    "docs/research/pu_survey/data/pan_source_review_20261018.json"
).read_text(encoding="utf-8"))
assert validate_source_checkout(review["source"], Path("/tmp/pu-pan-source")) == 9
```

## 本轮验收记录

- 扩展专项 **89 passed / 2 skipped / 0 failures / 0 errors**，43.346 秒；
  两项 CUDA 测试因当前 CPU torch 环境跳过。新来源/推理测试共47项。
- 四个仅内存的故障注入均被抓住：D 符号反转、C 梯度缩放、C 复用旧 U、递归恢复
  覆盖子层模式；生产源码不做临时修改，不执行任何作者程序。
- 12 项本地静态门禁通过。五方法准入证据和阶段预算草稿校验继续通过，
  新 PAN 来源包九文件可选检查及扩展草稿再生成一致；禁止 torch 导入时也可核验。
- sdist → wheel 构建通过；源码目录外从 wheel 核对当前 PAN SHA，CNN 拟合、
  混合模式恢复与 pickle 回放均通过，4个实际更新步骤与 OS 元数据一致。
  wheel/sdist 未混入私人数组、上传压缩包/Excel、模型或作者代码。
- 全量快层 `pytest tests/ -m 'not slow and not e2e'`：
  **4456 passed / 58 skipped / 0 failures / 0 errors**，309.599 秒；
  比同步后的 `c62ada9` 增加47项通过。JUnit 在本机 `/tmp/pan-source-fast.xml`。
  既有 solver/代价等 warning 保留，不声称全仓零 warning 或远程多平台 CI 已通过。

未启动正式训练或 GPU；P3MIX 全文来源、Split-PU #122、正式预算/资源及负责人准入
继续保持各自前置，不据本项解除。
