# CVIR 校准与先验接线（2026-10-22）

本批补 P3.2 的独立工程入口，不批准正式算法准入，不改冻结协议、比较矩阵、
历史结果或人工签署。复杂任务经 topic branch → PR 交付；不直接推 main。

## 做了什么

- `CVIRClassifier.fit(..., os_or_ts="ts")`：训练原始 U 后接 P，形成全局风险池；
  warm-start、逐轮排序/截断与保留负例训练均使用该池，原始 P 正例项保留。
  默认 `os`、原生 `input_sampling="ts"` 的数值路径与预算不改；二者不能叠加校准。
- `CVIRPopulationPriorTrainer`：只接受 OS 来源、SCAR、显式 known-prior 来源，
  不覆盖调用方已经设置的 α_U/source。OS 按 `(π*n−n_L)/(n−n_L)`，校准视图按 π；
  只依赖训练 PU 计数，不读隐藏真值，不用请求 c 替代取整后的实际计数，不夹断非法先验。
  仍禁止将总体 π 直接传入 estimator 的 `fit(class_prior=...)`。
- 记录先验来源、公式/计数、实际风险池及校准范围。来源是调用方声明，不伪称独立验证。
  临时构造参数绑定成功/异常均恢复，实际训练 α 保留在 `unlabeled_positive_prior_` 和 manifest。
- 复用逐 epoch weights-only writer，PA/OA 外部独立选择，训练不收外部验证/test。
  容量预检使用同一 writer 目录和最大候选的 warm+CVIR 轮数，实测快照超估值则拒收并留证。
  可按 D25 回收未选中快照；warm-start 快照的记录沿用旧工程语义，不批准正式候选资格。
- 旧 `CVIRKnownPriorTrainer` 仍是显式 α_U 的 OS-only 接线，不能被新能力无声放宽。
  新类仍拒绝 versioned Survey 和 SAR；只有正式 v3 units/runner/预算完成后才能进主榜。

## 为什么这样处理（已验证工程选择，不重复请求 review）

| 来源／协议 | 本项目处理与原因 | 验证范围 |
|---|---|---|
| [论文 Algorithm 2，印刷页6](https://proceedings.neurips.cc/paper_files/paper/2021/file/47b4f1bfdf6d298682e610ad74b37dca-Paper.pdf)：先对整个 U 排序截断，再分批优化 | 按[协议 §2.3](../pu_survey_protocol.md#23-pu-数据集生成)先用训练 `U∪P` 替换风险池，再截断；不能截断后强行加入 P。理由是截断非线性且需保留全局排序语义。不是论文独立抽样 TS 数据 | 与显式角色展开的原生组件逐位对照损失、全部权重、负例身份、分数、阶段和更新次数；另测 P 能进入排序/被保留，不靠真标签强排除 |
| 协议 §3：规范总体 π 与视图对应 | OS 用实际 n/n_L 换算，TS 用 π；不是以标记频率或经验 U 真值重估先验 | 手算与质量恒等式、非整数标记数、非法/端点拒收、PA 仍用总体 π |
| 作者发布代码与本项目旧默认并不相同 | 保留论文 α 加权、旧 warm-start、Adam、取整/稳定平局和阶段预算，避免借校准任务改旧结果；历史差异见[来源记录](../admission/cvir_source_cnn_review_20261015.md) | 与本项修改前本项目代码逐位兼容对照；不声称重放作者代码 |

这里的“逐位对照”只针对同设备、确定性合成数据和短预算。显式角色展开的 P 复制是测试
代数构造，不是新采样的真实独立 TS 数据；不能由此推出论文精度、GPU 跨平台等价性或
数据生成假设成立。论文/源码既有差异没有新增待签署清单，本批也不代签既有方法学审批。

## 调用方式

以下为独立数组 API 的工程入口，不是正式 Survey CLI 配方。先提供四路隔离的
`DatasetPart`（train/pu_val/clean_val/test），按原协议设置 test 的 `for_selection=False`：

```python
from pu_toolbox.estimators.risk.cvir import CVIRClassifier
from pu_toolbox.experiment.cvir_adapter import CVIRPopulationPriorTrainer
from pu_toolbox.experiment.runner import ExperimentRunner
from pu_toolbox.experiment.strategies import ProtocolOA, ProtocolPA

runner = ExperimentRunner(
    class_prior=population_pi,  # 数据生成 metadata 的规范 π，不从 test 子集反推
    seed=0,
    protocols=[ProtocolPA(), ProtocolOA()],
    config={
        "c": 0.5,
        "os_or_ts": "ts",
        "trainer": CVIRPopulationPriorTrainer(
            population_prior_source="describe the generation metadata and its identity"
        ),
        "checkpoint_bytes_per_component": measured_component_upper_bound,
        "reclaim_unselected_checkpoints": True,
    },
    manifest_path="path/to/new-run/manifest.json",
)
result = runner.fit(CVIRClassifier(device="cpu"), train, pu_val, clean_val, test)
```

容量数值须按实际网络给出，不把小合成 MLP 的字节数外推为 CNN 上界。
此处不含正式数据、正式选模阶段裁决、frozen-lock GPU 或多 seed 效果结果。

## 验收与后续

专项 **217 通过、1 项 CUDA 跳过**，另补台账/准入草稿/生命周期门禁契约 **65 通过**。
最终完整快速回归 **5,474 通过、59 条件跳过、0 失败/错误**（5,533 项；
`not slow and not e2e`），本机 JUnit 为
`/tmp/cvir-calibration-full-fast-final-20261022.xml`。最终回归以
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` 限制 CPU 线程，
不是 frozen-lock/GPU 或远程多平台 CI 验收。

与本项修改前本项目 `456f1f4` 的 **27 组默认 OS 配置**逐位一致：3 seeds × warm0/1/2 ×
resident/CPU-stream/注入 CNN；核损失、保留行身份、全部权重、分数、阶段与更新次数。
校准与显式角色展开另覆盖 18 组配置；未执行作者代码，不把代数构造称为独立采样。

独立数组 runner 合成 CPU 验收同时跑 OS/校准 TS，每种两个候选、PA/OA 双轨，
各捕获 7 份快照并按选择回收到 1 份权重。分别保存两份选中预测模型，
均在新 Python 进程中回放分数/预测逐位一致；风险先验分别为 .25/.4，
无训练失败或重试。临时制品：`/tmp/pu-cvir-population-20261022-bnrehnuz`，
不是正式数据结果。规范源码 SHA256：
`0f421c7753977099b8c2ed9060cdeefc43c214c1e0340f344dbef7caccdace1a`。

格式、文档链接/API、元数据、数学、Skills、注释、测试质量、目录/层依赖、
baseline、recipe draft 和 candidate fit contracts 本机门禁通过。
`uv build --offline` 成功产出 sdist/wheel，包内容含新代码，不含用户上传物、原始数据
或 `.tmp` 探针。首次全仓发现审计引用/准入草稿快照不同步，均已补齐后完成上述绿色复测。
冻结 protocol/comparison JSON 与历史来源/结果/签署未变。

后续缺口是正式 v3 profiles/units 和适用 runner 接线、预算冻结、真实数据/GPU 验收，
不是继续重复审核本批已经做过对照的工程处理。
