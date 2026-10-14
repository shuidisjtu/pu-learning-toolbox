# P4.1-A 训练前配方绑定工程（2026-10-14）

本轮交付 runner、单次 CLI、pilot 预检/续跑、批次审计/汇总配方消费与**已存在 draft 的 schema CI**。
这不是 P4.1-B 配方批准，不生成正式 locked 注册表，不扩展旧 Pilot 的执行矩阵，
不改任何历史 manifest 或实验结果，也不代填 HENG958/shuidisjtu 的审阅。

## 已接通的链路

1. 显式指定注册表才进入绑定路径；不指定时，旧入口、目录与 manifest 字段保持原样。
2. 校验完整 registry/protocol 一致性、顶层及方法 profile 的 `locked` 生命周期、
   所请求数据集/训练路径的 scope、候选池、已保存 resolved snapshot 和可选预检摘要。
   draft、under_review、signed_off、retired 均不能用作运行锁。
3. 单次 CLI 在读取 split、建立编码器或输出目录之前完成此检查；pilot 在读取 split 元数据或启动批次之前核完整请求。
4. runner 在生成标签或 fit/retry 之前复核同一协议/配方；程序化模型的非协议字段必须保持当前构造器默认。
   class_prior、seed、device 和受原协议校验的模型/编码器属于单独的运行输入，不被当作调参覆盖。
5. 通过后冻结候选列表并克隆模型模板，避免后续策略修改外部 config/模型对象改变已绑定的训练候选。
   顶层 manifest、candidate_runs 与每项 PA/OA selection 写入相同的完整绑定身份。
6. CLI 的绑定结果置于原输出根下的 `recipe_<完整registry摘要>/`，不覆盖同请求的 legacy 路径。
7. pilot 把启动前解析的摘要传给每个子进程；若源注册表在批次间发生配方身份变化，子进程拒收而不是默默采用新配方。
   dry-run plan 按 planned run 记录绑定身份，覆盖范围不随 pending 缩小。
8. 绑定续跑要求与预检身份及协议版本/摘要逐字段相等，包括显式 `inner_search_id=null` 的存在性，
   同时检查候选/选择记录及单外层候选的 index=0。legacy、另一注册表、缺字段或错配记录不能满足绑定请求。

## 使用契约

模块级入口（不新增包根公共 API）：

```python
bind_recipe_candidate(
    registry, protocol, *, method, dataset, training_path,
    candidates=None, constructor_overrides=None, expected_sha256=None,
)
```

返回 `recipe_binding_status="registry_bound"` 及六个字段：`recipe_registry_version`、
`recipe_registry_sha256`、`recipe_profile_id`、`recipe_candidate_id`、
`resolved_params_sha256`、`inner_search_id`。不是对人工作出批准的证明。

`ExperimentRunner.config` 可显式传入已加载的 `recipe_registry` 对象，
并用 `expected_recipe_registry_sha256` 固定预检摘要；必须同时提供既有 `survey_protocol` 运行请求。
解析失败不会伪造 registry_bound 身份；版本化拒收按原失败 manifest 路径留痕。

获得真实批准和 locked 文件后才可使用以下命令；占位路径不是仓库现有已批准文件：

```bash
python scripts/run_survey_experiment.py <split-dir> \
  --protocol survey-v1.2 --dataset spambase --method upu \
  --class-prior <approved-population-prior> \
  --recipe-registry <approved-locked-registry.json>

python scripts/run_survey_pilot.py --dry-run --datasets spambase --methods upu \
  --recipe-registry <approved-locked-registry.json> --plan-json <new-plan.json>
```

`--recipe-registry-sha256` 是单次入口的可选 pin，主要由 pilot 自动传递。
没有注册表时不能单独提供 pin，也不能在 technical-smoke 路径偷偷忽略配方请求。

`completed_runs` / `pending_runs` 的新增可选 `expected_recipe_bindings` 是 run key → 完整身份映射。
pending 检查要求覆盖全部 planned keys；映射中的部分身份是调用错误，不降级为 legacy。
未传此参数时保留历史续跑语义，不能据此宣称做了注册表复核。

## v1 范围与 CI

当前冻结协议只有一个外层候选。现有 artifact 的 `recipe_candidate_id` 是标量，
无法忠实表示 PA/OA 各选不同外层候选的搜索结果；因此运行绑定 **明确拒绝多个外层候选**，
不把第一个候选冒充整池身份。未来需版本化 pool/selection binding 契约，再解除这条限制。
PUSB 的 72 对内部 CV 是一个外层候选的 `inner_search_id`，可以正常绑定。

resolved snapshot 仍只展开协议明示的参数/预算，不事后把全部默认值复制进旧注册表。
当前运行代码默认值由模型参数检查与 manifest 的 estimator_parameters 留痕；
这不替代正式阶段所需的代码、依赖与环境身份核验。
模板克隆会增加暂存内存；本轮合成 CPU 回归不代表正式 GPU/资源验收，P3.3 须测实际获批的绑定路径。

CI 仅检查已入库的 `survey_recipe_registry_v0_1_draft.json`，使用
`check_survey_recipe_registry.py --registry <committed-draft> --require-present`：
draft 丢失或 schema/协议不符则失败。其通过不意味着 locked。
**R5 的未物化 frozen 默认门禁仍不接 CI**，不添加一个检查不到文件的“绿色门禁”。
人工调用缺省门禁仍报告 Nothing was validated；`--require-present` 可显式拒绝缺失输入。

## 验证与尚未完成的部分

新增测试覆盖生命周期/范围/摘要/构造参数、runner 拒收先于标签生成、候选冻结、
CLI 真实合成运行及 legacy 字节不变、pilot 摘要转发/计划、续跑身份丢失与嵌套错配。
测试中的 locked 文件是临时 synthetic fixture，不是正式批准。

本轮 **不把 P4.1-A 整体标为完成**：批次白名单/审计/汇总配方消费已在下述扩展接入，
正式阶段仍须采用外部 v2 义务并完成接口复核；多外层候选的版本化工程契约随后见
[候选池身份 v2](p4_1_pool_binding_20261014.md)，正式冻结产物的 CI 激活仍未完成。
P4.1-B、P3.3 资源验收、RV8 方法准入和主榜制品的前置保持不变。

## 同日续项：外部批次义务与数值消费

新模块 `pu_toolbox/experiment/survey_recipe_audit.py` 接入
`audit_survey_batches.py`、`summarize_survey_results.py` 和底层 `aggregate_survey_runs.py`。
注册表身份和可比性不能只凭 manifest 自报，故新增白名单版本
`survey-batch-roots-2`，每个 batch 必须显式声明 `recipe_policy`。

绑定批次示意（仅接口示例，不是获批配置；将摘要替换为启动时固定的 canonical registry SHA256）：

```json
{
  "schema_version": "survey-batch-roots-2",
  "batches": [{
    "name": "new-approved-batch",
    "root": "/path/to/new-results",
    "expected_manifests": 5,
    "role": "formal",
    "recipe_policy": "registry_bound",
    "recipe_registry": "approved-locked-registry.json",
    "recipe_registry_sha256": "<64 lowercase hex characters>"
  }]
}
```

- 注册表路径相对配置文件解析；摘要是 canonical 内容摘要，**不是文件原始字节哈希**。
  不接受缺省推断的摘要，否则事后换一份 registry 会改变验收对象。
- `registry_bound` 重算 registry/protocol、locked 状态、实际 method/dataset/path scope 和 resolved 参数，
  逐字段核 root、唯一外层 candidate、每项 selection 与协议身份；显式 null 不能用缺字段代替。
  候选记录的实际 `params` 也必须存在并等于冻结候选池，不能仅靠一份正确的身份摘要掩盖矛盾参数。
  所有绑定字段被删除仍按外部义务拒收。draft 不能作为运行/验收批准。
- v2 的 `legacy_unbound` 不得携带 registry 路径/摘要或任何嵌套绑定声明。
  历史 B1–B4 的 v1 白名单及无绑定 manifest 无须迁移，输出字段/CSV 列保持原样，禁止回填旧制品。
  v1 结果若携带绑定声明则要求提供 v2 外部义务，不根据文件名或目录名猜测。
- 同份报告只消费一个 registry namespace；同一 registry 中不同方法的 profile/candidate 可以不同。
  不同 registry 或 bound/legacy 混合输入会明确拒收，需分开生成报告，不在旧 row/cost key 下池化。
- 审计报告新增独立、版本化 `recipe_audit` 扩展（`survey-recipe-batch-audit-1`），
  **不新增 A18，不改写 A01–A17 含义或历史报告**。
  配方拒收时 overall=fail，并在运行数值门槛前返回；不输出可排名行或成本合计。
  身份缺失/矛盾的失败标为 `not_reproducible` 并列出受影响路径；仅不同 namespace 的混用标为 `refused`，
  不把本来可复现的两份结果误记为字段缺失。summary coverage 同步不可复现计数。
  扩展通过仍须接受原有 17 项检查，不能把配方通过读成正式主榜通过。
- 成功 summary 的 JSON 行和 run 成本记录携带完整 `recipe_binding`，CSV 仅对扩展报告增加七个配方列；
  Markdown 显示 batch namespace、方法 profile/candidate/resolved/inner 身份及拒收证据。
- 底层 `aggregate(..., expected_recipe_bindings=...)` 只消费外部审计重算的完整方法身份。
  独立旧聚合 CLI 没有 v2 外部义务输入，遇到绑定结果会拒收并指向白名单审计/汇总入口；
  `--diagnostic` 不绕过该检查。历史 legacy 聚合行为不变。

新增消费回归覆盖完整身份、根/嵌套/全部删除、错误协议/registry pin、draft/缺失注册表、
跨方法正确绑定、跨批次同源接受/异源拒收、legacy 保留、CLI 失败产物与底层防绕过。
临时 synthetic locked fixture 仅验证工程契约，不代表项目签署或正式制品验收。

续项最终验证（Python 3.11、CPU）：新增 47 项测试；完整 fast suite
`pytest tests/ -m 'not slow and not e2e'` 为 **3639 passed / 59 skipped / 0 failures / 0 errors**。
格式、测试质量、文档链接/API 覆盖、项目元数据、MathJax、Skills 同步、baseline 配置、
注释质量、目录结构/层依赖及 materialized draft checker 均通过；源码包与 wheel 离线构建成功，
两种包内均确认包含新增消费模块。此处不声称跨平台 CI、slow/e2e、正式 GPU 或原始制品验收。
