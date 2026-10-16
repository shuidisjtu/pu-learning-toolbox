# P4.1-A：候选池身份契约 v2（2026-10-14）

本次扩展的是 **artifact 身份表达**，不是实验协议、注册表参数语义或正式配方批准。
未修改冻结协议、比较矩阵、645-run 结果、draft 注册表或人工签署。
先前单候选绑定与批次消费见[同日工程记录](p4_1_runtime_binding_20261014.md)。

## 为什么需要单独版本

旧标量绑定的根字段 `recipe_candidate_id` 只能忠实表达一个外层候选。
多候选运行中 PA/OA 可选择不同候选，且候选失败后，成功 trajectory 的位置会压缩。
例如候选 0 被排除，候选 1/2 对应 trajectory 0/1；把 run_index 当 candidate_index 会读错权重与配方。

新增模块 `pu_toolbox/experiment/survey_recipe_binding.py` 定义
`survey-recipe-binding-2`，三层身份各有用途：

| 层 | 记录 | 约束 |
|---|---|---|
| 运行根 | registry 版本/摘要、profile、完整有序 `recipe_candidates` 及其摘要 | 不写一个虚构的全局已选 candidate，也不写标量 resolved/inner 身份 |
| candidate_runs | candidate_index、该候选完整身份、status、trajectory_index | 完整覆盖池；排除项保留身份且 trajectory_index=null；成功 trajectory 一对一、从 0 连续编号 |
| PA/OA selection | 实际 candidate_index、该候选完整身份、run_index | 每项可选择不同候选；run_index 必须对应该候选的实际 trajectory；不可选择 excluded 项 |

池中每个身份包含原有七个绑定字段、显式 v2 schema 与整数 candidate_index。
`recipe_candidates_sha256` 是有序身份列表的严格 JSON 摘要；每个 `resolved_params_sha256`
由对应 candidate 的完整 snapshot 重算，不能用池摘要代替逐候选身份。
候选列表位置、bool 冒充整数、重复 ID/index、缺失 null、错误 hash、混合标量/池字段均拒收。
实际候选记录可按不同顺序传输，但必须保留正确 index；其 params 仍逐 index 对照外部协议。

## 显式接入，不自动升级

模块级 `bind_recipe(..., binding_schema_version=None)` 默认调用既有标量 binder，输出不变。
显式传 `binding_schema_version="survey-recipe-binding-2"` 才生成池身份；
也可用 `bind_recipe_pool(registry, protocol, **request)`，请求参数与单候选 binder 相同。
`candidate_binding_at` 为候选和其实际 selection 返回独立身份副本。

- runner 配置：`recipe_registry` 对象 + `recipe_binding_schema_version`，可同时 pin registry 摘要。
- 单次 CLI / pilot：`--recipe-registry <approved-locked.json>` 与
  `--recipe-binding-schema-version survey-recipe-binding-2`。
- CLI 输出使用 `recipe_pool_<registry canonical SHA256>/`；旧标量继续 `recipe_<SHA256>/`，
  legacy 路径也不变。单候选可显式使用新格式，但不会覆盖同一 registry 的标量结果。
- pilot 在数据读取/启动前检查完整请求，在 dry-run plan 固定池身份及协议摘要，向每个子进程传递
  schema 与 registry pin；续跑不把 scalar/legacy、错误池或错误 selection join 当作完成。
- 外部批次白名单仍用 `survey-batch-roots-2`。绑定 batch 额外显式声明
  `recipe_binding_schema_version: "survey-recipe-binding-2"` 才要求池格式；省略时仍要求旧标量格式。
  显式 null、未知版本或 legacy batch 声明 schema 均拒收，不推断版本。
- 单份报告的 namespace 现在同时包含 registry 摘要和绑定 schema；不同格式也必须分开报告。
  `recipe_audit` 保留独立版本，不改 A01–A17。底层聚合器使用同一 schema-aware matcher。
- v2 summary JSON 每行保留完整池身份，并新增 `recipe_selection_bindings`，逐 seed/manifest
  记录该行 PA 或 OA 实际选中的候选身份与 run_index。run 成本仍按运行计，不复制到 PA/OA。
- 扩展 CSV 增加 schema、池摘要及 `recipe_selection_bindings_json`。旧标量候选列在池根无值时留空，
  不把某个 seed/准则的选择冒充整行或整次运行的统一候选。Markdown 列出池与各候选身份。
- 旧 `validate_manifest_binding` 是标量格式校验器，遇到池/混合格式返回 finding，不能将其读成 legacy。
  池制品须由外部重算的 v2 expected binding 及共享 matcher 校验。

## 未放宽的边界

1. registry/profile 必须已 locked，scope、来源、snapshot、协议、候选数量及预算仍逐项验证。
2. **注册表 schema v1 仍只允许 current_protocol_binding，禁止参数级 overrides**。
   池 artifact v2 不解除此禁令，也不允许把协议中的非空参数候选冒充已解析配方。
   真正新增超参搜索仍需相应协议/注册表语义版本、来源与预算批准，不能仅加本 flag。
3. 冻结 Pilot 仍只有一个外层候选。多候选回归仅在临时 synthetic protocol/registry 下进行，
   这些候选均为现有默认参数绑定，不是实际获批搜索空间；没有生成、上传或批准正式 locked 文件。
4. PUSB 的 72 个内部 CV 参数对仍归属于一个外层候选，不展开为 72 个 recipe candidates。
5. 多候选回归中受控选择/失败只验证身份与恢复接线，不证明哪个候选数值更优。
   正式 GPU 资源、扩大的候选池预算、真实运行制品与双方审阅仍须独立验收。
6. 本交付不把 P4.1-A 整体完成或 P4.1-B/RV8 放行。正式冻结产物/CI 激活、接口复核与正式运行前置保留。

## 验证范围

新增回归覆盖 registry/profile lock、pin、参数覆盖拒绝、池摘要和严格字段、不同 PA/OA 选择、
首候选排除与后续重试、压缩 trajectory 对账、跨 schema 续跑拒绝、单次 CLI 的旧字节保留、
pilot plan/子进程 schema 固定、外部义务、JSON/CSV 逐 seed 身份和畸形制品拒收。
真实合成 UPU 单候选 CLI → 审计 → 汇总链路不使用替代协议；多候选链路使用明确的临时 synthetic 输入。
最终 CPU/Python 3.11 验证：新增 targeted **58 passed**；全量 fast suite
`pytest tests/ -m 'not slow and not e2e'` 为 **3697 passed / 59 skipped / 0 failures / 0 errors**。
格式、测试质量、文档链接/API 覆盖、项目元数据、Skills 同步、MathJax、baseline 配置、注释质量、
目录结构/层依赖及 materialized draft checker 均通过；源码包/wheel 离线构建成功，wheel 包含新增池模块。
不据此声明跨平台 CI、slow/e2e、正式 GPU、实际扩大参数搜索或正式制品验收。
