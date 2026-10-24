# profile 与算法身份分离（2026-10-24）

本批落实 [v3 接入请求](survey_v3_proposal_20261023.md)的第一个工程前置：
一个注册算法可以对应多个独立 profile，而无需把配方变体注册成新算法。
**不发布正式 v3，不变更冻结 v1/v2、历史结果、方法台账或人工签署。**

## 已接通的链路

显式路径协议必须同时声明 `protocol_version: survey-v3.0-draft` 和
`profile_identity_schema_version: survey-profile-identity-1`；每个 profile 显式填写
`algorithm`。`--method`、execution unit、输出目录和批次键保留 profile ID，
模型构造、原生类校验、先验要求、OS/TS 台账及快照组件数使用 canonical algorithm ID。
没有新增全局 registry alias，也没有安装一个默认可运行的 v3 协议文件。

新 manifest 增加 `profile_id`、`algorithm_id` 和身份 schema。续跑须匹配这三项及
完整来源协议 SHA；`--datasets` 分片不把筛选后字典摘要冒充完整来源摘要。
缺字段、身份错误、协议漂移、布尔 seed 或未知 schema 都不算已完成。
两个 profile 即使共享同一原生算法，也分别保存运行身份，不相互覆盖。

旧协议仍以 method 为算法名，不新增这些 manifest 字段；原有部分规划字典 API 保持。
新字段不能偷偷塞进无 schema 的旧协议；profile 名不能重定向既有算法/别名或 oracle，
不接受未知算法、先验估计器冒充分类器、不安全路径或 oracle 别名。

## 仍然拒绝的路径

- 新扩展强制追加 `profile_identity_extension_engineering_only`，即使输入草案把
  review status 改成 accepted 或清空 blockers，也不会生成正式可比结果。
- 新身份记录 `comparison_registration_status=missing_versioned_profile_mapping`；
  不借用冻结矩阵里同算法的旧映射。旧 registry v1 的 v2 refs 明确拒绝绑定新身份。
- 原 `self_pu` 保留 no-meta 语义。`self_pu_full_meta` 必须对应同一个 `self_pu` 类，
  显式 `require_validation=true`、`paper_population` 和独立 clean support 声明。
  类可以构造，但正式五角色适配未接通前，通用四角色 CLI 在读数据前拒绝它。
- PULNS/LaGAM 同样不能借 alias 绕开 support 契约；CVIR 仍须版本化的逐视图先验/
  风险池适配，不能直接走通用四角色路径。此前独立工程入口不受此限制替代。

## 验证口径

专项覆盖显式映射/拒绝、原生类与旧初始化逐位一致、真实 UPU 双轨执行、两个 profile
独立输出、原生 nnPU 两轮训练与 CLI 快照逐张量一致、先验/视图/组件成本、
实际 pilot 子进程与再次续跑、分片来源摘要及正式准入阻断。
数据为临时合成 fixture，CPU 运行；这些不是论文数值复现或正式数据实验。
没有更改算法目标、原生默认参数或校准公式，没有运行 PN-CNN 5 seed 或 GPU 实验。

规划 JSON 仅刷新当前源码绑定并纳入新身份/执行文件；22 算法、23 profile 请求、
184 个阻断组合和所有待定预算/签署不变。它仍是 planning schema，不可传给训练入口。
冻结协议/比较矩阵、真实数据收据和历史结果均未重写。

新增专项 **37 passed / 0 skipped**，收据
`/tmp/profile-identity-new-final-20261024.xml`。最终完整快层
**5,641 passed / 59 skipped / 0 failures/errors**，共5,700项、376.179秒，收据
`/tmp/profile-identity-full-fast-20261024.xml`。保留既有环境/资源估算/收敛告警；
CUDA 条件跳过不是 GPU 验证。本批结果未借用父分支的测试计数。

格式、测试/注释质量、文档/API、元数据/数学渲染、skills、目录/层依赖、baseline、
候选失败契约、旧 recipe draft 和当前 proposal 来源检查通过。
完整格式门禁另在暂存交付的独立快照通过，未修改用户原有
`benchmarks/deep_pu/official_data.py`；原有上传也不纳入本批。
离线 wheel/sdist 构建、仓库外安装后的身份映射及原生 UPU 类构造通过；
这不代表仓库外完整跑批、未运行的多平台远程 CI 或正式数值验收已通过。

## 后续优先级

1. 方法族预算、候选与阶段资格的版本化接线，不把原生默认或 smoke 短预算当批准值。
2. 正式五角色 support 消费及图像/逐视图先验适配。
3. 新版 registry / comparison 与正式 v3 单元绑定，再进入对应资源和数值验收。

已验证的身份工程选择不额外请求合作者复核；待定的协议/预算/方法学裁决仍保留。
PN-CNN 5 seed 按用户指令继续暂缓，未代签或关闭任何既有审批事项。
