# ADR-0004:registry 元数据驱动

- 状态:已接受
- 触发复审:新算法需要注册表之外的发现机制时

## 背景

已实现的算法需要统一的发现、推荐与 CLI 展示;每新增算法不应改动
推荐器与 CLI 调用方。

## 决策

- 所有算法经 registry 管理,元数据(name/aliases/family/scenario/assumption/
  requires_class_prior/backend/maturity/source_status/training_cost)驱动发现与推荐。
- 注册即被推荐器与 CLI 感知;CLI 辅助命令(list-methods/list-priors)从
  registry 实时读取。
- `register_method(metadata, estimator_cls)` 同时登记元数据和具体实现类；没有实现的占位符不能注册。
- 内置条目在同一处记录元数据和实现类路径。导入 registry 不加载内置目录；首次公共查询、
  发现或注册时按需初始化。`get_algorithm` 和 `get_metadata` 无需调用者先初始化。
- `clear_registry()` 重置状态，下次公共操作会重新按需加载。
- 规范算法 ID 使用小写，查询忽略大小写；仅大小写不同的拼写不登记为别名。

## 备选方案

- **硬编码算法清单**:每加算法需改多处调用方,漂移风险高。否决。

## 后果

- 合入前用测试验证实现的公共契约及其声明的能力，并检查元数据与实现一致；运行时不执行测试。
- `source_status` 与 `maturity` 分别表达来源和成熟度，不把可用性等同于稳定性或论文复现程度。
