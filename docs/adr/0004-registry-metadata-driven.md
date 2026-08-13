# ADR-0004:registry 元数据驱动

- 状态:已接受
- 触发复审:新算法需要注册表之外的发现机制时

## 背景

17 个已注册算法需要统一的发现、推荐与 CLI 展示;每新增算法不应改动
推荐器与 CLI 调用方。

## 决策

- 所有算法经 registry 管理,元数据(name/aliases/family/scenario/assumption/
  requires_class_prior/backend/maturity/source_status/implementation_status/
  training_cost)驱动发现与推荐。
- 注册即被推荐器与 CLI 感知;CLI 辅助命令(list-methods/list-priors)从
  registry 实时读取。
- 别名解析逻辑集中在 `registry/registry.py` 一处。

## 备选方案

- **硬编码算法清单**:每加算法需改多处调用方,漂移风险高。否决。

## 后果

- 元数据与实现必须同步(防漂移测试锁死);`api_only` 不得伪装为可训练实现。
