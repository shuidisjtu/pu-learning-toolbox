# ADR-0002:核心包轻量 + torch 可选依赖

- 状态:已接受
- 触发复审:需要引入 torch 之外的重量级依赖时

## 背景

工具箱面向 PU 研究者,基础安装应保持轻量;深度学习方法依赖 torch,体积与
安装成本显著高于 numpy 技术栈。

## 决策

- Core 包零深度学习依赖;torch 方法放入 optional extras(`[torch]`/`[research]`)。
- 深度估计器默认 `device=None` 自动检测 CUDA(共享 `core/device.py` 单源,
  见 ADR-0011)。
- 可选依赖不得让基础包导入失败;缺失依赖时延迟导入并给出可行动错误。

## 备选方案

- **torch 进必装依赖**:污染轻量安装,非深度用户承担成本。否决。
- **深度方法单独成包**:注册表统一性与推荐器感知被破坏。否决。

## 后果

- `pip install pu-toolbox` 保持轻量;深度方法需显式 extras。
- 每个深度模块需自行保证「无 torch 可导入、有 torch 可用」。
