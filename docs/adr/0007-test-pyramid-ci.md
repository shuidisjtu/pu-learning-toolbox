# ADR-0007:测试金字塔与 CI 分层

- 状态:已接受
- 触发复审:测试执行时间失衡,或 CI 反馈周期超时

## 背景

811+ 测试全量跑慢;慢速/子进程测试拖累 PR 反馈;2026-08-09 CI 曾因本地
漏跑 `ruff format --check` 失败。

## 决策

1. **金字塔分层**:unit/math/property/contract(算法特有/公式/不变量/契约)→
   integration(跨组件)→ e2e(真实子进程旅程)→ slow(慢速)。
2. **CI 分层**:PR 快层 `-m "not slow and not e2e"`(unit + integration +
   静态门禁);nightly 顶层 `-m "slow or e2e"`(每周一 03:23 UTC +
   workflow_dispatch)。
3. **check_format.py 为第 6 道门禁**:ruff check + format --check 全目录,
   本地与 CI 同一入口。
4. **check_test_quality 严格默认**:每文件 ≤15 测试方法、basic/param/edge/
   determ 四分类全覆盖;`--lenient` 显式退出。

## 备选方案

- **全量每次跑**:PR 反馈分钟级变小时级。否决。
- **无测试质量门禁**:分类缺口与超限文件无人发现(曾实际发生)。否决。

## 后果

- PR 反馈周期分钟级;慢速与 e2e 由 nightly 兜底。
- 测试质量(分类/上限)由门禁持续执行,而非依赖人工记忆。
