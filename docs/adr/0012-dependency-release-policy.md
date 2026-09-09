# ADR-0012:依赖与发布策略

- 状态:已接受
- 触发复审:Python 支持矩阵调整,或打包方式变更时

## 背景

项目是 library,需在 CI 持续验证声明范围内的最新可解析依赖;首版发布时
项目从未出过正式版(0.1.0.dev0),但 roadmap 0.1→0.6 功能已全部完成
(17 算法、6 门禁)。曾发生 `.python-version` 覆盖 CI matrix、未验证却
声明 3.13、源码可用但发布包缺文件的问题。

## 决策

1. **Python 支持收敛 3.10-3.12**;CI 显式锁定 matrix interpreter,分离
   测试、静态门禁和 wheel 安装冒烟。
2. **pyproject.toml 为依赖权威来源**;`uv.lock` 提交入库并在 CI 与本地共用
   (跨平台复现锁,确定性);`requirements.txt` 已由 `uv.lock` 取代并删除。
3. **首版策略**:功能齐全后跳过预热迭代,直接发布首个正式版(版本演进
   记录于 release_process.md)。

## 备选方案

- **uv.lock 不入库(2026-09-08 修订前旧策)**:理由"library 需要验证声明
  范围内最新可解析依赖,锁文件会掩盖漂移"。保留该诉求的新机制:nightly
  (每周 slow+e2e)以 `uv sync --no-lock` 重新解析最新依赖,PR 快层用
  已提交 lock 保证确定性。并入本次修订。
- **requirements.txt 快照(旧策)**:一次开发环境快照,问题复查用,需手工
  与 pyproject 保持同步——已被 `uv.lock` 取代(2026-09-08 删除)。
- **保留 .python-version 覆盖 CI**:曾导致声明 3.13 未验证。否决。

## 后果

- 版本演进/发布状态记录于 `docs/dev/release_process.md`
  与 `docs/dev/process_checklist.md` 发布状态节;本 ADR 不承载版本状态。
- CI matrix 与 extras 的一致性由 check_project_metadata 门禁维护。
- 2026-09-08 修订:因 PU 调研实验执行计划(依赖复现口径)与多环境
  (T600/HENG958 主力机)一致性,uv.lock 入库、requirements.txt 删除;
  nightly --no-lock 承担"最新可解析"验证。
