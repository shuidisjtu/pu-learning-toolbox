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
2. **pyproject.toml 为依赖权威来源**;`uv.lock` 不入库;`requirements.txt`
   仅作开发环境快照(问题复查用,不与 pyproject 手工双向同步)。
3. **首版策略**:功能齐全后跳过 0.x 迭代,直接发布首个正式版(版本演进
   记录于 release_process.md)。

## 备选方案

- **提交 uv.lock**:library 需要验证声明范围内最新可解析依赖,锁文件
  会掩盖漂移。否决。
- **保留 .python-version 覆盖 CI**:曾导致声明 3.13 未验证。否决。

## 后果

- 版本演进/发布状态记录于 `docs/project_management/release_process.md`
  与 `process_checklist.md` 发布状态节;本 ADR 不承载版本状态。
- CI matrix 与 extras 的一致性由 check_project_metadata 门禁维护。
