## 变更目的

<!-- 说明问题、用户影响和本 PR 的边界。 -->

## 主要修改

<!-- 列出公共 API、算法、测试、文档或 benchmark 的关键变化。 -->

## 验证

- [ ] `python -m pytest -q`
- [ ] `ruff check pu_toolbox tests benchmarks examples scripts`
- [ ] `ruff format --check pu_toolbox tests benchmarks examples scripts`
- [ ] `python scripts/check_test_quality.py`
- [ ] `python scripts/check_doc_links.py`
- [ ] `python scripts/check_project_metadata.py`
- [ ] `uv build`
- [ ] `git diff --check`

## 论文与实验边界

- [ ] 不涉及论文算法或 benchmark
- [ ] 已说明 official / clean-room / paper-like 状态
- [ ] 配置、seed、数据来源和生成命令可以追溯
- [ ] 未使用测试真值选择超参数、类先验或阈值

## 兼容性与管理

- [ ] 公共 API 和 registry metadata 已同步
- [ ] 可选依赖缺失时基础导入仍可用
- [ ] README、架构、目录结构和进度清单按需更新
- [ ] 未提交凭据、本地环境文件或无关生成产物

## 剩余风险

<!-- 写明未覆盖的平台、依赖、数据、GPU 或复现风险。 -->
