## 变更目的

<!-- 说明问题、用户影响和本 PR 的边界。 -->

## 主要修改

<!-- 列出公共 API、算法、测试、文档或 benchmark 的关键变化。 -->

## 验证

- [ ] `python -m pytest -q`
- [ ] `python scripts/check_format.py`（ruff check + format --check）
- [ ] `python scripts/check_comment_quality.py`
- [ ] `python scripts/check_test_quality.py`
- [ ] `python scripts/check_doc_links.py`
- [ ] `python scripts/check_project_metadata.py`
- [ ] `python scripts/check_math_rendering.py`
- [ ] `python scripts/check_skill_sync.py`
- [ ] `uv build`
- [ ] `git diff --check`

## 论文与实验边界

- [ ] 不涉及论文算法或 benchmark
- [ ] 已说明 official / clean-room / paper-like 状态
- [ ] 配置、seed、数据来源和生成命令可以追溯
- [ ] 未使用测试真值选择超参数、类先验或阈值

### 与论文／锁定源码不同的处理（如适用）

<!-- 简述：来源行为 → 本项目处理 → 原因 → 验证方式与覆盖边界。
已充分验证的工程选择放在这里，不重复列为待 review 问题；
测试通过不等于论文数值复现，不覆盖尚未验证的数据、配方或 GPU 路径。 -->

### 确实需要合作者确认的点（没有则写“无”）

<!-- 只列尚未解决的方法学、协议／预算或证据冲突；不代替已有正式签署要求。 -->

## 兼容性与管理

- [ ] 公共 API 和 registry metadata 已同步
- [ ] 可选依赖缺失时基础导入仍可用
- [ ] README、架构、目录结构和进度清单按需更新
- [ ] 未提交凭据、本地环境文件或无关生成产物

## 剩余风险

<!-- 写明未覆盖的平台、依赖、数据、GPU 或复现风险。 -->
