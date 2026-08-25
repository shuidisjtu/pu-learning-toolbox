# 发布流程(Releases)

> v1.0.0 首版发布于 2026-08-10(PyPI + GitHub Release)。本文档把发布流程固化为可复用清单,后续每个版本(1.0.x / 1.x.0 / 2.0.0)按此执行。

## 1. 版本策略

| 版本 | 触发条件 | 示例 |
|---|---|---|
| `1.0.x` | 只修 bug,不破坏 API | 用户反馈修复 |
| `1.x.0` | 新功能,向后兼容 | 新算法、新 CLI 子命令 |
| `2.0.0` | 破坏性变更 | 标签语义、参数删除、契约变更 |

版本号在 `pyproject.toml` 维护,并同步 `README` 徽章与 `process_checklist.md` 发布状态节。

## 2. 发布前检查(我侧,本地可完成)

1. **质量基线**:`uv run pytest tests/ -m "not slow and not e2e"` 全绿;7 道质量门禁全绿
2. **元数据**:`pyproject.toml` 版本号、`Development Status` classifier 与版本匹配、`name` 未被占用(可选:`curl -s https://pypi.org/pypi/<name>/json` 返回 404 即可用)、LICENSE/readme/入口点齐全
3. **构建预检**:
   ```bash
   uv build                              # 出 sdist + wheel
   uvx twine check dist/*                # 元数据校验,必须 PASSED
   ```
   检查 wheel 内容只含包本体(`py3-none-any`,无 test/`__pycache__`)
4. **干净环境验收**(自动化测试无法覆盖"安装后"路径,必做):
   ```bash
   uv venv /tmp/pu-release-venv --python 3.11
   uv pip install --python <venv>/Scripts/python.exe dist/*.whl
   # 在 venv 内跑 quickstart 三步:make-demo-data → run → report.json schema_version 校验
   # 跑 PUPipeline 最小流程 + 抽查 examples/minimal/ 1-2 个脚本
   ```
   验收后清理临时 venv

## 3. 上传 PyPI(用户侧)

1. 登录 https://pypi.org/manage/account/,确认 2FA 已启用(强制,启用后不可禁用;保存恢复码)
2. `API tokens → Add API token`,**Scope 选 "Project: <包名>"**(最小权限)
3. 用户终端执行(token 不进对话,不共享):
   ```bash
   cd <repo>
   $env:UV_PUBLISH_TOKEN = "pypi-..."   # PowerShell;Git Bash 用 export
   uv publish                            # 默认上传 https://upload.pypi.org/legacy/
   ```
4. 验证:`curl -s https://pypi.org/pypi/<name>/json` 返回 200

## 4. 发布后收尾

1. **从 PyPI 实测安装**(真实用户视角):
   ```bash
   uv venv /tmp/pu-pypi-check --python 3.11
   uv pip install --python <venv>/Scripts/python.exe <name>
   # 跑 quickstart 三步 + list-methods
   ```
2. **更新安装说明**:README.md / README.zh-CN.md / docs/user/quickstart.md 中的安装命令与"尚未发布"表述
3. **GitHub Release**:main 打 tag(如 `v1.0.0`)+ 创建 Release(功能摘要见 `process_checklist.md` 发布状态节与 commit log)
   - **Release notes 内容要求**:只写用户可感知的变更——新功能(API/CLI)、修复、行为变化。
     - **不写**验证结果废话:测试数量、"门禁全绿"、CI 通过等一律不写——测试不通过根本不会进入发布流程,写了就是噪音
     - **不写**工程内部事项:架构治理、文档目录调整、构建流程、开发者内部 skill 的改动等(对 `pip install` 用户无意义)
     - **要写**用户可感知的 skill 变化:skill 的功能或使用方法变化(如新增 `pu-toolbox skill install` 子命令、skill 行为/参数变化)属于用户可见变更,应列出
     - **不写**依赖外部环境的未完成项;未包含的功能(如 v1 范围外)单独列小节标注「未包含」,不得混入功能列表
     - **不写**本机路径等内部细节(如 `F:/lab` 这类验证目录)——路径对用户无意义
     - 语言:中文,保持简洁准确
4. 跑 7 道门禁,提交文档改动

## 5. 回滚与纠错

- PyPI **不能删除版本**,只能 **yank**(标记不可安装;已安装用户不受影响)
- 严重问题:yank 当前版本 → 修复 → 发下一个 patch 版本
- GitHub Release 可删除重打(不影响已发布的 PyPI 版本)

## 6. 维护与权限

- **联合开发者**:PyPI 项目页 `Manage → Collaborators` 添加 —— `Maintainer`(上传/yank)或 `Owner`(全部控制);各用各的账号与 token
- **Owner 转让**:添加为 Owner → 原 Owner 自行移除;PyPI 与 GitHub 权限互不相通,需分别转让
- **上游依赖升级**:CI 每次重新解析最新依赖,nightly 捕捉兼容问题;新 Python 版本经 CI 矩阵验证后再声明支持
- **自动化进阶(可选)**:GitHub Actions 打 tag 自动 `uv publish`(token 存 GitHub Secrets)+ 自动创建 Release
