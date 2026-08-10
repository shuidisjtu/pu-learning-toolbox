# 开发与兼容性政策

## 1. 支持范围

| 项目 | 当前策略 |
|---|---|
| Python | 3.10、3.11、3.12 |
| 推荐开发版本 | 3.11（`.python-version`） |
| 核心依赖 | numpy、scipy、pandas、scikit-learn |
| 深度学习 | `torch` extra |
| 研究扩展 | `research` extra，包含 densratio/torch/torchvision/lightning/tqdm |
| CI 平台 | GitHub Actions `ubuntu-latest` / `windows-latest` / `macos-latest` |
| 构建后端 | Hatchling |

Python classifier、ruff target、CI matrix 和本文档必须保持一致。新增 Python 版本前，需要完成完整测试和 wheel 安装冒烟；删除版本支持时必须在发布说明中记录。

## 2. 安装组合

```bash
pip install -e .                 # 核心 numpy/sklearn 功能
pip install -e ".[torch]"       # nnPU、Dist-PU 和深度方法
pip install -e ".[research]"    # uLSIF、图像与 Lightning 研究环境
pip install -e ".[all]"         # 全部运行时功能，不含开发工具
pip install -e ".[dev]"         # 测试、lint 和构建工具
```

可选依赖采用延迟导入。只安装核心依赖时，`import pu_toolbox` 必须正常工作；调用需要
torch 的训练路径或 PUSB benchmark 的 uLSIF 对照时，再提供明确安装提示。

## 3. 依赖来源

`pyproject.toml` 是安装规范。作为 library，项目使用最低版本约束并在 CI 中重新解析依赖，用于发现上游兼容问题。

`uv.lock` 不提交到仓库；它可以在本地由 `uv sync` 临时生成。`requirements.txt` 是一次已知开发环境快照，仅用于重现特定问题，不代表全部受支持组合。

## 4. CI 结构

CI 分为四个独立职责：

1. **Tests（PR 快层）**：在 Ubuntu / Windows / macOS 三平台（3 × 3 矩阵）显式使用 Python 3.10/3.11/3.12，安装 dev + torch，运行非 slow 且非 e2e 测试（unit + integration）。
2. **Static quality gates**：在 Python 3.11 运行格式门禁（`check_format.py`：ruff check + format --check 全目录）、测试质量、文档一致性、项目 metadata 一致性、方法卡 MathJax 渲染检查和 Skill 同步检查。
3. **Build and install wheel**：构建 sdist/wheel，在隔离环境安装 wheel，并从仓库目录外验证版本、diagnostics 导入与 registry 条目非空。
4. **Nightly（顶层全量）**：每周一 03:23 UTC 在 3 × 3 矩阵运行 slow + e2e 测试（`-m "slow or e2e"`）。

显式解释器断言用于防止 `.python-version` 意外覆盖 CI matrix。

## 5. 构建内容

Hatchling 配置位于 `pyproject.toml`：

- wheel 只包含 `pu_toolbox` 和 distribution metadata。
- sdist 包含源码、测试、benchmark、文档、示例、脚本和项目配置。
- 构建不依赖 `MANIFEST.in`。

发布前运行：

```bash
uv build
```

并在干净环境中安装生成的 wheel，避免本地源码目录掩盖缺失文件。

## 6. 已知边界

- 深度方法的基础接口由普通 torch 环境验证，论文级 GPU、CUDA 和历史依赖环境另行锁定。
- DGPU 的完整实验需要外部 EDM/扩散生成器后端。
- clean-room 和 paper-like 结果不等同于官方历史环境复现。
