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

`uv.lock` 提交入库并在 CI 与本地共用（跨平台解析、保证确定性）；nightly 用
`uv sync --upgrade` 重新解析以发现上游兼容问题（仅更新 CI 临时 checkout 的锁文件）。`requirements.txt` 已由
`uv.lock` 取代（2026-09-08 删除）。

**基础导入的可选依赖口径（2026-10-04 复核）**：判据是「**本包自身的 import 语句不引入可选依赖**」——
实测屏蔽 `torch`/`torchvision`/`streamlit` 后 `import pu_toolbox` 成功；本包零**无条件**
module-level `import torch`——模块作用域里 `import` `torch` 的只有两处，且都不可无条件执行：
可执行的那一处在 `pu_toolbox/estimators/deep/vision.py:11` 的 `try:` 内（`from torch import nn`，
带 `except ImportError` 兜底）；另一处 `pu_toolbox/core/device.py:8` 的 `import torch` 在
`if TYPE_CHECKING:` 下、运行时不执行（写「只有一处」会被 AST 复核证伪，故写明两处）。
**本判据不覆盖**「装了可选依赖时基础导入不加载它」：装了 `torch` 时，根包的伞形 re-export 命中
`try` 分支而加载 torch（`torch/hub.py` 再带入 `tqdm`），实验层本身与其 research 依赖
（`densratio`/`lightning`/`sentence_transformers`）**不加载**。要把口径收紧到后者，须把根包
re-export 改为惰性导出——属独立立项（所有者 2026-10-04 裁定）。

## 4. CI 结构

CI 分为四个独立职责：

1. **Tests（PR 快层）**：在 Ubuntu / Windows / macOS 三平台（3 × 3 矩阵）显式使用 Python 3.10/3.11/3.12，安装 dev + torch + torchvision（视觉单元测试真实运行），运行非 slow 且非 e2e 测试（unit + integration）。
2. **Static quality gates**：在 Python 3.11 运行静态质量门禁。**本节不逐条枚举**——权威清单与逐条命令见 `README.md` Development 小节的 Quality gates 命令块与 `.github/workflows/tests.yml` 的 `quality` 作业。此前本节只列了其中 7 道，漏 `check_api_docs`、`check_comment_quality`、`generate_structure`、`generate_layer_deps` 四道，故改为指针，避免再与那两处漂移。
3. **Build and install wheel**：构建 sdist/wheel，在隔离环境安装 wheel，并从仓库目录外验证版本、diagnostics 导入与 registry 条目非空。
4. **Nightly（顶层全量）**：每周一 03:23 UTC 在 3 × 3 矩阵运行 slow + e2e 测试（`-m "slow or e2e"`），依赖同快层（含 torchvision）。

显式解释器断言用于防止 `.python-version` 意外覆盖 CI matrix。

**测试选择口径（2026-10-04 复核）**：CI 的 `quality` 作业只跑静态门禁；测试作业在每次
push/PR 到 `main` 时跑 `pytest tests/ -m "not slow and not e2e"`（`.github/workflows/tests.yml`），
**不按路径过滤**——故改动任何一个子包（含 `experiment/`）都会跑整套 fast 套件。
慢层与端到端（`slow or e2e`）只在 `nightly.yml` 每周跑。因此「Survey 变更不会默认触发通用
工作流全量回归」只能按「不触发慢层全量」理解（所有者 2026-10-04 裁定）：fast 套件全量跑是
设计，不是缺陷。

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
