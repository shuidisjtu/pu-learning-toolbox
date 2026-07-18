# PU Learning Toolbox

Positive-Unlabeled Learning Python Toolbox — sklearn-compatible API, extensible framework, 15 paper methods.

**Status: Phase 1 — 核心 PU 风险估计进行中。** 包骨架、core 基类、registry 已完成，5/8 算法已实现 NATIVE（Elkan-Noto、uPU、nnPU、ReCPE、PNU），测试套件通过（270 tests）。15 篇论文方法中 10 个为 `api_only` 占位，逐个集成中。

Full documentation: [`docs/README.md`](docs/README.md)

项目管理文档: [`docs/project_management/`](docs/project_management/) — 决策日志、进度清单

## Quick Start

```bash
# 1. Clone
git clone https://github.com/shuidisjtu/pu-learning-toolbox.git
cd pu-learning-toolbox

# 2. Create virtual environment
uv venv              # reads .python-version, uses Python 3.11

# 3. Install
uv pip install -e ".[dev]"

# 4. Verify
uv run pytest tests/ -v
uv run ruff check pu_toolbox tests
uv run ruff format --check pu_toolbox tests
```

### 使用 pip / conda？

项目推荐 Python 3.11，最低要求 **Python >= 3.10**（`pyproject.toml` 声明）。`.python-version` 仅被 uv / pyenv 识别，对其他工具链无约束力，但可作为推荐版本的声明。

```bash
# pip
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# conda
conda create -n pu-toolbox python=3.11
conda activate pu-toolbox
pip install -e ".[dev]"
```

## Python Version & Compatibility

工具箱核心要求 **Python >= 3.10**，开发基线为 **Python 3.11**。

集成的论文源码（位于 `external/`）各有各的 Python 版本和依赖要求，**不要求统一改写**。SourceAdapter 通过以下方式桥接不兼容的代码：

- **同进程**：无冲突时直接 import
- **子进程**：Python 版本兼容但依赖冲突时，通过独立 venv + subprocess 通信
- **Docker**：Python 版本不兼容（如老旧 TF 1.x 代码），容器化运行，adapter 负责对接

工具箱只保证 `pu_toolbox` 自身 API 稳定，不替你管理论文源码的环境。

## 测试

```bash
uv run pytest tests/ -v                    # 全部测试
uv run pytest tests/ -v -m "not slow"      # 快速反馈（跳过慢速）
uv run pytest tests/ -v -m math            # MATH golden tests（失败 = 代码 bug）
uv run pytest tests/ -v -m property        # PROPERTY 不变量测试（失败 = 代码 bug）
uv run pytest tests/ -v -m contract        # 跨分类器 API 契约
```

测试分层：`contract/`（写一次、所有分类器复用）→ `unit/`（算法特有逻辑）→ `integration/`（跨模块）→ `regression/`（慢速/论文复现）。详见 `CLAUDE.md`。

## 开发流程

- `main` 分支保持稳定可运行，不直接提交代码。
- 每个功能/修复开独立分支：`feature/<name>` 或 `fix/<name>`，完成后提 PR 合并。

## 文档约定

方法卡等 Markdown 文件中的数学公式使用 **GitHub 保护定界符**（避免 `_` 被 Markdown 误解析为斜体）：

- 行内公式：`` $`...`$ ``（而非 `$...$`）
- 块公式：```` ```math ... ``` ````（而非 `$$...$$`）

详见 [GitHub 官方文档](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)。
