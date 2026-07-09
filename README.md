# PU Learning Toolbox

Positive-Unlabeled Learning Python Toolbox — sklearn-compatible API, extensible framework, 15 paper methods.

**Status: Phase 0 — planning & scaffolding.** No training code yet.

Full documentation: [`docs/README.md`](docs/README.md)

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
pytest tests/ -v
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
