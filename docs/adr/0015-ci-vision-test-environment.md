# ADR-0015:CI 测试环境安装 torchvision

- 状态:已接受
- 触发复审:视觉测试规模显著扩大(如新增多组 resnet/增强矩阵),或 CI 安装时长成为瓶颈时

## 背景

`torch` extra 只声明 `torch>=2.0`(ADR-0002),`torchvision` 归属 `research` extra;
CI 快层与 nightly 均按「dev + torch」安装(tests.yml / nightly.yml)。2026-08-17
E2/E3 CNN 序列化修复(79504d6)引入的视觉测试因此空转:

- `tests/unit/estimators/test_deep_vision_pickle.py` 的 augmentation 测试运行时
  `from torchvision.transforms import v2` 失败,CI 直接 ImportError 红一次;
- `tests/unit/estimators/test_deep_pu_vision.py` 整文件(CNN backbone、resnet18/50、
  SimAugment/RandAugment 共 7 个测试)模块级 skip torchvision,CI 上从未真实执行。

即:v1.5.1 核心回归的视觉部分在 CI 上零覆盖,只有本地装了 torchvision 才运行。

## 决策

1. **CI 测试环境显式安装 torchvision**:`tests.yml`(PR 快层)与 `nightly.yml` 在
   `uv sync` 后追加 `uv pip install "torchvision>=0.15"`。
2. **不修改 extras 语义**:`torch` extra 保持仅 torch,用户轻量安装不变(ADR-0002);
   CI 环境比用户 extra 更完整是测试环境的刻意差异,而非依赖声明变化。
3. **测试内保留 `pytest.importorskip("torchvision")` 作为本地防御**:无 torchvision
   的本地环境(如仅 `.[torch]` 的用户复现测试)优雅跳过而非失败,与
   `test_deep_pu_vision.py` 既有风格一致。

## 备选方案

- **仅测试内 skip,CI 不装**:改动最小,但 E2/E3 回归与全部视觉测试在 CI 永久
  空转,回归保护形同虚设。否决。
- **torchvision 并入 `torch` extra**:所有 `pip install .[torch]` 用户被迫多装
  一个包,违反 ADR-0002 轻量原则;CNN 视觉是可选能力,不应抬高基础门槛。否决。
- **视觉测试标记 slow 移入 nightly**:改变测试层级语义,快层对视觉路径零覆盖,
  且 nightly 同样需要装 torchvision。否决。

## 后果

- CI 快层视觉测试真实运行:982 → 990(快层 +8 个,nightly 层相应恢复)。
- CI 安装成本:troch 已约 2GB,torchvision 增量几 MB,uv 缓存可复用,可忽略。
- 后续新增视觉测试无需再评估 CI 环境是否具备依赖;测试内 importorskip 仍须保留,
  保持「无 torchvision 环境可复现」的兼容面。
- CI 环境与 `pyproject.toml` extras 的差异集中在唯一一处显式
  `uv pip install "torchvision>=0.15"`(两工作流各一行),已同步
  `docs/dev/compatibility.md` §4。
