# CLI 设计：pu-toolbox 命令行入口

日期：2026-08-05
状态：已批准（用户确认 run 单命令+辅助、双文件输入、目录三件套、方案 A argparse、make-demo-data、--save-model 默认关）

## 背景与目标

PUPipeline 已将「数据画像 → 类先验估计 → 模型训练 → 交叉验证 → 评估诊断」封装为
一次 Python 调用，但非专家用户仍需写 Python 代码。CLI 是其上的命令行封装：
用户准备两个 CSV（特征 + PU 标签），一条命令即可得到完整、可审计的训练评估报告。

核心目标：**开箱即用**（`pip install pu-toolbox` 后 `pu-toolbox` 命令直接可用）、
**薄封装**（所有逻辑在 PUPipeline，CLI 只做参数解析/CSV IO/错误映射）、
**维护性**（辅助命令从 registry 实时读取，新算法注册后 CLI 自动可见）。

## 范围

### 包含

- `pu-toolbox run`：一键式全流程，参数与 PUPipeline API 一一对应
- `pu-toolbox list-methods` / `list-priors`：只读辅助命令
- `pu-toolbox make-demo-data`：生成 SCAR 演示数据 CSV
- `[project.scripts]` 入口、测试、文档同步

### 不包含（后续工作）

- skill 集成（`pu-workflow` skill 自然语言驱动，内部调 PUPipeline 或 CLI）
- 模型导出格式（ONNX 等，v1 仅 pickle 可选保存）
- 多命令全面版（profile / sensitivity 独立命令）

## 技术选型

**argparse**（方案 A，已确认）。理由：零新增依赖（项目运行时依赖保持
numpy/scipy/pandas/sklearn 四件套）；项目既有 CLI 惯例即 argparse
（benchmarks 三个 runner）；命令仅 3-4 个，typer/click 收益不抵依赖成本。

## 命令结构

```
pu-toolbox
├── run                  # 核心：一键式全流程
├── list-methods         # 列出注册表全部算法（名称/实现状态/是否需要先验/能否自动实例化）
├── list-priors          # 列出可用先验估计器（recpe / pen_l1 / km1 / km2）
└── make-demo-data       # 生成 SCAR 演示数据（CSV）
```

### run 参数

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--data X.csv` | ✅ | — | 特征矩阵，pandas 读 CSV |
| `--labels y_pu.csv` | ✅ | — | PU 标签 {1,0}，单列，宽容忽略表头 |
| `--out-dir results/` | ✅ | — | 输出目录（report.json + report.md + 控制台 summary） |
| `--true-labels` | — | — | 可选 oracle 评估（映射 `fit_evaluate(y_true=...)`） |
| `--class-prior` | — | — | 显式先验（优先级 1，跳过估计） |
| `--prior-estimator` | — | `recpe` | `recpe`/`pen_l1`/`km1`/`km2`/`none` |
| `--classifier` | — | `auto` | 方法名（`nnpu` 等）或 `auto` |
| `--cv` | — | `5` | 折数 |
| `--metrics` | — | 默认四件套 | 逗号分隔（`pu_risk,recall,auc`） |
| `--seed` | — | `42` | 随机种子 |
| `--save-model` | — | 关 | 额外保存 `model.pkl`（final_model，pickle） |
| `--quiet` | — | 关 | 只打印错误，不打印 summary |

`--prior-estimator=none` 等价于 `PUPipeline(prior_estimator=None)`（禁用自动估计）。
`--class-prior` 与 `--prior-estimator` 同给时以显式先验为准（与 PUPipeline 优先级一致）。

### 辅助命令

- `list-methods`：从 registry 实时读取（`register_all_builtin_methods` +
  `registry.list_algorithms(trainable_only=True)`，官方遍历 API），
  输出表格：名称、family、实现状态（native/api_only）、requires_class_prior、
  能否自动实例化。「能否自动实例化」复用 `pipeline._check_auto_instantiable`
  的检查逻辑（同包内部导入，避免两处逻辑漂移——CLI 判定与 auto 模式实例化判定
  必须一致）。新算法注册后自动出现，无需改 CLI。
- `list-priors`：输出可用先验估计器名（含 `none` 与实例不可用说明）。
- `make-demo-data --out-dir demo/ --n 500 --n-positive 100 --n-features 5 --seed 42`：
  用 `make_scar_data` 生成 `X.csv` / `y_pu.csv` / `y_true.csv` 三个文件，
  产出可直接被 `run` 消费（自洽闭环，同时是 CLI 端到端测试的天然 fixture）。

## 数据流与错误处理

### run 数据流

```
--data/--labels CSV (pandas 读入)
  → validate_pu_X_y（标签规范化、形状、最少正样本）
  → PUPipeline.fit_evaluate(X, y_pu, y_true?, class_prior?)
  → 写 out-dir/report.json（严格 JSON，无 NaN）+ report.md
  → 控制台 print(report.summary())
  → --save-model 时写 model.pkl（pickle final_model）
  → exit 0
```

### 错误处理边界

| 场景 | 行为 |
|---|---|
| 输入文件不存在 / 列数不一致 | stderr 清晰消息 + exit 1 |
| 标签含非法值（非 {1,0}） | `ValidationError` 消息透传 + exit 1 |
| 无效方法名 / 先验名 | `PipelineError` 消息透传 + exit 1 |
| `--out-dir` 已存在 | 覆盖写入（创建时打印提示），不询问 |
| `--cv` / `--seed` 非法 | ValueError 透传 + exit 1 |
| 未知异常 | 完整 traceback + exit 1（保留调试能力） |
| 成功 | summary 打印 + exit 0 |

退出码约定：`0` 成功；`1` 用户/运行错误（`PULearningError`、`ValueError`、
文件错误——stderr 输出清晰消息，无 traceback）；`2` argparse 语法错误（默认）。

## 模块结构

```
pu_toolbox/cli/
    __init__.py      # main() + 子命令分发（argparse subparsers）
    run.py           # run 命令：读 CSV → PUPipeline → 写报告
    info.py          # list-methods / list-priors
    demo.py          # make-demo-data

pyproject.toml:
    [project.scripts]
    pu-toolbox = "pu_toolbox.cli:main"
```

顶层 `pu_toolbox/__init__.py` 不导出 CLI（import 无副作用）。

## 测试（tests/unit/cli/，≤15 方法门禁）

- `parse_args` 单测：默认值、`--metrics` 逗号分隔解析、缺必填报错
- run 端到端（tmp_path 写 CSV fixture）：report.json/report.md 生成且可解析、
  内容键齐全、退出码 0
- `--true-labels`：oracle 指标（auc 等）出现在报告中
- 错误路径：坏文件、坏标签、坏方法名 → SystemExit(1)
- `--save-model`：model.pkl 存在且可 pickle.load 并 predict
- make-demo-data：生成文件可被 run 消费（闭环）
- determinism：同种子跑两次输出一致

## 文档同步

1. `docs/user/cli.md`（新）：命令 + 参数表 + 3 条命令上手示例（make-demo-data → run）
2. `README.md` + `README.zh-CN.md`：CLI 快速上手小节
3. `docs/README.md` 索引 + `docs/project_structure.md`（cli 模块块）
4. `docs/project_management/process_checklist.md` 最近完成记录行
5. `pyproject.toml`：`[project.scripts]`

## 流程

`feature/cli` 分支 → dev-workflow（实现 → 测试 → 4 质量门禁 → ruff check + format → checklist）→ 手动验证（命令行跑一遍 make-demo-data → run 全流程）→ PR。
