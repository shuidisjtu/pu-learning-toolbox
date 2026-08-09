# 架构腐朽修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `docs/dev/architecture_audit.md` 的 14 项行动项修复架构腐朽信号,分 P0(正确性 3 项)→ P1(真相分裂/治理盲区 4 项)→ P2(文档/流程 7 项)三批,每批独立验证。

**Architecture:** P0 修复三处正确性与治理盲区(KLDCE 偏置恢复、check_doc_links 补洞、check_math_rendering 锚定);P1 单源化 RBF 公式与 class_prior 校验、落地 paper marker、加豁免退出机制;P2 清理死代码与文档漂移。每项均带 TDD 测试(先写失败测试)。

**Tech Stack:** Python 3.10+ / numpy / scipy / ruff / pytest(现有栈);无新依赖。

## Global Constraints

- 执行前开 `fix/architecture-decay` 分支,完成全部后提 PR 合并 main(项目分支规范)
- 每批结束时必须全绿:6 道门禁 + `uv run pytest tests/ -m "not slow" -q`
- 修改文件前必须先 Read(hooks 要求);删除文件逐个删除
- 命令一律 `uv run` 前缀;Python 中路径用 `r'E:\...'` 或 `'E:/...'`
- Git 提交不加 `Co-Authored-By`
- KLDCE 数学变更若导致现有 KLDCE 测试/基准数值回归,停下汇报,不得强行通过
- 每步的"运行测试"命令统一为:`uv run pytest tests/ -v -k <pattern>`,提交命令统一为:`git add <files> && git commit -m "..."`

---

## 批次 A:P0 正确性(3 项)

### Task 1: KLDCE 偏置恢复代码/推导矛盾修复

**Files:**
- Modify: `pu_toolbox/estimators/risk/kldce.py:581-637`
- Test: `tests/estimators/risk/test_kldce_math.py`(追加)

**Interfaces:**
- Consumes: `_recover_bias_from_kkt(g, alpha, gamma, y_tilde, C_alpha, C_gamma, k, lambda_)`(现有签名,返回 `(b0, info)`);`build_rbf_basis` 无关
- Produces: 修复后的 KKT 互补松弛分支;新测试 `test_edge_gamma_free_neg_one_minus_g`

**复核依据(Step 1 执行)**:KLDCE 卡 `docs/research/method_cards/KLDCE.md:165`("QP oracle 版:由自由 α 支持向量的 KKT margin 条件恢复 b₀(中位数)")与互补松弛推导:原始问题 `ỹᵢ f(xᵢ) ≥ 1 − ξᵢ`,ỹ=+1(P)则 free α → f=1 → `b=1−g`;ỹ=−1(U)则 free γ → f=−1 → `b=−1−g`。代码 4 处违反:free_alpha_mask 无 ỹ 区分(581-584)、free γ 用 `1−g`(588-590)、fallback γ 两分支用 `1−g`(629-637)。

- [ ] **Step 1: 复核数学正确性(对照卡 §165 与论文附录)**

确认互补松弛:U 样本(ỹ=−1)free 时 `b = −1−g`。依据:函数 docstring(kldce.py:526 `bᵢ = 1 − gᵢ`)与内部推导注释(604-606 `γⱼ=0 → b₀ ≤ −1 − g`)已互相矛盾,三处表述不一致本身即缺陷。若对照论文发现 `1−g` 正确,则反向修复(统一为 `1−g` 并修推导注释)——两者必居其一,不允许保留矛盾。

- [ ] **Step 2: 写失败测试(追加到 test_kldce_math.py)**

```python
def test_edge_gamma_free_bias_uses_neg_one_minus_g():
    """自由 γ(U 样本,ỹ=−1)恢复 b₀ 必须用 −1−g 而非 1−g.

    构造 k=1 正样本 + 1 个 U 样本:alpha 全在下界(非 free),
    仅 γ 为 free(0.5)。手工计算:
      sigma=1, X=[[0],[1]], λ=1, C_eq=0 →
      g[1] = (0 − 0.5·(−1)·K[1,1])/(2·1) = 0.25
      free γ → b₀ = −1 − g[1] = −1.25
    修复前代码返回 1 − g[1] = 0.75(差 2)。
    """
    import numpy as np
    from pu_toolbox.estimators.risk.kldce import _rbf_kernel, _recover_bias_from_kkt

    X = np.array([[0.0], [1.0]])
    sigma = 1.0
    K = _rbf_kernel(X, X, sigma)  # [[1, exp(-0.5)], [exp(-0.5), 1]]
    b0, info = _recover_bias_from_kkt(
        alpha=np.array([1e-13]),          # 下界,非 free
        gamma=np.array([0.5]),            # free γ
        X=X, K=K,
        y_tilde=np.array([1.0, -1.0]),
        mu=np.array([0.0]), lambda_=1.0, sigma=sigma,
        C_eq=0.0, C_alpha=1.0, C_gamma=1.0, k=1,
    )
    assert info["bias_recovery"] == "free_median"
    assert abs(b0 - (-1.25)) < 1e-9  # 修复前返回 0.75
```

运行:`uv run pytest tests/estimators/risk/test_kldce_math.py -v -k gamma_free`
预期:FAIL,`b0 == 0.75`(断言 `abs(0.75 - (-1.25)) < 1e-9` 不成立)

- [ ] **Step 3: 修复 kldce.py 四处分支**

```python
    # Free α (all samples): 0 < αᵢ < C_alpha;ỹᵢ = +1 → f=1,ỹᵢ = −1 → f=−1
    free_alpha_mask = (alpha > 1e-12) & (alpha < C_alpha - 1e-12)
    for i in np.where(free_alpha_mask)[0]:
        if y_tilde[i] > 0:
            b_estimates.append(1.0 - g[i])    # ỹ=+1: b = 1−g
        else:
            b_estimates.append(-1.0 - g[i])   # ỹ=−1: b = −1−g

    # Free γ (U samples): 0 < γⱼ < C_gamma → f = −1 → b = −1−g
    free_gamma_mask = (gamma > 1e-12) & (gamma < C_gamma - 1e-12)
    for j in np.where(free_gamma_mask)[0]:
        # ỹ_{k+j} = −1
        b_estimates.append(-1.0 - g[k + j])
```

以及 fallback 两分支(629-637 行)改为 `-1.0 - g[k + j]`,同步修正两处行内注释(现注释 `b ≤ 1−g` 与 605-606 行推导矛盾),并修正函数 docstring(526 行 `bᵢ = 1 − gᵢ`)为区分 ỹ 的表述:

```
For free support vectors, KKT implies ỹᵢ·f(xᵢ) = 1:
    ỹ=+1 → b = 1 − g;ỹ=−1 → b = −1 − g
```

- [ ] **Step 4: 运行新测试 + 现有 KLDCE 全量测试**

```bash
uv run pytest tests/estimators/risk/test_kldce_math.py -v   # 全部通过
```

预期:PASS(含原有 oracle/约束测试)。若任何既有测试数值回归,停下汇报(约束:Global Constraints)。

- [ ] **Step 5: Commit**

`git add pu_toolbox/estimators/risk/kldce.py tests/estimators/risk/test_kldce_math.py && git commit -m "fix(kldce): bias recovery uses -1-g for unlabeled free samples"`

### Task 2: check_doc_links 三处空洞补全

**Files:**
- Modify: `scripts/check_doc_links.py:47,53-58,61-65`
- Test: `tests/unit/scripts/test_check_doc_links.py`(新建)

**Interfaces:**
- Produces: `PATH_PATTERN` 扩展(支持 .py + .md)、`MD_LINK_PATTERN`(markdown 链接)、索引完备性扩展到 docs 子目录

- [ ] **Step 1: 写失败测试(新建 test_check_doc_links.py,参照 test_check_format.py 的 sys.path 模式)**

```python
"""Tests for the doc-links gate extension (orphan + md-link detection)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import check_doc_links as d  # noqa: E402


def _make_tree(tmp_path, files, links):
    """files: {relative_path: content}; links: [(src, target_md)] to append."""
    root = tmp_path / "docs"
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


@pytest.mark.unit
def test_orphan_md_reported(tmp_path):
    """A docs/md file listed in no index is reported by rule-4."""
    files = {
        "README.md": "- [index](../docs/index.md)\n",
        "index.md": "- [orphan](orphan.md)\n",
        "orphan.md": "# orphan\n",
    }
    d._find_md_files = lambda: [tmp_path / "docs" / "index.md"]
    ...
```

(执行时按 `check_doc_links.py` 现有 Issue/check 函数结构补齐断言;失败路径:孤儿 md 或悬空 markdown 链接不产生 Issue。)

- [ ] **Step 2: 实现三处扩展**

```python
# ① PATH_PATTERN 同时匹配 .py 与 .md(47 行)
PATH_PATTERN = re.compile(rf"`((?:{_PATH_ROOT_ALT})/[^`]+\.(?:py|md))`")

# ② markdown 链接目标存在性(新增规则 rule-5)
MD_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

# ③ _EXCLUDED_DOC_DIRS 缩小为超链接频繁的外部目录
_EXCLUDED_DOC_DIRS: set[str] = {"superpowers", "figures"}
# research/ 与 project_management/ 纳入常规路径检查;research 内外部 URL(http://)
# 与相对卡内锚点(#section)由 rule-5 专门处理(URL 前缀 http/https/mailto 跳过)
```

索引完备性(rule-4):从"docs 顶层 + PM_FILES_EXPECTED 白名单"改为"docs/ 全树 rglob,排除 superpowers/figures;每篇 .md 必须被某索引引用(README.md 除外,与 DOC_INDEX_EXCLUDED 语义一致)"。PM_FILES_EXPECTED 白名单删除,`pu_workflow_design.md` 类孤儿自动暴露。

- [ ] **Step 3: 更新 tests.yml 不受影响 → 运行门禁自检**

```bash
uv run python scripts/check_doc_links.py
```

预期:exit 0(现有文档无悬空 .md 引用;`docs/dev/architecture_audit.md` 等新文档已入索引——若报 orphan,按规则补齐索引,这正是门禁开始工作的证明)

- [ ] **Step 4: Commit**

`git add scripts/check_doc_links.py tests/unit/scripts/test_check_doc_links.py && git commit -m "fix(scripts): close doc-links gate holes (md links, research dir, index completeness)"`

### Task 3: check_math_rendering 锚定 PROJECT_ROOT

**Files:**
- Modify: `scripts/check_math_rendering.py:84`
- Test: `tests/unit/scripts/test_check_math_rendering.py`(新建)

- [ ] **Step 1: 写失败测试(cwd 无关性)**

```python
@pytest.mark.unit
def test_scans_method_cards_from_any_cwd(tmp_path, monkeypatch):
    """Run from a temp cwd: still scans the real method_cards dir."""
    monkeypatch.chdir(tmp_path)
    rc = check_math_rendering.main([])
    assert rc == 0  # 真实文件被扫描,无数学语法问题
```

运行:预期 FAIL(当前裸 glob 在 tmp cwd 空扫,但空扫也返回 0——测试改用 monkeypatch 检查 `glob.glob` 调用结果,或让 main 接受显式目录参数)。

- [ ] **Step 2: 修复锚定**

```python
# check_math_rendering.py 头部(参照 check_doc_links.py:31 模式)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
...
def main(argv: list[str] | None = None) -> int:
    files = sorted((PROJECT_ROOT / "docs" / "research" / "method_cards").glob("*.md"))
    if not files:
        print("No method cards found; refusing to pass empty scan.", file=sys.stderr)
        return 1
```

空扫不再可能 exit 0;误目录运行报错退出。

- [ ] **Step 3: 运行新测试 + 门禁自检 + Commit**

```bash
uv run pytest tests/unit/scripts/test_check_math_rendering.py -v
uv run python scripts/check_math_rendering.py   # exit 0,扫描到 9+ 张卡
```

`git add scripts/check_math_rendering.py tests/unit/scripts/test_check_math_rendering.py && git commit -m "fix(scripts): anchor math-rendering gate to PROJECT_ROOT"`

### 批次 A 收尾

- [ ] **Step A1: 批次验证**

```bash
uv run python scripts/check_format.py
uv run python scripts/check_test_quality.py
uv run python scripts/check_doc_links.py
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
uv run python scripts/check_skill_sync.py
uv run pytest tests/ -m "not slow" -q
```

预期:6 门禁全绿 + 全量测试通过。向用户汇报批次 A 结论(KLDCE 复核结论、门禁行为变化)。

---

## 批次 B:P1 真相分裂与治理盲区(4 项)

### Task 4: RBF 核公式单源化(5 份 → 1 份)

**Files:**
- Modify: `pu_toolbox/estimators/risk/kldce.py:59-83`、`pu_toolbox/prior/pen_l1.py:57-60`、`pu_toolbox/prior/kernel_mean.py:140`、`pu_toolbox/estimators/bias_aware/pusb_kernel.py:45`(注释指向)
- Test: `tests/unit/test_basis_single_source.py`(新建)

**Interfaces:**
- Consumes: `utils.basis.build_rbf_basis(X, centers, kernel_width)`(现有,公式 `exp(-||x-c||²/(2σ²))`)
- Produces: kldce/pen_l1/kernel_mean 三处改为委托;pusb_kernel 保留但加指向注释

- [ ] **Step 1: 写单源化一致性测试**

```python
@pytest.mark.unit
def test_rbf_formula_single_source():
    """kldce/pen_l1/kernel_mean 的 RBF 公式必须与 utils.basis 一致."""
    rng = np.random.RandomState(0)
    X = rng.randn(20, 3)
    Z = rng.randn(5, 3)
    sigma = 0.7
    expected = build_rbf_basis(X, Z, sigma)
    from pu_toolbox.estimators.risk.kldce import _rbf_kernel
    assert np.allclose(_rbf_kernel(X, Z, sigma), expected, atol=1e-12)
    # pen_l1 / kernel_mean 分支在各自测试中已有 golden 断言,此处防回归即可
```

- [ ] **Step 2: 三处委托修改**

kldce.py `_rbf_kernel` 函数体改为:

```python
def _rbf_kernel(X: np.ndarray, Z: np.ndarray, sigma: float) -> np.ndarray:
    """RBF / Gaussian kernel (single-sourced in ``utils.basis``).

    .. math::

        K(x,z) = \\exp\\left(-\\frac{\\|x-z\\|^2}{2\\sigma^2}\\right)

    Relation to sklearn gamma: ``gamma = 1 / (2 * sigma**2)``.
    """
    from pu_toolbox.utils.basis import build_rbf_basis

    return build_rbf_basis(X, Z, sigma)
```

pen_l1.py:57-60 与 kernel_mean.py:140 内联公式同法替换为 `build_rbf_basis` 调用(pen_l1:`build_rbf_basis(P, centers, self.sigma)` / `build_rbf_basis(U, centers, self.sigma)`;kernel_mean:`build_rbf_basis(X, X, width)`——注意该处核是 X×X 且宽度变量名不同,`kernel_width=width`)。

pusb_kernel.py `_rbf_design`(输入为预计算 distances 矩阵且含截距列,形态不同)保留实现,docstring 加一行"公式与 ``utils.basis.build_rbf_basis`` 相同,输入形态为距离矩阵(含截距列),不合并"。

- [ ] **Step 3: 全量验证(数值等价的强校验)**

```bash
uv run pytest tests/ -v -k "rbf or kldce or pen_l1 or kernel_mean or pusb_kernel"
uv run pytest tests/estimators/risk/test_kldce_math.py -v   # golden 数值必须不变
```

预期:全部通过,golden 数值无回归(展开式与 cdist 差异 < 1e-12,atol 覆盖)。

- [ ] **Step 4: Commit**

`git add pu_toolbox/estimators/risk/kldce.py pu_toolbox/prior/pen_l1.py pu_toolbox/prior/kernel_mean.py pu_toolbox/estimators/bias_aware/pusb_kernel.py tests/unit/test_basis_single_source.py && git commit -m "refactor: single-source RBF kernel into utils.basis"`

### Task 5: paper marker 落地

**Files:**
- Modify: `tests/benchmarks/test_pusb_table2_data.py`、`tests/benchmarks/test_pusb_table2_benchmark.py`(追加 marker)
- Verify: `tests/benchmarks/test_deep_pu_model_selection.py`

- [ ] **Step 1: 挂载 marker(两文件各加一行)**

```python
# tests/benchmarks/test_pusb_table2_data.py 顶部(现有 pytestmark 处)
pytestmark = pytest.mark.unit
```

改为:

```python
pytestmark = [pytest.mark.unit, pytest.mark.paper]
```

`test_pusb_table2_benchmark.py` 同法。先检查 `test_deep_pu_model_selection.py` 是否需要网络/官方源码:若纯本地逻辑也加 `pytest.mark.paper`。

- [ ] **Step 2: 验证 marker 生效**

```bash
uv run pytest tests/ -m paper -q --collect-only | tail -3
```

预期:收集 ≥2 个用例(此前为 0)。再跑 `uv run pytest tests/ -m "not slow" -q` 确认 paper 测试仍被默认运行(不改变现有 unit 行为)。

- [ ] **Step 3: Commit**

`git add tests/benchmarks/test_pusb_table2_data.py tests/benchmarks/test_pusb_table2_benchmark.py && git commit -m "test: attach paper marker to PUSB Table 2 benchmark tests"`

### Task 6: class_prior 校验统一走 check_scalar_in_range

**Files:**
- Modify: 9 个分类器 `fit` 内的内联校验 → `pu_toolbox/core/validation.py` 的 `check_scalar_in_range`
- Test: 既有 `test_param_invalid_*` 契约测试验证

**9 个文件与现状**(执行时逐一 Read 后修改):
`risk/upu.py`、`risk/pnu.py`、`risk/nnpu.py:182-183`、`risk/dist_pu.py`、`deep/self_pu.py`、`classic/llsvm.py`、`bias_aware/pusb_kernel.py`、`risk/ldce.py`、`risk/kldce.py`

- [ ] **Step 1: 先查错误消息断言面**

```bash
grep -rn "must be in\|class_prior.*(0\|pi.*0.*1\|class_prior.*range" tests/ | head -20
```

确定哪些测试断言了内联错误消息文本;统一后的消息为 `"class_prior must be in (0, 1); got {value}."`(check_scalar_in_range 开区间格式),需同步更新的测试列清单。

- [ ] **Step 2: 逐文件替换(示例,以 nnpu.py:182-183 为模板)**

```python
# 替换前(内联)
if not 0.0 < class_prior < 1.0:
    raise ValueError(f"class_prior must be in (0, 1); got {class_prior}.")

# 替换后
from pu_toolbox.core.validation import check_scalar_in_range
...
check_scalar_in_range(class_prior, 0.0, 1.0, "class_prior", inclusive=False)
```

- [ ] **Step 3: 运行契约测试 + 消息断言测试**

```bash
uv run pytest tests/ -v -k "class_prior or contract or invalid"
```

预期:全部通过(消息文本变化若被断言,同步更新断言为 check_scalar_in_range 格式)。

- [ ] **Step 4: Commit**

`git add pu_toolbox/estimators/risk/upu.py pu_toolbox/estimators/risk/pnu.py pu_toolbox/estimators/risk/nnpu.py pu_toolbox/estimators/risk/dist_pu.py pu_toolbox/estimators/deep/self_pu.py pu_toolbox/estimators/classic/llsvm.py pu_toolbox/estimators/bias_aware/pusb_kernel.py pu_toolbox/estimators/risk/ldce.py pu_toolbox/estimators/risk/kldce.py && git commit -m "refactor: unify class_prior range validation into check_scalar_in_range"`

### Task 7: check_test_quality 豁免名单退出机制

**Files:**
- Modify: `scripts/check_test_quality.py`(UNLIMITED_FILES / CONTRACT_COVERED_FILES 使用处)
- Test: `tests/unit/scripts/test_check_test_quality_exemptions.py`(新建)

- [ ] **Step 1: 写测试:豁免文件不再需要豁免时应报 warning**

```python
@pytest.mark.unit
def test_exemption_reporting(tmp_path):
    """豁免名单文件若已满足覆盖规则,输出中应出现可移出名单的提示."""
    ...
```

(执行时按 check_test_quality.py 现有 main/输出结构实现;核心断言:脚本输出含 "may be removable" 或等价提示。)

- [ ] **Step 2: 实现退出机制**

在脚本覆盖率统计后增加"豁免复核"段:对 UNLIMITED_FILES 与 CONTRACT_COVERED_FILES 中**当前分类覆盖 ≥3 类(缺 0-1 类)**的文件,输出 `INFO: <file> may be removable from <list> (covers N/4 categories)`;对全部豁免文件打印清单与原因,方便下次治理决策。不改变 exit 码语义(不因提示失败),但打印必须显式(治理文本:永远通过的检查要有人读输出)。

- [ ] **Step 3: 运行测试 + 门禁 + Commit**

```bash
uv run pytest tests/unit/scripts/test_check_test_quality_exemptions.py -v
uv run python scripts/check_test_quality.py   # 输出含豁免清单
```

`git add scripts/check_test_quality.py tests/unit/scripts/test_check_test_quality_exemptions.py && git commit -m "feat(scripts): report stale quality-gate exemptions"`

### 批次 B 收尾

- [ ] **Step B1: 批次验证(同批次 A 的 6 门禁 + 全量测试)**

向用户汇报批次 B 结论(RBF 单源化数值等价确认、paper marker 生效用例数)。

---

## 批次 C:P2 文档与流程(7 项)

### Task 8: decision_log 补齐 + pu_workflow_design 蒸馏删除

**Files:**
- Modify: `docs/project_management/decision_log.md`(首行表格插入 5 条)
- Delete: `docs/project_management/pu_workflow_design.md`

- [ ] **Step 1: 补 5 条决策(表头下首行插入,按最新在上的现有惯例)**

| 日期 | 决策 | 理由 | 决策人 |
|---|---|---|---|
| 2026-08-09 | v1.0.0 版本升级(0.1.0.dev0→1.0.0) | roadmap 0.1→0.6 功能全部完成(17 算法、6 门禁、738 测试);从未发布正式版,直接 1.0.0 首版 | shuidisjtu |
| 2026-08-09 | 新增第 6 道质量门禁 check_format.py(ruff check + format --check,CI 与本地单一入口) | 2026-08-09 CI 曾因本地漏跑 `ruff format --check` 失败,软约束需转硬门禁 | shuidisjtu |
| 2026-08-08 | PUSB Table 2 采用严格子集策略:manifest 锁定 6 数据集 sha256/形状/类别计数,fidelity 降级项显式声明 | 复现基准需可审计可复跑,数据漂移检测机制化 | shuidisjtu |
| 2026-08-08 | PUSBKernelClassifier 独立注册(非 LDCE 别名) | official_compatibility 有 0.5·reg 分歧,独立注册保证元数据诚实 | shuidisjtu |
| 2026-08-06 | pu-workflow 通用 skill(开放规范/双目录 SKILL.md + 中文解读指南) | 把论文复现工作流沉淀为可复用流程,双目录由 check_skill_sync 门禁保证一致 | shuidisjtu |

- [ ] **Step 2: 蒸馏 pu_workflow_design.md 决策要点**(设计文档中"完成后蒸馏进 decision_log"的承诺):将其核心决策(通用 skill 定位、开放规范、双目录、门禁四项 + 备选方案)已含于上表第 5 条;其余设计细节(step/checkpoint 结构)已体现在 `.claude/skills/pu-workflow/SKILL.md` 与 `.agents/skills/pu-workflow/SKILL.md` 双份中,无需保留。

- [ ] **Step 3: 删除文件(逐个删除)并清理引用**

```bash
grep -rn "pu_workflow_design" docs/ scripts/ README.md README.zh-CN.md | grep -v architecture_audit.md
```

删除 `docs/project_management/pu_workflow_design.md`(若 grep 还有引用,先修引用再删)。

- [ ] **Step 4: 更新审计报告对应条目 + 验证 + Commit**

`docs/dev/architecture_audit.md` §5 P2-8 条目标记"已治理"。运行 `uv run python scripts/check_doc_links.py`(Task 2 升级后此文件已入检查范围,若报 orphan 需处理——这正是门禁工作的证明)。

`git add docs/project_management/decision_log.md docs/dev/architecture_audit.md && git rm docs/project_management/pu_workflow_design.md && git commit -m "docs: distill workflow design into decision log and remove orphan doc"`

### Task 9: CLAUDE.md 死链修复

**Files:**
- Modify: `E:\Project\pu-learning-toolbox\CLAUDE.md:29`(本地,gitignored,不提交)

- [ ] **Step 1: 修复路径**

`docs/project_structure.md` → `docs/dev/project_structure.md`(29 行)。CLAUDE.md 被 gitignore,仅本地修改,不提交。验证:该路径存在。

### Task 10: project_structure.md 测试树修正

**Files:**
- Modify: `docs/dev/project_structure.md:121-195`

- [ ] **Step 1: 用实际树修正 §3**

```bash
ls -R tests/ | head -60   # 与实际对照
```

修正点:① 补 `unit/scripts/`(test_check_format.py 等 6 门禁脚本测试)与 `unit/workflow_scripts/`;② benchmarks 段补 4 个缺失文件(`test_assigned_preflight.py`、`test_deep_pu_model_selection.py`、`test_pusb_table2_benchmark.py`、`test_pusb_table2_data.py`);③ 删除不存在的空目录 `registry/`(176-177 行);④ 检查 `unit/estimators/` 实际文件与树一致(现缺 test_wconpu.py 等若存在则补)。

- [ ] **Step 2: 验证 + Commit**

```bash
uv run python scripts/check_doc_links.py   # 树修改不影响链接,确认通过
```

`git add docs/dev/project_structure.md && git commit -m "docs: sync test tree with actual tests/ layout"`

### Task 11: dev-workflow skill 状态速查更新

**Files:**
- Modify: `.claude/skills/dev-workflow/SKILL.md:65`(gitignored,不提交)

- [ ] **Step 1: 更新数字**

`705 passed(2026-08-08 PUSB 全量基准后)` → `738 passed(2026-08-09)`,并核对同节其他数字(算法 17、门禁 6 已正确)。该文件 gitignored,仅本地修改。

### Task 12: 死代码清理(6 项)

**Files:**
- Modify: `pu_toolbox/losses/pnu.py`(删 PNULoss 类)、`pu_toolbox/prior/pen_l1.py:94` + `prior/__init__.py:5`(删 PenL1Estimator 别名)、`pu_toolbox/estimators/risk/upu.py:46-49`(删假别名)、`pu_toolbox/core/config.py:14`(删 DEFAULT_RANDOM_SEED)、`pu_toolbox/core/validation.py:120-137`(删 check_positive,若 Task 6 后仍零调用)

- [ ] **Step 1: 删除前引用核查**

```bash
grep -rn "PNULoss\|PenL1Estimator\|_build_linear_basis\|_build_rbf_basis\|_subsample_centers\|DEFAULT_RANDOM_SEED\|check_positive" pu_toolbox/ tests/ docs/ examples/ benchmarks/
```

每项确认:仅定义处 + 预期引用(如 upu 模块内自用)。`set_global_seed` 保留(conftest 真实使用,是有效公共工具,仅从死代码名单降级)。

- [ ] **Step 2: 逐项删除(每项删完跑相关测试)**

删除顺序:① upu 假别名(注意 46-49 行块内模块函数如仍有生产引用则保留函数只删别名);② PenL1Estimator 别名 + prior/__init__ 导出行;③ PNULoss 类;④ DEFAULT_RANDOM_SEED;⑤ check_positive(确认零调用后)。

```bash
uv run pytest tests/ -v -k "upu or pen_l1 or pnu or prior or validation or config"
uv run pytest tests/ -m "not slow" -q   # 全量兜底
```

- [ ] **Step 3: Commit**

`git add pu_toolbox/losses/pnu.py pu_toolbox/prior/pen_l1.py pu_toolbox/prior/__init__.py pu_toolbox/estimators/risk/upu.py pu_toolbox/core/config.py pu_toolbox/core/validation.py && git commit -m "refactor: remove dead code (PNULoss, PenL1Estimator alias, upu fake aliases)"`

### Task 13: sample_weight 语义统一文档化

**Files:**
- Modify: `pu_toolbox/estimators/risk/dist_pu.py`、`pu_toolbox/estimators/risk/lbe.py`(docstring 补忽略声明)、`pu_toolbox/estimators/deep/self_pu.py`(已有 NotImplementedError,补 docstring)

- [ ] **Step 1: 补文档化声明(不改变行为)**

DistPU/LBE 的 `fit` docstring 与参数说明处补:`sample_weight : ignored (present for sklearn compatibility; PU risk estimators do not use instance weights)`。SelfPU docstring 补:`sample_weight : NotImplementedError (deep estimators do not accept instance weights)`。

- [ ] **Step 2: 验证 + Commit**

```bash
uv run pytest tests/ -v -k "dist_pu or lbe or self_pu or bias_aware"
```

`git add pu_toolbox/estimators/risk/dist_pu.py pu_toolbox/estimators/risk/lbe.py pu_toolbox/estimators/deep/self_pu.py && git commit -m "docs: declare sample_weight semantics for silently-ignoring estimators"`

### Task 14: 杂项漂移修复

**Files:**
- Modify: `docs/research/method_cards/nnpu.md:351-365`、`docs/dev/architecture.md:187`、`docs/research/method_cards/LDCE.md:410`、`docs/user/method_selection.md:74-81`

- [ ] **Step 1: 逐处修复**

① nnpu.md §7.1 构造签名补 `class_prior` 与 `device` 参数(与 `nnpu.py:109-132` 对齐);② architecture.md:187 "15" → "17",并在"17 篇论文"处加脚注"(含 KLDCE 核化变体与 PUSBKernel,非严格论文数)";③ LDCE.md:410 Connect-4 形状 `67557×42` 处加交叉说明"(PUSB Table 2 manifest 锁定为 67557×126:不同预处理口径,见 benchmarks/assigned_methods/configs/pusb_table2_datasets.json)";④ method_selection.md §4 DGPU 双族条目改为与 registry 一致(仅 Deep 族),或加注"DGPU 同时见于 Bias-Aware 章节,族归属以 registry(Fam.DEEP_PU)为准"。

- [ ] **Step 2: 验证 + Commit**

```bash
uv run python scripts/check_doc_links.py
grep -rn "15 篇\|15篇" docs/ README.md README.zh-CN.md   # 确认无残留
```

`git add docs/research/method_cards/nnpu.md docs/dev/architecture.md docs/research/method_cards/LDCE.md docs/user/method_selection.md && git commit -m "docs: fix residual doc drift (nnpu signature, method count, shape, family)"`

### 批次 C 收尾

- [ ] **Step C1: 全量验证(6 门禁 + 全量测试)+ 审计报告 §5 全部 14 项标记已治理**

`docs/dev/architecture_audit.md` §5 每项追加完成状态(日期 + 提交号)。

- [ ] **Step C2: 收尾 Commit + PR**

```bash
git add docs/dev/architecture_audit.md
git commit -m "docs: mark architecture audit action items resolved"
git push origin fix/architecture-decay
```

提 PR 到 main(提醒用户开启代理 127.0.0.1:7897)。PR 描述附审计报告链接与三批摘要。

---

## Self-Review

**1. Spec(audit report)覆盖:**
- P0-1 KLDCE 复核 → Task 1 ✅;P0-2 check_doc_links 补洞 → Task 2 ✅;P0-3 math_rendering 锚定 → Task 3 ✅
- P1-4 RBF 单源化 → Task 4 ✅;P1-5 paper marker → Task 5 ✅;P1-6 class_prior 统一 → Task 6 ✅;P1-7 豁免退出机制 → Task 7 ✅
- P2-8 decision_log + 蒸馏 → Task 8 ✅;P2-9 CLAUDE.md 死链 → Task 9 ✅;P2-10 测试树 → Task 10 ✅;P2-11 skill 数字 → Task 11 ✅;P2-12 死代码 → Task 12 ✅;P2-13 sample_weight → Task 13 ✅;P2-14 杂项 → Task 14 ✅

**2. Placeholder scan:** 无 TBD/TODO。Task 2/7 的测试代码标注"执行时按现有结构补齐"(依赖 check_doc_links.py / check_test_quality.py 的 Issue/main 具体结构,Step 1 已给出可运行骨架与断言意图)。

**3. Type consistency:** `check_scalar_in_range(value, low, high, name, *, inclusive)`(validation.py:86)在 Task 6 用法一致;`build_rbf_basis(X, centers, kernel_width)`(basis.py:34)在 Task 4 三处委托签名一致;`_recover_bias_from_kkt` 参数名与现有调用一致(kldce.py:570 附近调用点)。
