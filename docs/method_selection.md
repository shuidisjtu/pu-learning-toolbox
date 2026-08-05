# Method Selection Guide

本指南根据数据场景、假设、硬件、实现状态整理候选算法，供人工选型参考。自动化推荐器见 §4。

算法族总览：

| 算法族 | 方法 |
|---|---|
| Class Prior Estimation | penL1, KM1/KM2, TIcE †, AlphaMax †, ReCPE |
| Classic & Calibration | Elkan-Noto, PU Bagging †, Biased SVM †, Weighted LR † |
| Risk Estimation | uPU, nnPU, PNU, Dist-PU, LDCE, KLDCE, LLSVM |
| Bias-Aware PU | PUSB, LBE |
| Deep PU | Self-PU, InfoMax PU, WConPU, DGPU |

> † 扩展参考方法，不在 v1 版本（15 篇核心论文）范围内。详见文末「扩展参考」章节。

## 1. 按数据场景选择

### Single-training-set

| 方法 | 原因 |
|---|---|
| Elkan-Noto | 简单直观，SCAR baseline |
| PU Bagging † | 鲁棒，不依赖复杂假设（扩展参考） |
| Weighted LR / Biased SVM † | 易部署，可解释性好（扩展参考） |

### Case-control

| 方法 | 原因 |
|---|---|
| uPU | 无偏风险估计，理论清晰 |
| nnPU | 缓解 uPU 负风险，适合深度模型 |
| PNU | 有部分负样本时组合 PN/PU/NU 风险 |
| Dist-PU | 标签分布视角的 PU 风险估计，深度模型 |
| Self-PU | 自训练 + meta reweight + 蒸馏，需 clean validation |
| InfoMax PU | 信息论表征学习 + nnPU，无需 π |
| WConPU | 加权对比学习 + 难负样本挖掘 |

### Selection-biased / SAR

| 方法 | 原因 |
|---|---|
| PUSB | 直接面向 selection bias |
| LBE | 显式估计 labeling bias |
| DGPU | 判别-生成联合建模，支持 SCAR 与 SAR |

## 2. 按标记机制选择

### SCAR（$P(s=1 \mid y=1, x) = c$）

| 方法 | 需要 $\pi$ | 备注 |
|---|---|---|
| Elkan-Noto | 否 | 需估计 $c$ |
| PU Bagging † | 否 | 启发式 baseline（扩展参考） |
| Weighted LR † | 是 | 适合工程部署（扩展参考） |
| uPU / nnPU / PNU | 是 | case-control 下更自然 |
| Dist-PU | 是 | 标签分布视角，深度模型 |
| Self-PU | 是 | 自训练 + 蒸馏，需 clean validation |
| InfoMax PU | 否 | 信息论表征 + nnPU，自动估计 π |
| WConPU | 是 | 加权对比学习 + 难负样本挖掘 |

### SAR（$P(s=1 \mid y=1, x) = c(x)$）

| 方法 | 说明 |
|---|---|
| PUSB | selection bias 基线 |
| LBE | 显式建模 labeling bias |
| DGPU | 判别-生成联合建模，同时支持 SCAR 与 SAR |

## 3. 按数据规模选择

| 数据特点 | 推荐 |
|---|---|
| 小数据、低维 | Elkan-Noto, Weighted LR † |
| 小数据、高维文本 | Biased SVM †, PU Bagging † |
| 中等数据 | PU Bagging †, Biased SVM † |
| 大数据、有 GPU、$\pi$ 已知 | nnPU, Dist-PU, Self-PU, WConPU |
| 大数据、有 GPU、$\pi$ 未知 | InfoMax PU (自动估计 π)，或 TIcE † / AlphaMax † 估计 → 风险估计方法 |
| 大数据、$\pi$ 未知 | TIcE † / AlphaMax † 估计 $\pi$ → 风险估计方法 |
| 怀疑标记有偏 | PUSB, LBE, DGPU |

## 4. 推荐器

`pu_toolbox.advisor.recommend_methods` 实现自动化算法推荐，流程：

```
数据画像 (规模/稀疏性/PU比例) + 用户输入 (scenario/assumption/π)
    → registry 过滤 (trainable_only + scenario + sparse + class_prior + assumption)
    → 七维评分 (assumption匹配 + maturity + source_status + 数据规模 + 训练成本 + GPU + 标记充足度)
    → 按综合分排序
    → 返回 top-k
```

### 基本用法

```python
from pu_toolbox import recommend_methods, recommend_from_profile, profile_pu_data

# 数据驱动（自动 profiling）
result = recommend_methods(X, y_pu, class_prior=0.3, has_gpu=True)

# Profile 驱动（跳过重复 profiling）
profile = profile_pu_data(X, y_pu)
result = recommend_from_profile(profile, scenario="case_control")

# 查看结果
for c in result.candidates:
    print(f"{c.rank}. {c.name} (score={c.score:.1f})")

# 导出
result.to_json()       # JSON 字符串
result.to_markdown()   # Markdown 报告
result.save("recommendations.json")
```

推荐器元数据 schema 见 [`architecture.md`](architecture.md) §6。

## 5. 风险提示

推荐器必须显式提示：

1. SCAR 不成立时，Elkan-Noto、uPU、nnPU 结果可能有偏。
2. $\pi$ 估计错误会显著影响风险估计方法。
3. labeled positive 过少时，深度方法不一定优于简单 baseline。
4. selection bias 明显时，优先尝试 PUSB / LBE。
5. `api_only` 算法仅展示接口，不能用于训练。

## 扩展参考（不在 v1 范围内）

以下方法是 PU 学习领域的经典或实用算法，但不在本工具箱 v1 版本集成的 15 篇论文范围内。此处列出供参考，未来版本可能纳入。

| 方法 | 类型 | 说明 |
|---|---|---|
| TIcE | 类先验估计 | 基于决策树的类先验估计 |
| AlphaMax | 类先验估计 | 基于混合模型的类先验上界估计 |
| PU Bagging | 分类器 | 基于 Bagging 的 PU 启发式方法 |
| Biased SVM | 分类器 | 对正负类使用不同代价的 SVM |
| Weighted LR | 分类器 | 加权逻辑回归 PU 包装 |
