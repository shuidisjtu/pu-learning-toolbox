# 论文源码资源统计

> 15 篇 PU Learning 论文的官方源码状态与集成依据。更新时间：2026-07-16。

`implementation_status` 语义见 [`architecture.md`](architecture.md) 的“算法注册表”部分；`source_status` 的当前枚举以 `pu_toolbox/core/tags.py` 为准，本文档给出论文源码状态统计。

## 总体统计

| 状态 | 数量 |
|---|---|
| `official_exact` | 8 |
| `official_bundle` / `official_related` | 3 |
| `third_party_only` | 1 |
| `not_found` | 3 |

## 论文源码清单

| # | 方法 | 状态 | URL |
|---|---|---|---|
| 1 | Class-Prior Estimation (penL1) | `official_related` | http://www.mcduplessis.com/index.php/software/ |
| 2 | ReCPE | `official_exact` | https://github.com/a5507203/Rethinking-Class-Prior-Estimation-for-Positive-Unlabeled-Learning |
| 3 | Elkan-Noto | `third_party_only` (native implemented) | https://github.com/pulearn/pulearn |
| 4 | Convex PU / uPU | `official_bundle` (native implemented) | https://github.com/t-sakai-kure/pywsl |
| 5 | nnPU | `official_exact` | https://github.com/kiryor/nnPUlearning |
| 6 | PNU | `official_exact` | https://github.com/t-sakai-kure/pywsl |
| 7 | Centroid (LDCE) | `official_related` | https://gcatnjust.github.io/ChenGong/code/CEGE_PAMI20.rar |
| 8 | LLSVM | `official_exact` | https://gcatnjust.github.io/ChenGong/code/LLSVM_TNNLS19.rar |
| 9 | Dist-PU | `official_exact` | https://github.com/Ray-rui/Dist-PU-Positive-Unlabeled-Learning-from-a-Label-Distribution-Perspective |
| 10 | PUSB | `official_exact` | https://github.com/MasaKat0/PUlearning |
| 11 | LBE | `official_exact` | https://gcatnjust.github.io/ChenGong/code/LBE_TPAMI21.rar |
| 12 | Self-PU | `official_exact` | https://github.com/VITA-Group/Self-PU |
| 13 | InfoMax PU | `not_found` | — |
| 14 | Weighted Contrastive PU | `not_found` | — |
| 15 | DGPU | `not_found` | — |

## 集成方式速查

| 方法 | 方式 |
|---|---|
| uPU | native NumPy（C-DH SLSQP / C-LL L-BFGS / Squared 闭式解，pywsl 算法参考） |
| nnPU | native PyTorch（clean-room，Algorithm 1 参考 kiryor/nnPUlearning） |
| PNU | native NumPy（squared loss 闭式解，pywsl 算法参考） |
| ReCPE | wrapper + base CPE interface |
| PUSB / LBE | native sklearn-compatible clean-room implementation |
| Dist-PU | native PyTorch implementation |
| Self-PU / LLSVM | adapter / wrapper (planned) |
| Centroid (LDCE) ✅ | native NumPy（clean-room，PAMI 2021 论文公式直译） |
| InfoMax / WConPU / DGPU | clean-room，方法卡先行 |
| Elkan-Noto | clean-room（已实现 native，以 pulearn 为算法验证参考） |
| Class-Prior Estimation (penL1) | clean-room（论文公式直译，MATLAB 源码不对应论文） |

## 待复核

1. 许可证：GitHub 标注 MIT 的可优先复用；`.rar` 或压缩包无 license 的默认只作参考，不直接再分发。
2. 旧仓库可能依赖过时 Python/Chainer/MATLAB/CUDA，需容器化复现。
3. 深度方法的预处理、PU split、class prior 设置嵌在训练脚本中，集成时须抽象为统一 dataset protocol。
