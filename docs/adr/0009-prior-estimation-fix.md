# ADR-0009:类先验估计修复与 auto 默认切换

- 状态:已接受(2026-08-10)
- 触发复审:新估计器接入,或 auto 默认再评估时

## 背景

v1.2.0 真实验证发现:常规 SCAR 数据上全部类先验估计器系统性低估
(recpe 坍缩至 0.036 vs 真值 0.5),级联导致 auto 模式 UPU 全判负
recall=0。根因:固定 σ 尺度失配、max-MMD 带宽偏选宽、OOF 折内类比例
偏移、1% 分位数 + 未校准概率坍缩。旧测试宽区间 0.25–0.75 恰好掩盖
27% 偏差。

## 决策

1. pen_l1 默认 `sigma=None` 数据自适应(0.6×标准化数据中位 pairwise
   距离,显式 σ 兼容)。
2. KM 默认 `width_selection="relative"`(0.1×中位距离;作者 mmd_grid 保留
   可选)。
3. ElkanNoto OOF 折内 sqrt 类平衡 sample_weight(几何中点插值)。
4. ReCPE 默认 base 换官方风格 KM2 + `_DensityRatioCPE` 分位数 0.01→0.25;
   边界声明:全部 base 变体仍固有低估,修复目标仅为不坍缩。
5. **auto 默认估计器 recpe → pen_l1**(跨分离度回归入带后切换;recpe 保留
   用于 irreducibility 失效场景)——取代此前的 recpe 默认。
6. prior 测试补数值准确性断言(math golden + 跨分离度/先验护栏 + 集成
   auto 带护栏)。

## 备选方案

- **仅调参、不换默认**:recpe 坍缩场景仍复发。否决。
- **全量替换 recpe**:irreducibility 失效场景下 recpe 仍有价值。否决。

## 后果

- auto 路径默认可靠;测试从「宽区间」升级为数值断言。
- 全 base 变体固有低估的边界在文档中显式声明,不假装消除。
