# ADR-0006:SAR 中长期定位

- 状态:已接受
- 触发复审:SCAR 方法覆盖饱和,或出现更强 SAR 工具时

## 背景

通用 PU 工具多数只支持 SCAR 假设;SAR/Instance-Dependent PU 是工具箱
可能的差异化方向,但实现成本高。

## 决策

- SAR/Instance-Dependent PU 为中长期差异化重点:selection-bias 数据模拟器
  (常数/线性/非线性 propensity)、LBE/PUSB bias-aware 估计器、SCAR vs SAR
  对比 benchmark(3 mechanisms × 10 seeds)已落地。

## 备选方案

- **专注 SCAR**:无差异化,与通用工具同质。否决。
- **立即全面 SAR**:实现成本高,且前置依赖(类先验、风险估计)未就绪时
  不可行。否决。

## 后果

- 后续阶段向 SAR 深度方法倾斜;SCAR 仍是基础面,SCAR/SAR 识别边界在
  文档与诊断报告中显式声明(非识别性筛查不伪装成识别)。
