# ADR-0005:复现可信度分级

- 状态:已接受
- 触发复审:新论文接入策略选择,或官方数据/源码状态变化时

## 背景

论文复现可信度差异大:官方源码可获得性与授权状态不同,官方数据可能
需单独授权;项目要求「不假装可用、不虚报复现」。

## 决策

1. **source_status 分级**:official_exact > official_bundle > official_related >
   third_party_only > not_found;有官方源码的论文优先 adapter,无源码走
   clean-room 实现。
2. **claim-safe 原则**:benchmark 产物默认 `paper_claim=false`;配置未锁定的
   维度(如论文未公开类别分组)必须声明「暂定协议」,不得宣称论文数值复现。
3. **PUSB Table 2 严格子集策略**:manifest 锁定 6 数据集 sha256/形状/类别
   计数,fidelity 降级项显式声明,可审计可复跑。
4. **provenance 锁**:磁盘配置 == resolved_config.json == manifest
   config_sha256 三重硬锁,锁测试强制执行;因此 runtime.resume_required 等
   字段不可无代价删除(保留 + 文档化「记录意图、未强制」)。
5. **PUSBKernelClassifier 独立注册**(非 LDCE 别名):与官方实现有 0.5·reg
   分歧,独立注册保证元数据诚实。

## 备选方案

- **全部 clean-room**:有官方源码时浪费最高可信度来源。否决。
- **宣称官方复现但无 provenance 锁**:数据/源码漂移不可检测,违背诚实
  记录。否决。

## 后果

- benchmark 产物可审计可复跑;配置字段变更需显式决策(锁测试会拦截)。
- 官方数据/历史环境全量运行仍依赖执行方提供(非工具箱缺口)。
