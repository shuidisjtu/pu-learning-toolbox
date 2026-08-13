# ADR-0013:docs 目录合并

- 状态:已接受
- 触发复审:文档导航再混乱,或 docs/ 新增受众层时

## 背景

docs/project_management/ 仅 3 份文档(281 行),读者(开发者/维护者)与
docs/dev/ 受众重合;cli_design.md 参数表已静默漂移(缺 5 个后增参数)
且与 docs/user/howto/cli.md 重复;ADR-0008 的 user/dev/research/
project_management 四层分层在实际演进中收敛为三层。

## 决策

1. **project_management/ 解散**:process_checklist.md、release_process.md
   git mv 平铺至 docs/dev/;cli_design.md 删除,独特决策并入 ADR-0010。
2. **进度清单历史压缩**:每 Phase 一行摘要,逐条明细归 git log;未完成项
   与发布状态节保留。
3. **文档原则确立**:代码 docstring/注释是行为细节的真相源,文档只记
   决策与理由,不重述「是什么」(cli_design 参数表漂移即反例)。

## 备选方案

- **docs/dev/process/ 子目录**:重造受众子层,与「目录即受众」的目标矛盾。否决。
- **原样平铺**:保留历史流水与漂移的参数表,违背压缩目标。否决。
- **三份全部并入 CONTRIBUTING**:流程清单混入贡献指南,受众混淆。否决。

## 后果

- ADR-0008 决策 #2 加修订注,原文不改。
- 全库 12 处引用重定向至 docs/dev/ 路径;docs/README 索引三节合并。
- 后续工作:lychee(链接/锚点检查)与 mkdocstrings+Griffe(API 参考自动
  生成)在本项目的兼容性配置与使用评估;评估结论与采用决定另立 ADR。
