# ADR-0010:CLI/auto/skill 工作流

- 状态:已接受
- 触发复审:CLI 命令数膨胀,或 agent 生态变化时

## 背景

CLI 需零新增依赖、可扩展;agent 工作流(pu-workflow skill)需可复用、
可分发;默认 run 路径曾实测 ~30s。

## 决策

1. **CLI 采用 argparse 单命令薄封装**(run/list-methods/list-priors/
   make-demo-data/…):所有逻辑在 PUPipeline,CLI 只做参数解析/CSV IO/
   错误映射;辅助命令从 registry 实时读取。
2. **run 默认 auto 模式引入推荐器训练成本维度**(第 7 维)+ LLSVM 收敛
   早停(默认开):默认 run 实测 30s → 2s。
3. **Deep PU 接入 Pipeline/CLI**:两级参数 `architecture`(mlp/cnn)+
   `backbone`(cnn13/resnet18/resnet50);`--data` 接受 .npy 4D NCHW 图像;
   DGPU/Self-PU 不接入(无单骨架插拔概念)。
4. **pu-workflow skill 通用化**:开放规范/双目录 SKILL.md + 中文解读指南,
   check_skill_sync 门禁保证一致。
5. **skill install 子命令**:SKILL.md 随 wheel 分发,一键安装到
   `~/.claude/skills/` 与 `~/.agents/skills/`(默认跳过已存在,--force 覆盖)。

## 备选方案

- **click/typer**:引入新依赖,违反零依赖原则。否决。
- **每算法硬编码 CLI 命令**:注册即感知更省维护。否决。

## 后果

- 新算法注册后 CLI 自动可见;skill 三份副本风险由门禁与随包分发控制。
- 默认路径耗时是产品面指标,后续改动受此约束。
