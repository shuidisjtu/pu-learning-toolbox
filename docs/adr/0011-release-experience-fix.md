# ADR-0011:发布体验修复

- 状态:已接受(2026-08-10)
- 触发复审:用户以真实安装实测暴露新体验问题时

## 背景

用户以真实 PyPI 安装 + GPU 实测发现问题:GPU 机器默认吃 CPU;WConPU
默认 800 epoch 无早停需 ~1.5h;skill 环节脚本在包外导致 pip 用户无法
使用;发布的 wheel 中 `import pu_toolbox` 返回错误版本号。

## 决策

1. 深度估计器默认 `device=None` 自动检测 CUDA(共享 `core/device.py`
   单源;CLI `--device` 默认 auto)。
2. profile/recommend/sensitivity 收为 CLI 子命令(scripts 改兼容包装)。
3. CLI `--max-epochs` 透传;WConPU 默认 max_epochs 800→100。
4. `__version__` 漂移修复:`pu_toolbox/__init__.py` 硬编码版本改为与
   pyproject 同步;check_project_metadata 门禁新增 `__version__` 与
   `project.version` 一致性检查(负向验证通过)。

## 备选方案

- **仅文档提示**:默认路径体验是产品面,文档救不了默认行为。否决。

## 后果

- GPU 机器默认吃 GPU;pip 用户可用 skill 环节脚本;版本漂移有门禁拦截。
- device 解析逻辑单源化,后续新增后端复用同一入口。
