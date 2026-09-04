# ADR-0017:API 参考定位与信息分区

- 状态:已接受
- 触发复审:文档导航再次混乱;或 mkdocstrings+Griffe 自动生成评估(ADR-0013 登记项)有结论需调整分工时;或公共符号 > 100 或 api.md > 2000 行时(转换到决策 7 的拆分结构)

## 背景

ADR-0008 的触发复审条件("文档导航混乱")已出现:使用 API 时需在
`user/reference/api.md` 与 howto/method card 之间来回切换。根因是 api.md
定位为"只给精确契约,参数含义与使用示例在对应 howto 中",而分类器参数
语义分散在 method card 的论文对照表与总览表摘要、部分 howto 的职责说明中,
信息没有单一权威位置。

## 决策

1. **受众分层显式化**(目录结构不动,ADR-0008 原文不改):
   `user/` = 库使用者(含第三方仓库开发者),按**任务类型**分层
   (concepts 原理 / howto 操作 / reference 契约);
   `dev/` = 仓库维护者;`research/` = 论文研究内容。声明写入
   `docs/README.md` 与 `docs/user/README.md`。
2. **api.md 定位改为"公共 API 权威契约"**:每个公共入口自含签名、参数表、
   返回结构与最小示例;参数契约统一在 api.md,method card 里只留论文研究内容、
   实现边界与复现状态,并加一行指向 api.md(单向链接,反向只在 api.md 内指向
   "深入/教程")。
3. **防漂移门禁**:新增 `scripts/check_api_docs.py`——静态提取包根 `__all__`
   与注册表 `AlgorithmMetadata(name=...)`,校验全部出现在 api.md;新增公共
   符号未登记即失败(本门禁在本决策落地时即抓到历史漂移:PUUncertaintyReport)。
4. **版本与变更元信息**:api.md 头部标注与 `pyproject.toml` 同步的版本,
   变更记录沿用 `docs/dev/release_process.md`;不单独建 CHANGELOG(YAGNI)。
5. **错误信息统一**:api.md 新增"错误与异常"章(异常族谱见
   `pu_toolbox.core.exceptions.PULearningError` 家族、各 API 场景→异常→建议
   表与问题码索引)。异常消息文本真相源在代码(ADR-0013 原则不变)。
   代码侧异常类文件分裂(`workflows/_errors.py` 的 `PipelineError` 等 vs
   `core/exceptions.py`)随本变更收敛:定义统一回 `core/exceptions.py`
   (原 `workflows/_errors.py` 已删除;`workflows.PipelineError` 与
   `progress.RunCancelledError` 公共路径经再导出保留)。
6. **后续项承接 ADR-0013**:mkdocstrings+Griffe 自动生成 API 参考的可行性
   评估继续;其前置是 docstring 规范化——本决策落地时发现约 10 个分类器
   具名类 docstring 缺 `Parameters` 段,纳入后续工作。
7. **API 参考拆分预案**(触发条件:公共符号 > 100 或 api.md > 2000 行,
   任一先到):拆为 `docs/user/reference/api/` 目录——
   `index.md` 总索引(即现状注册表索引表)、`classifiers.md` 分类器
   (占比最大,算法扩展到 58+ 时的首要增长点)、`tools.md`(PUPipeline、
   数据生成、评估、漂移、其他函数)、`errors.md`(错误与异常)。配套改动:
   `check_api_docs.py` 扫目录(递归 glob)、method card 与 examples 指针
   链接改到对应子文件、docs/README 索引同步。**现在不拆**(YAGNI):
   单一文件 + 分组目录(注册表索引)在 55 符号/1090 行规模仍可管理,
   且预留拆法后不需在 mkdocstrings 自动生成的过渡期做第二次结构变更。

## 备选方案

- **参数契约定在 docstring / method card,api.md 只作索引**:切换成本依旧,
  与 scikit-learn/Hugging Face/FastAPI 的"API 参考自含参数语义"主流不一致。否决。
- **api.md 与使用方法合并为一篇**:检索目标(快捷查参)与学习路径(完整教程)
  混淆,文档整体膨胀。否决。
- **错误信息单独成文**:需求规模尚小,api.md 内独立章承载。否决。

## 后果

- api.md 成为单一参数权威(约 1000 行);method card 增加一行指引;
  质量门禁由 7 项增至 8 项(检查清单见 dev-workflow 与 CI 已同步)。
- 后续 api.md 与代码同步由 `check_api_docs.py` 自动盯防,人工 Review 聚焦语义。
- mkdocstrings 自动生成若采用,将替换 api.md 参数表为 docstring 反射
  (api.md 保留自写概要与错误章),届时本 ADR 触发复审。
