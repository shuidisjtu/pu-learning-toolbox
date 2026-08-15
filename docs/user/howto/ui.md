# 使用图形界面

图形界面基于 Streamlit，作为可选依赖提供，不会增加核心安装体积。

## 安装与启动

```bash
pip install "pu-toolbox[ui]"
pu-toolbox-ui
```

命令会在本机启动 Web 页面并打开浏览器。服务只处理用户在页面中上传的数据；工具箱
本身不会主动把数据发送到外部服务。部署到共享服务器时，文件可见性、认证和网络访问
策略由部署者负责。

## 页面流程

1. 上传特征 CSV 或 4-D NCHW `.npy` 图像数组。
2. 上传单列 `{1, 0}` PU 标签；真实 `{1, 0}` 标签可选。
3. 查看样本、特征、已标正例和未标记样本摘要。
4. 使用自动推荐，或手动选择模型、先验估计方式、CV 和指标。
5. 在高级面板输入固定参数，或打开“比较多组超参数”填写参数网格。
6. 启动分析，查看指标图表、诊断提示和全部调参 trial。
7. 下载 JSON/Markdown 报告、预测 CSV、模型以及调参记录。

CSV 第一行必须是非数字列名，特征必须全部为有限数值。图像模式目前支持
InfoMax PU 和 WConPU，可选择 CNN13、ResNet-18 或 ResNet-50；需同时安装 torch：

```bash
pip install "pu-toolbox[ui,torch]"
```

DGPU 的 `generator` 是 Python 对象协议，不能用 JSON 表单安全构造，因此暂不出现在
UI 的模型下拉框中；仍可通过 Python API 完整配置。

## 参数输入示例

固定参数：

```json
{"loss": "logistic", "max_iter": 2000}
```

参数网格：

```json
{"reg_lambda": [0.001, 0.01, 0.1]}
```

固定参数会应用到每个 trial，同一个参数不能同时出现在固定参数和参数网格中。

下一步：[调整模型与搜索超参数](model_tuning.md)；[指标与精确 API](../reference/api.md)。
