# Method Card: GEN-PU

Hou et al., IJCAI 2018，DOI 10.24963/ijcai.2018/312；[原论文](https://www.ijcai.org/proceedings/2018/0312.pdf)
式 (3)–(11)。两生成器、三判别器，之后训练合成 PN 分类器；负生成器与 Dn 的目标是同向分离，不能误写成标准 GAN。

## 来源与明确分歧

[作者 Code 页](https://qibinzhao.github.io/codes/)明确链接[源码归档](https://qibinzhao.github.io/assets/publications/IJCAI_2018_HouMing/gen_pu_demo.zip)。
2026-10-05 下载 SHA256 `a0975a3da5979febbd0504d5ee5b2638b4f9074a66e72d45198278933f54e6c0`。
只读文本，无作者程序或归档 pickle 被执行，也不将归档纳入项目。未找到明确许可证，授权待复核。

| 项 | 论文 / 本组件 | 官方归档 |
|---|---|---|
| U 分布 | §2.1 总体边缘；原生 TS | `utils/data_wrapper.py:160-168` 排除已标记 P，形成 OS |
| Du 假样本项 | 式 (3) 用 π、1−π 混合 | `pun_demo_main.py:249-263` 两项未按 π 加权 |
| generator | 论文 minimax，Gn 的 Dn 项反号 | `pun_demo_main.py:271-297` 非饱和 BCE：Gp 对 Dp/Du target=1；Gn 对 Dn=0、Du=1 |
| 最终 PN | 本组件采用目标总体先验加权 risk（工程选择） | `gpn_classifier.py:219-226` 等权 P/N BCE |

因此登记 `official_related` + `benchmark-adapted`，不称代码逐行复现。正式协议需选择目标口径，
本组件不能替负责人决定论文/代码哪个用于数值对照。

## 当前组件与校准

`GenPUClassifier` / `genpu`，二维稠密 MLP、CPU/CUDA；独立重写。
原始 logit 上用 logsigmoid，Gp 最小化标准 GAN 项，Gn 最小化 Du 项并最大化 Dn 项。
D 步生成流停止梯度，G 步冻结三个 D 参数但保留输入梯度。最终 PN 只用新生成流，不读取真值标签。
有限训练不会证明 Nash 平衡、生成质量或分类性能。

总体先验必填且构造/fit 值不得冲突。`os_or_ts="ts"` 仅将 Du 的真实训练 risk 池换成 U∪P，
Dp/Dn 的真实池始终是 P；默认 OS 明确不是论文原生边缘池。生成流与 PN 阶段不随此校准改标签。
台账 / Survey 自动路由默认 TS，`calibration_applied=true`、`calibration_hooked=true`；
直接组件 fit 默认 OS 保留显式消融入口，真实运行视图按 manifest。

无界 generator 输出适配表格特征，不是 demo 的图像 tanh、官方完整训练配方或共享 CNN。
`max_epochs` 为 GAN 轮数，`classifier_epochs` 为 PN 轮数；每轮步数由实际 Du 池大小决定，
5 个 GAN optimizer update 加 1 个 PN update 分别计数。`epoch_callback` 仅在合成 PN 分类器轮末
调用，使用零起始连续编号；GAN 阶段没有可部署分类器，不输出空壳分类器快照。
`checkpoint_stage_="synthetic_pn"`、`checkpoint_epoch_count=classifier_epochs` 用于解释与磁盘预检。
`EpochCheckpointTrainer` 支持各 PN 轮分类器权重的 CPU 回放，不包含 GAN/优化器/RNG 的训练恢复；
哪些 PN 轮进入正式 PA/OA 选择需另行预注册。快照不改变 TS 风险池或两阶段更新预算。
保存全部六模型的可信 pickle；外部未知 pickle 禁止用于此组件验证。正式预算、存储和选模尚未准入。

## 可复核证据

`tests/unit/estimators/test_gen_pu.py`：概率公式 golden、anti-GAN 梯度方向、极端 logit 稳定性、
训练池角色、固定种子、两阶段 optimizer 数、clone/pickle 与先验 fail-loud。
新增 GPU smoke 待独立执行；不借用其它方法的历史 GPU 记录，不进入冻结 pilot。
