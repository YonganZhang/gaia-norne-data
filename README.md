# Gaia Norne forecast data

可公开复用的 Norne 注水决策模拟数据，以及 Gaia 论文经济结果复算附件。

## 数据内容

- `norne-forecast-data.zip`：主数据 8,000 个样本、确认数据 500 个样本；每个样本含 6 个控制阶段 × 4 口注水井的控制量、22 口生产井的油水产量与井底压力、实际注水率和全场累计量。
- `economics/`：原十次选定主方案、基准与固定参考共十二份物理输出；可独立复算平均收益、六十种经济情景、五档注水约束及收益拆解。
- `tables/`：经济敏感性、水限敏感性、收益拆解、十三次修复后反馈运行和九次 Luna 单轮配置验证的汇总。三类 Agent 运行单独报告。
- `data_manifest.json`：每个输入文件的 SHA-256、数组形状、井顺序与来源版本。

主数据归档作为 GitHub Release 附件分发，下载后放在仓库根目录。

## 快速使用

```bash
python -m pip install -r requirements.txt
python verify_data.py --archive norne-forecast-data.zip
python economics/reproduce_economics.py
python -m zipfile -e norne-forecast-data.zip .
```

数据不需要模型账号、API key、GPU 或油藏模拟器即可读取。最后一个命令解压后，样本位于 `data/norne_fc/shards/00000.npz`；用 `numpy.load(..., allow_pickle=False)` 打开。详细字段见 DATA_DICTIONARY.md。

## 适用范围

这是固定地质模型下的模拟预测数据，不是现场实测数据；主要变化是注水控制。预测生产井按上限自由生产，地质参数没有随机化。部分井在预测期关闭，F-4H 的实际注水可能受井底压力上限截断。不能直接声称跨油田、跨地质稳健性或现场经济收益。

确认集与主数据的第零号样本都是相同基准控制，不能将两者都视为独立随机样本；对主数据旧测试集曾进行模型选择，确认集也已经用于论文评估，不再是未使用的测试集。

经济附件验证的是已有模拟轨迹的计价，不复现 Agent 的候选生成、代理模型训练或全物理模拟执行。十三次反馈验证与九次 Luna 验证均为独立补充，不能混进原十次主队列，也不能当成已完成九十次角色对比。

## 来源与许可

数据源于 [OPM Norne benchmark](https://github.com/OPM/opm-data/tree/master/norne)。数据许可与署名见 DATA_LICENSE.md 和 LICENSES/；代码为 MIT。

论文工作标题：*Gaia: a layered agent team for reservoir decision making, with authority allocated by measured competence — a water-injection scheduling case study on the Norne field*。这是与稿件配套的数据发布，论文尚未在此声明为已发表。引用信息见 CITATION.cff。

协作入口见 CONTRIBUTING.md；本仓库不含私人项目历史、模型对话或账号配置。
