# 黏性指进界面预测

用二维径向 Hele–Shaw 仿真监督空气注入油时的界面演化。中心恒流注入 $Q=2\pi$，表面张力 $\sigma=0.015$，初始面积 $\pi$。网络在等效半径 $R(t)=\sqrt{1+2t}$ 归一化的坐标中预测闭合界面。边界积分与时间推进离线生成监督标签，推理只调用网络。

## 核心模型

| 模型 | 预测方式 |
|---|---|
| `nn_direct` | 时间条件 Fourier 算子，给定初形和时刻直接预测界面。 |
| `nn_block` | 从初形、当前界面和近期法向运动联合预测四帧，在块末反馈。 |
| `nn_geometry` | 逐帧反馈，联合 Fourier 主干和读取曲率、尖端凸度、法向历史及空间近邻的几何分支。 |

`nn_block` 和 `nn_geometry` 共用滚动训练与 Fourier 主干。`nn_geometry` 可用 `--long-rollout --initial-starts` 启用最长 16 帧的训练课程并增加从初形起步的片段。`nn_direct` 和 `nn_geometry` 可用 `--tip-weight` 加入指尖与凹槽的空间监督。

## 数据与目录

| 路径 | 内容 |
|---|---|
| [T=100 轨迹](data/supervised/trajectories_t100_dense_128/summary.json) | 128 个初形，每条 177 帧，训练、验证、测试按初形划分为 112、8、8。 |
| [T=1000 轨迹](data/supervised/trajectories_t1000_128/summary.json) | 同一批初形，每条 265 帧。 |
| [独立测试轨迹](data/supervised/drift_diagnostic_fresh/summary.json) | 8 个独立初形，每条轨迹覆盖 $T=0$ 至 $T=100$。 |
| [界面叠加图](data/img) | 每个长时案例一张图，八个时刻共用物理坐标尺度。 |
| `results/supervised/models` | 三种模型的 $T=100$ 检查点、共享滚动热启动权重，以及 `nn_direct/t1000`。 |
| `results/supervised/runs` | 新训练、评估和绘图的输出。 |

训练时对整段轨迹同步应用连续旋转、反射和周期重编号。原 64/96 初形数据路径保留配置和验算摘要，通过同级正式库读取对应子集。数据由 `src.traditional.generate`、`src.traditional.expand` 和 `src.traditional.expand_long` 构建，最后一阶段按 `--stage prepare`、`--stage run`、`--stage report` 执行。

## 运行

依赖见 [requirements.txt](requirements.txt)。

```sh
python -m src.nn_direct.train
python -m src.nn_block.train --warm results/supervised/models/rollout_warm.pt
python -m src.nn_geometry.train --warm results/supervised/models/rollout_warm.pt --long-rollout --initial-starts

python -m src.predict nn_geometry --case 7
python -m utils.render --case results/supervised/models/nn_geometry
python -m utils.render_dataset
```

长时直接预测可指定对应数据和权重。

```sh
python -m src.predict nn_direct --data data/supervised/trajectories_t1000_128 --model results/supervised/models/nn_direct/t1000/model.pt
```

默认直接模型采用带指尖监督的 128 初形检查点，同目录的 `baseline.pt` 保留原直接模型。几何模型默认使用长滚动课程检查点，`finetuned.pt` 保留此前界面对比所用的坐标微调权重。

架构、训练和诊断的逐项结果与分析见 [实验记录](docs/experiments.md)。
