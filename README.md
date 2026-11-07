# 黏性指进界面预测

用二维径向 Hele–Shaw 仿真监督空气注入油时的界面演化。中心注入流量为 (Q=2pi)，表面张力为 (sigma=0.015)，初始面积为 (pi)。界面用周期等弧长坐标表示，神经网络预测除去整体膨胀后的形变。数值推进用于离线生成标签，滚动推理从初形开始，只反馈模型自己的界面。

## 三种模型

| 模型 | 架构与意图 |
|---|---|
| `nn_direct` | 时间条件 Fourier 算子。输入初形和查询时刻，一次输出该时刻的界面，用于衡量从初形直接预测分叉的能力。 |
| `nn_block` | Fourier 主干读取初形、当前界面和近期法向运动，联合预测四个时刻。推理只在块末反馈，用于衡量降低反馈频率的作用。 |
| `nn_geometry` | 共享 Fourier 主干，每次推进一帧。几何分支读取两种尺度的曲率、尖端凸度、近期法向运动和窄缝两侧的空间近邻，输出法向位移增量，并与主干联合训练。 |

两个滚动模型使用 Self Forcing。训练时从仿真界面起步，展开模型自己的预测，再对后续输出施加真实轨迹监督。`src.nn_geometry.train` 的 `--no-geometry` 开关用于逐帧 Fourier 消融。三个模型目录只放各自的架构和训练入口。Fourier 层、滚动训练、数据加载和指标共用 `src` 下的模块。`utils/render.py` 绘制损失和界面。

## 数据

- `results/supervised/trajectories` 保存原 64 条 (T=1000) 轨迹，初值划分为 48／8／8。
- `results/supervised/trajectories_t100_dense` 保存 96 条 (T=100) 轨迹，每条 177 帧。保留原验证和测试初值。在原时间表上，给 (0<t<10) 增补间隔 0.25 的快照，给 (10le tle40) 增补间隔 0.5 的快照。八个模态族各新增四个训练初形，划分为 80／8／8。
- 加密旧初形时，与原标签在共有时刻验算。新增初形用两档空间分辨率和步长验算。接受条件为最大双向轮廓差不超过 (0.35\%R)、通量相对误差不超过 `5e-5`，且保存界面无自交。

```sh
python -m src.traditional.expand
```

生成程序跳过已经验收的轨迹。数据目录中的 `summary.json` 记录进度和逐例验算结果。

## 训练与推理

训练入口默认使用加密数据。原数据可通过 `--data results/supervised/trajectories` 指定。`nn_direct` 可用 `--horizon 100` 裁剪原长时标签，两个滚动模型的当前训练跨度为 (T=100)。

```sh
python -m src.nn_direct.train --save-predictions
python -m src.nn_block.train --warm results/supervised/models/rollout_warm.pt
python -m src.nn_geometry.train --warm results/supervised/models/rollout_warm.pt
```

已训练的三个 (T=100) 检查点保存在 `results/supervised/models`。新训练默认写入 `results/supervised/runs`。用 `--model` 指定其他检查点。下面的命令用原保存时间表重跑已有模型。

```sh
python -m src.predict nn_geometry --case 7 --data results/supervised/trajectories
python -m utils.render --case results/supervised/models/nn_geometry
```
