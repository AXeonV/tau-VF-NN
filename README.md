# 黏性指进界面预测

用二维径向 Hele–Shaw 仿真监督空气注入油时的界面演化。中心注入流量为 Q=2π，表面张力为 σ=0.015，初始面积为 π。界面用周期等弧长坐标表示，神经网络预测除去整体膨胀后的形变。数值推进用于离线生成标签，滚动推理从初形开始，只反馈模型自己的界面。

## 三种模型

| 模型 | 架构与意图 |
|---|---|
| `nn_direct` | 时间条件 Fourier 算子。输入初形和查询时刻，一次输出该时刻的界面，用于衡量从初形直接预测分叉的能力。 |
| `nn_block` | Fourier 主干读取初形、当前界面和近期法向运动，联合预测四个时刻。推理只在块末反馈，用于衡量降低反馈频率的作用。 |
| `nn_geometry` | 共享 Fourier 主干，每次推进一帧。几何分支读取两种尺度的曲率、尖端凸度、近期法向运动和窄缝两侧的空间近邻，输出法向位移增量，并与主干联合训练。 |

两个滚动模型使用 Self Forcing。训练时从仿真界面起步，展开模型自己的预测，再对后续输出施加真实轨迹监督。`src.nn_geometry.train` 的 `--no-geometry` 开关用于逐帧 Fourier 消融。三个模型目录只放各自的架构和训练入口。Fourier 层、滚动训练、数据加载和指标共用 `src` 下的模块。`utils/render.py` 绘制损失和界面。

## 数据

- `data/supervised/trajectories_t1000_128` 使用 265 帧长时网格，覆盖 `T=0–1000`。原 64 条长时轨迹沿用已验收标签，其余轨迹从对应的 `T=100` 状态续算。
- `data/supervised/trajectories_t100_dense_128` 使用 177 帧早期加密网格，覆盖 `T=0–100`。两套数据共用 128 个独立初形，按 112／8／8 划分。新增初形在八个模态族内按旋转与反射对齐后的形状距离挑选。
- 两套数据的 `augmentation.json` 定义每个训练初形的 8 个旋转角与 2 种镜像状态。在线训练对同一片段同步做连续随机旋转、反射和循环重编号。验证和测试沿用原初形。
- `data/img` 每个初形一张叠加图，用紫到黄的轮廓和浅色填充标出 `T=0、10、20、50、100、200、500、1000` 的界面。各时刻使用同一物理坐标系，保留界面随时间膨胀的尺度。原 64 条和 96 条数据仍保留在 `data/supervised`，用于复现之前的对照。`results` 保存训练产物与运行日志。

```sh
python -m src.traditional.expand_long --stage prepare
python -m src.traditional.expand_long --stage run
python -m src.traditional.expand_long --stage report
python -m utils.render_dataset
```

生成程序跳过已经验收的轨迹。续算以两档空间分辨率和步长验算，要求最大双向轮廓差不超过 0.35% R、通量相对误差不超过 `5e-5`、保存界面无自交。数据目录中的 `summary.json` 记录逐例验算结果。

## 训练与推理

训练入口默认使用 128 初形的加密数据。`nn_direct` 可通过 `--data data/supervised/trajectories_t1000_128` 使用长时标签，也可加 `--horizon 100` 裁剪。两个滚动模型当前的训练跨度为 `T=100`。

```sh
python -m src.nn_direct.train --save-predictions
python -m src.nn_block.train --warm results/supervised/models/rollout_warm.pt
python -m src.nn_geometry.train --warm results/supervised/models/rollout_warm.pt
```

已训练的三个 `T=100` 检查点保存在 `results/supervised/models`。新训练默认写入 `results/supervised/runs`。用 `--model` 指定其他检查点。下面的命令用原保存时间表重跑已有模型。

```sh
python -m src.predict nn_geometry --case 7 --data data/supervised/trajectories
python -m utils.render --case results/supervised/models/nn_geometry
```
