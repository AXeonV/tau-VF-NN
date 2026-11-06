# 黏性指进界面监督学习

用高精度二维 Hele–Shaw 时间推进生成轨迹，训练条件 Fourier 神经算子 `(z0, t) → z(α,t)`。固定中心恒流注入 `Q=2π`、表面张力 `σ=0.015`、初始面积 `π`，输出覆盖 `0 ≤ t ≤ 1000`。第一版处理始终由一条光滑闭合曲线表示的界面。

```text
src/traditional/   初始形状、FP64 边界积分、半隐式推进、标签验算
src/nn/            数据增强、条件 CurveFNO、训练和推理
src/nn_ins/        短窗界面更新对照
src/nn_world/      同时刻局部修正、截断检查和数值回退
utils/             界面图和 loss 图
results/           仿真标签、模型和结果
archive/           旧实验与代码快照
```

64 个初始形状来自 8 个主导模态族。每族有 6 条训练、1 条验证和 1 条测试轨迹。初值通过次级模态、振幅和相位变化打破对称性。保存 `t=0`、`t=1000`、均匀长时快照和对数分布的早期快照。

生成标签时逐级提高空间分辨率并收紧步长。采用更细一级的界面作为标签，要求抽查时刻上两级曲线的最大双向几何距离不超过等效半径的 0.35%，压力通量相对误差不超过 `5e-5`，保存帧的界面没有检测到自交。`case_*.npz` 保存通过检查的轨迹，`provisional_*.npz` 保存待加密的轨迹，`summary.json` 记录各案例的分辨率和结果。生成程序支持继续运行。

网络沿等弧长周期坐标处理 `(x,y)`，把查询时间编码到每层，预测剔除整体膨胀的界面变形。训练在线对初值和标签同步旋转、反射和循环重编号。划分以独立初值为单位。

```sh
python -m src.traditional.checks
python -m src.nn.checks
python -m src.traditional.generate
python -m src.nn.train
python -m src.nn.predict
python -m utils.render
```

完整流程可用 `./run.ps1`。模型输出在 `results/supervised/model`，评估含留出初值的几何距离、终点径向峰数和自交检查。旧代码已归档到 `archive/pre_supervised_code.zip`。

## 三档训练对照

对照实验保持数据和 48／8／8 初值划分不变。`nn` 从初值直接查询时刻，`nn_ins` 从当前界面做短窗更新，包含等弧长重采样和表面张力高频阻尼。测试统一从第 0 帧出发，`nn_ins` 另外报告以真实当前帧为输入的单步误差。验证 MSE 使用同一组时间探针。两个模型各做 4000 个优化步。

```powershell
foreach ($T in 5,20,1000) {
    python -m src.nn.train --horizon $T --out "results/supervised/comparison/t$T/nn" --steps 4000 --save-predictions
    python -m src.nn_ins.train --horizon $T --out "results/supervised/comparison/t$T/nn_ins" --steps 4000
}
python -m utils.compare
```

## T=20 界面修正实验

固定已有的 T=20 CurveFNO 和初值划分。先比较逐点修正、沿弧长卷积、空间邻近但弧长分离的界面段、分频带监督和 JEPA 未来几何特征。再固定修正器结构，对比训练输入来自原 FNO、加噪界面或交叉拟合的 FNO。两次训练种子共用相同的数据和粗预测。

```sh
python -m src.nn_world.proposals
python -m src.nn_world.train --stage structure
python -m src.nn_world.train --stage repair
python -m src.nn_world.train --stage unroll
python -m src.nn_world.audit
python -m src.nn_world.verify
python -m utils.render_world
```

复跑第二个修正器种子时，对三个训练阶段加上 `--seed 20261107 --cache results/supervised/world_t20 --out results/supervised/world_t20/seed2`。结果在 `results/supervised/world_t20`。截断阈值只用验证轨迹选定。密集核验帧用于测量首次超限时刻，触发回退后由初始界面重新推进。

## T=100 长时预测对照

沿用 48／8／8 初值划分。结构对照共用一个 Fourier 主干，依次增加任意时间对、跨时刻耦合、初值和近期运动。微调从同一个历史模型出发，对比真实历史、混合历史、截断梯度、自生成三块或六块、变噪声、尖端加权和路径一致性。独立数值推进用于测量初值敏感性，并提供重锚定界面。

```sh
python -m src.nn.train --horizon 100 --steps 4000 --out results/supervised/rollout_t100/curvefno --save-predictions
python -m src.nn_rollout.experiment --stage structure
python -m src.nn_rollout.experiment --stage finetune
python -m src.nn_rollout.refine
python -m src.nn_rollout.sensitivity
python -m src.nn_rollout.anchors --stage prepare
python -m src.nn_rollout.anchors --stage compare
python -m src.nn_rollout.anchors --stage cost
python -m src.nn_rollout.evaluate_events
python -m utils.render_rollout
```

产物在 `results/supervised/rollout_t100`。`summary.json` 分别记录直接查询、自由滚动和每段使用仿真状态。持续新尖端统计要求径向峰延续到后一帧，并按时间和方位匹配，母尖端明确的分裂单独计数。

第二个种子复验了五种结构和三种微调。结构阶段使用 `--only direct pairs block anchor history`，微调阶段使用 `--only teacher self pushforward`。两个阶段均加上 `--seed 20261107 --out results/supervised/rollout_t100/seed2`。
