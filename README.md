# 黏性指进界面监督学习

用高精度二维 Hele–Shaw 时间推进生成轨迹，训练条件 Fourier 神经算子 `(z0, t) → z(α,t)`。固定中心恒流注入 `Q=2π`、表面张力 `σ=0.015`、初始面积 `π`，输出覆盖 `0 ≤ t ≤ 1000`。第一版处理始终由一条光滑闭合曲线表示的界面。

```text
src/traditional/   初始形状、FP64 边界积分、半隐式推进、标签验算
src/nn/            数据增强、条件 CurveFNO、训练和推理
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
