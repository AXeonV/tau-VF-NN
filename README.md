# τ 场学习与黏性指进

二维 Hele–Shaw 恒流注入，采用 Darcy 定律、外部压力 Laplace 方程和表面张力边界条件。当前目标是让 NN 求解一个长时椭圆案例：初始轴比 3:1、面积 π、面注入率 2π、σ=0.015、T=1000，无额外微扰。

## 结构

```text
src/traditional/   边界积分、物理推进初值、全时段传统优化
src/nn/            NN 表示、可微物理 loss、训练与验算
utils/             界面状态曲线图、loss 图
results/           模型、数值结果与两张结果图
```

## 单案例 NN

`fixed_model.py` 使用对数时间 Fourier 特征与 4 层、宽度 192 的 MLP，输入时间，输出闭合界面的空间 Fourier 系数（最高 255 阶，共 256 个实系数）。界面可以凹入、分叉和局部回退；本轮保存完整界面历史。保留纯椭圆问题的两条反射对称性。

初始形状、注入面积增长和初始物理速度由构造满足。时间导数和空间导数解析计算；全程 FP64。外部压力由可微边界积分求解，训练使用 512–1024 点，长时阶段使用 1024 点，最终以 2048 点验算。

先用物理推进轨迹初始化网络权重，再以运动学残差优化：`sqrt(1+2t)·(v_NN−v_pressure)` 的弧长加权平方均值，附加微小参数化均匀性项。物理优化阶段没有轨迹拟合项。Adam 后接覆盖线性时间与对数时间的 L-BFGS，另取独立时刻验算。

压力后端接口为 `BoundaryPressure(z, metric, normal, curvature, sigma) → 法向速度`，可替换为其他可微 GPU 算子。

```sh
python -m src.nn.prepare_fixed --out results/nn/ellipse_long_accuracy/initialization.npz
python -m src.nn.fixed_case --initialization results/nn/ellipse_long_accuracy/initialization.npz --out results/nn/ellipse_long_accuracy --hard-initial-velocity
python -m src.nn.refine_fixed --checkpoint results/nn/ellipse_long_accuracy/model.pt --out results/nn/ellipse_long_accuracy --steps 500 --time-points 768 --log-points 512 --quadrature midpoint --early-nodes 512 --early-until 200
python -m src.nn.refine_fixed --checkpoint results/nn/ellipse_long_accuracy/model.pt --out results/nn/ellipse_long_accuracy --steps 300 --lr 1 --time-points 768 --log-points 512 --quadrature midpoint --early-nodes 512 --early-until 200 --append-history
python -m src.nn.audit_fixed --case results/nn/ellipse_long_accuracy
python -m utils.render --case results/nn/ellipse_long_accuracy
python -m src.nn.fixed_checks
```

`model.pt` 用于连续时间推理；`optimized.npz` 保存界面快照；`accuracy.json` 保存独立物理残差、压力加密与几何检查。展示仅生成界面状态曲线 `distribution.png` 和 `loss.png`。

## 多初值算子试验

`model.py / train.py / predict.py` 保留早期 PINO：初始场与 σ → 网格 τ。该试验使用 32² 网格、T=1 和单调星形等时线适配器；长时案例使用上面的闭合界面网络。

```sh
python -m src.nn.train --steps 1000
python -m utils.render_nn --case results/nn/pino_pilot
```

## 传统方案

```sh
python -m src.traditional.evolution
python -m src.traditional.optimize --case results/ellipse_long
python -m src.traditional.checks
python -m utils.render --case results/ellipse_long
```

依赖见 `requirements.txt`。
