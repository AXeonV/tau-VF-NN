# 黏性指进界面预测

用二维径向 Hele–Shaw 仿真监督空气注入油时的界面演化。中心恒流注入 $Q=2\pi$，表面张力 $\sigma=0.015$，初始面积 $\pi$。界面沿周期等弧长坐标采样，网络在等效半径 $R(t)=\sqrt{1+2t}$ 归一化后的坐标中预测形变。边界积分与时间推进用于离线生成标签和诊断，部署时从初形自由滚动。

## 模型

| 模型 | 架构与用途 |
|---|---|
| `nn_direct` | 时间条件 Fourier 算子，从初形直接查询任意保存时刻。 |
| `nn_block` | Fourier 主干结合初形、当前界面和近期法向运动，联合预测四帧，在块末反馈。 |
| `nn_geometry` | 逐帧 Fourier 推进，联合训练界面几何分支。分支读取多尺度曲率、尖端凸度、法向历史与空间近邻，输出法向位移增量。 |

滚动模型采用 Self Forcing。`nn_geometry` 支持 `--no-geometry` 消融、`--long-rollout` 的 16 帧课程，以及在课程中提高初形起步占比的 `--initial-starts`。物理标签和物理输入对照使用独立的训练入口，沿用同一主干和测试集。

## 数据和结果

| 路径 | 用途 |
|---|---|
| [长时轨迹](data/supervised/trajectories_t1000_128/summary.json) | 128 个初形，每条 265 帧，覆盖 `T=0–1000`。 |
| [早期加密轨迹](data/supervised/trajectories_t100_dense_128/summary.json) | 同一批初形，每条 177 帧，覆盖 `T=0–100`。训练／验证／测试为 112／8／8，在线同步旋转、反射和循环重编号。 |
| [后段分叉窗口](data/supervised/late_tip_density/summary.json) | 间隔 0.5 的 42 个训练窗口和 14 个留出窗口，用于粗细标签对照。 |
| [独立诊断初形](data/supervised/drift_diagnostic_fresh/summary.json) | 8 个新初形，另保存正确和走偏状态的同状态数值续算，用于误差来源诊断。 |
| [法向速度标注](data/supervised/normal_velocity_labels/summary.json) | 252 个训练状态和 48 个留出状态。每个状态含有符号法向速度及 0.05、0.5 时距的数值位置标签。 |
| [界面叠加图](data/img) | 每个初形一张图，八个时刻共用物理坐标尺度。 |

历史 64 和 96 初形的数据用于复跑旧对照。`results/supervised/models` 保存基线与预热检查点，`results/supervised/runs` 保存训练和诊断结果。实验记录中的轮廓误差使用双向最大轮廓距离除以等效半径 $R$，再按形状汇总。分叉匹配用持续径向尖端及其时刻、角位置作为代理指标。

早期选样与四个加密先导窗口保留汇总，正式数据见上表。96 初形的同种子几何检查点共用 `results/supervised/runs/dense_t100/dense/nn_geometry/model.pt`。

## 运行

依赖见 [requirements.txt](requirements.txt)。默认训练使用 128 初形早期加密集。直接模型可用 `--data data/supervised/trajectories_t1000_128` 训练长时版本。

```sh
python -m src.nn_direct.train --save-predictions
python -m src.nn_block.train --warm results/supervised/models/rollout_warm.pt
python -m src.nn_geometry.train --warm results/supervised/models/rollout_warm.pt
python -m src.predict nn_geometry --case 7 --data data/supervised/trajectories
python -m utils.render --case results/supervised/models/nn_geometry
```

带物理输入的实验检查点也可通过 `src.predict nn_geometry --model` 加载。复跑法向标注与融合对照使用以下入口，实验产物分别写到各自的目录。

```sh
python -m src.traditional.normal_velocity_labels
python -m src.nn_geometry.train_velocity_labels --arm velocity --seed 20261106 --out results/supervised/runs/velocity_example
python -m src.nn_geometry.train_physics_inputs --variant fusion_trunk --seed 20261106 --out results/supervised/runs/fusion_example
```

长时数据生成使用 `src.traditional.expand_long` 的 `prepare`、`run`、`report` 三个阶段，界面图由 `utils.render_dataset` 生成。每个数据集的验算摘要记录分辨率差、通量误差和自交检查。

## 实验沿革与架构对照

文中的轮廓误差以该时刻的等效半径 $R$ 归一化。同一条目内的箭头表示对应对照。不同时间跨度、推理时间表、留出初形和分叉检测规则的数字分开解读。

### 从物理求解到轨迹监督

- **短时椭圆的全时段界面优化**。纯 3:1 椭圆在 $T=20$ 的全时段物理目标从 $9.56\times10^{-4}$ 降到 $8.25\times10^{-6}$，未认证全局最优。数值热启动后同时优化空间 Fourier 与时间样条系数可以改善短时解的残差。终点几何仍需单独验算。历史记录在提交 `f2a61ad` 的 `results/ellipse_short`。
- **长时椭圆的闭合界面优化**。$T=1000$ 的目标从 $2.08\times10^{-4}$ 降到 $5.95\times10^{-5}$，最低法向几何速度仍为 $-0.00475$。局部回退使首次到达场丢失后续占据历史，长时实验改为保存带辅助状态的完整闭合界面。历史记录在 `f2a61ad` 的 `results/ellipse_long`。
- **初值与优化器对照**。$T=5$ 粗推进约 0.025 秒，终点相对细推进差 0.0164%。接全时段优化约 8.70 秒，终点差 0.0170%。$T=20$ 粗推进与粗推进热启动优化的终点差分别为 1.07% 和 2.99%，后者独立速度残差反而更低。膨胀形状初值和逐段扩大时域两组均未改善终点。降低当前离散残差不足以保证更接近数值推进轨迹，因此保留时间推进作为监督标签来源。历史记录在 `f1996f8` 的 `results/initialization_benchmark`。
- **无轨迹标签的短时 PINO**。$T=1$、$32^2$ 网格试验用 8,000 个初形，留出物理损失从 0.01453 降到 0.00176。验证范围是单调星形短时域，无法外推到带回退和细指的长时演化。历史记录在 `f2a61ad` 的 `results/nn/pino_pilot`。
- **单案例连续时间界面 NN**。$T=1000$ 纯椭圆从数值轨迹初始化后，2048 点审计的独立物理残差 RMS 约 0.00389，846 个检查时刻未检出自交。Fourier 系数网络能表达已知单案例的复杂长时界面，这个试验没有测试新初形的预测。历史记录在 `f2a61ad` 的 `results/nn/ellipse_long_accuracy`。

### 架构对照

- **条件 Fourier 直接查询**。早期同一数据划分、4,000 步下，$T=5/20/1000$ 终点误差为 10.94%／18.89%／40.28%。从初形和时间直接查询避开逐步反馈，在短时更有竞争力。长跨度的终点误差和自交仍高。历史对照在 `c4cf2f9` 的 `results/supervised/comparison/summary.json`。
- **第一版短窗更新 `nn_ins`**。同三档终点误差为 130.91%／196.01%／82.80%，每档均无逐例胜出。初版短窗更新即使包含重采样和高频阻尼，仍未能在自由反馈下维持正确几何。三档对照分别在训练时域内评估。历史对照在 `c4cf2f9` 的 `results/supervised/comparison/summary.json`。

以下 $T=20$ 修补器从同一直接网络预测出发，终点基线为 18.89%。历史结果在 `c4cf2f9` 的 `results/supervised/world_t20` 及其 `seed2` 子目录。

- **逐点同帧修补**。两个种子终点约 18.97% 和 18.97%。逐点输入未提供足够的邻域关系。
- **沿弧长局部卷积**。终点为 18.30% 和 17.97%。局部弧段的形状信息带来小幅收益。
- **空间近邻修补**。终点为 18.06% 和 17.91%。空间靠近而沿界面相隔的弧段对窄缝修补有用，收益局限在该短时设置。
- **频带修补**。单轮终点为 18.14% 和 18.19%，重复两轮后为 17.44% 和 17.84%，重复三轮为 17.59% 和 18.25%。低频轮廓与高频局部形变可分开处理，但修补次数增加后收益没有单调提升。
- **JEPA 未来几何特征**。终点为 18.08% 和 18.18%，没有稳定优于近邻或频带修补。该设置没有显示潜空间预测带来额外收益。
- **修补器端到端展开**。终点为 18.24% 和 18.12%，未超过相同种子的两轮重复修补。展开训练在这批状态上没有显著拓宽修补能力。

以下 $T=100$ 结构对照的指标均为自由滚动终点误差。两个种子复验了前五项。历史结果在 `c4cf2f9` 的 `results/supervised/rollout_t100/summary.json` 与 `seed2/summary.json`。

- **直接模型反复反馈**。终点为 46.02% 和 45.58%，两种子均有 8/8 自交案例。直接模型训练时缺少这种反馈输入。
- **任意时间对更新**。终点为 32.63% 和 33.76%。显式学习局部时间映射改善了反馈，但留出终点仍偏离较大。
- **联合预测四帧**。终点为 32.41% 和 33.99%，和独立时间对接近。减少反馈次数本身不足以稳定分叉。
- **四帧加初形锚定**。终点为 34.48% 和 31.46%，两个种子的收益方向不一致。初形上下文提供全局条件，单独加入仍难矫正中途轨迹。
- **锚定加近期法向运动**。终点降至 28.12% 和 30.32%。若每次给真值历史则为 6.00% 和 6.67%。近期运动有用，真值与自由反馈之间的差距说明闭环误差累积仍是主要问题。
- **历史输入分频带**。单种子终点 28.05%，和未分频的 28.12% 接近。当前频带参数化没有提供明显增益。
- **四帧同块修补与流式细化**。单种子的自由终点为 28.56% 和 30.88%，流式版本单轮为 29.41%。新增细化过程未超过该轮 Self Forcing 的 27.05%。历史记录在同提交的 `results/supervised/rollout_t100/refinement.json`。
- **逐帧几何分支 `nn_geometry`**。128 初形、$T=100$ 同预算下，直接／四帧／几何模型的终点误差为 26.36%／24.34%／21.77%。直接模型在 $T=1000$ 的误差为 42.44%。曲率、尖端凸度、法向运动和窄缝近邻有助于当前滚动模型，但这组结果同时改变了网络和更新方式。逐帧 Fourier 消融有独立入口，现有留出结果不可把全部收益归因于某个几何输入。[128 初形模型对照](results/supervised/runs/expanded_128)

### 物理输出与输入试验

- **法向速度残差头**。800 步两种子对照中，新增速度头的半步误差与基线近乎相同，走偏状态 4 单位自由误差仍约 1.2%。新增头输出幅度很小。单加残差头没有改变主要演化路径。[架构探针](results/supervised/runs/drift_diagnostic/architecture_probe.json)
- **等容量几何头**。速度 RMSE 从 0.12029 到 0.11998，整轨终点从 22.44% 到 22.40%。即使将新增物理通道置零，等容量控制头也带来同量级变化，必须用它隔离增参效应。[物理输入对照](results/supervised/runs/physics_inputs/comparison.json)
- **三处空间近邻输入**。相对等容量控制，速度 RMSE 0.11998→0.11988，走偏分叉匹配 15→16/37。近邻方向和对侧法向提供少量局部信息，尚未带来显著长时改善。[物理输入对照](results/supervised/runs/physics_inputs/comparison.json)
- **双尺度精确历史通道**。相对等容量控制，速度 RMSE 0.11998→0.11985，整轨终点 22.40%→22.41%。附加历史信号在此结构中只起微弱作用。[物理输入对照](results/supervised/runs/physics_inputs/comparison.json)
- **邻域与历史同时加入**。相对等容量控制，速度 RMSE 0.11998→0.11974，走偏 4 单位自由误差 1.1645%→1.1614%，整轨误差基本持平。置零新增通道引起的输出位移远小于正常预测位移，侧路学习到的影响很弱。[结果](results/supervised/runs/physics_inputs/comparison.json) [通道用量](results/supervised/runs/physics_inputs/feature_usage.json)
- **原历史通道直接替换**。将现有历史特征改成精确的对数半径时间尺度后，速度 RMSE 升至 0.12590，整轨终点为 23.33%，自交由基线的 3 增至 5。直接改写暖启动模型依赖的输入尺度改变了主干所见的分布。这种替换没有优于原特征。[物理输入对照](results/supervised/runs/physics_inputs/comparison.json)

三组多近邻融合共用 252 条速度标注、同样的预热权重和 1,200 步预算。两个种子的速度监督基线在分叉扇区 RMSE 为 0.11989，走偏后 4 个时间单位轮廓误差为 1.167%，从初形滚到 $T=100$ 为 22.442%。新增投影零初始化，初始预测逐位一致。[融合对照](results/supervised/runs/physics_fusion/comparison.json)

- **多近邻进入 Fourier 主干**。扇区 RMSE 为 0.11971，走偏后 4 单位误差为 1.166%，$T=100$ 为 22.425%。近邻信息经过全局 Fourier 层，但本轮改变量很小，长时收益随种子变化。[融合对照](results/supervised/runs/physics_fusion/comparison.json)
- **多近邻进入主几何分支**。扇区 RMSE 为 0.11962，走偏后 4 单位误差为 1.164%，$T=100$ 为 22.442%。几何输入直接参与法向位移计算，局部误差略降，终点基本持平。[融合对照](results/supervised/runs/physics_fusion/comparison.json)
- **主干和几何分支同时融合**。扇区 RMSE 为 0.11948，走偏后 4 单位误差为 1.163%，$T=100$ 为 22.436%，自交案例从 3/16 增至 4/16。遮掉融合投影会改变约 3.4%–3.8% 的半步预测位移，高于上一轮附加输入的不足 1%。模型对物理通道的依赖更强，几何精度的收益仍较小。[融合对照](results/supervised/runs/physics_fusion/comparison.json) · [投影遮挡](results/supervised/runs/physics_fusion/feature_usage.json)

## 训练方法对照

### 历史滚动与修补训练

$T=20$ 修补结果见 `c4cf2f9` 的 `results/supervised/world_t20`。下列 $T=100$ 微调除注明外为第一个种子，终点指标来自同一时间表的自由滚动，记录在同提交的 `results/supervised/rollout_t100`。

- **给同帧修补器加噪**。终点为 21.05% 和 20.15%，均差于 18.89% 的直接基线。合成噪声没有覆盖实际粗预测的偏差。
- **用交叉拟合预测训练修补器**。终点 18.29% 和 18.57%，轻微改善。使用样本外预测构造偏差更接近实际输入，但没有超过频带重复修补。
- **真值历史微调**。$T=100$ 两种子自由终点 28.74% 和 32.23%，自交均为 5/8。作为单步精度对照，不能消除部署时预测历史与真值历史的差异。
- **混合真值与生成历史**。第一种子终点 29.73%，自交降到 3/8。部分混合输入改善几何有效性，终点精度仍未改善。
- **截断梯度推前**。两种子自由终点 28.03% 和 31.77%。用模型自行烧入后的状态训练有小幅收益，仍弱于同预算的 Self Forcing。
- **三块 Self Forcing**。两种子相对真值历史的终点为 28.74%→27.05% 和 32.23%→28.84%，自交均从 5/8 降为 4/8。训练暴露在自身误差下能改善闭环预测，但仍有大量终点与分叉偏差。
- **随时间变化的扰动噪声**。单种子自由终点 28.93%。按时间加噪未超过真值历史或实际生成历史。
- **尖端事件加权**。单种子自由终点 28.21%。提高局部径向形变的权重略改善全局误差，但未超过 Self Forcing。
- **直达与滚动路径一致性**。单种子自由终点 29.26%。强迫两条预测路径一致没有提升各自的正确性。
- **六块从初形连续滚动**。单种子自由终点 27.16%。增加展开块数能接近三块 Self Forcing，但该轮没有进一步改善终点。
- **早段目标重采样**。单种子自由终点 28.19%。增加早段目标没有解决后续误差累积。这与仅训练到 $T=5/10/20$ 再外推的设问不同。

### 数据扩大后的训练与纠偏

- **同一 128 初形训练更久**。同一种子从 6,000 增到 8,400 步，$T=100$ 终点误差为 21.77%→21.80%。这组数据上优化步数增加没有带来可测的终点收益。[训练量对照](results/supervised/runs/geometry_equal_exposure/summary.json)
- **16 帧中途监督**。两个种子合计 $T=40$ 误差 24.78%→21.87%，分叉匹配 36→41/116，$T=100$ 终点为 21.78%→22.19%。更长的训练闭环帮助中段，但改善未稳定传到终点。[逐种子对照](results/supervised/runs/geometry_long_rollout/comparison.json)
- **将一半片段固定从初形起步**。3,000 步 CPU 小试的终点 24.44%→26.48%。8,400 步 CUDA 两种子的平均终点 22.19%→21.52%，自交 4→1/16，分叉匹配 41→47/116，但 $T=10$ 真值起步的一步优势从约 29% 缩到 3%–6%。较高的初形起步比例改善自由滚动，代价是局部输入覆盖和短步表现下降，终点收益随预算和种子变化。[CPU 小试](results/supervised/runs/geometry_initial_starts/cpu_comparison.json) [CUDA 对照](results/supervised/runs/geometry_initial_starts/cuda_comparison.json)
- **直接用细时间标签替换粗标签**。0.5 时距的真值一步误差 0.268%→0.323%，窗口自由误差 2.683%→3.015%。替换削弱了粗步监督，短步位移在原坐标目标里也更小。[时间密度对照](results/supervised/runs/tip_density/comparison.json)
- **保留粗步并按时距平衡细步损失**。相同 0.5 推理时距下，真值一步 0.272%→0.253%，窗口自由误差 2.747%→2.489%，匹配分叉 10→12/28，整轨终点 21.68%→21.76%。细标签在局部可以学到，当前训练没有把收益稳定传到从初形开始的整轨。[时间密度对照](results/supervised/runs/tip_density/comparison.json)
- **用模型自身走偏状态训练纠偏**。和真值起步训练相比，8 步误差 3.48%→3.45%，两种子均未出现误差收缩，分叉均为 14/56。真实走偏输入比合成噪声更接近部署，但原始纠偏损失比全程回放小得多。[纠偏对照](results/supervised/runs/tip_correction/comparison.json)
- **纠偏损失权重提高到 16 倍**。8 步误差降到 3.13%，分叉匹配升至 21/56，仍无一次低于约 1.50% 的起步误差。整轨终点由微调前的 21.76% 升到 22.73%。加重监督能减慢误差增长，没有形成收缩。对全程均匀施加这类校正存在局部与长时表现的取舍。[纠偏对照](results/supervised/runs/tip_correction/comparison.json)
- **同状态瞬时法向速度监督**。独立集速度 RMSE 0.12706→0.12029，走偏后 4 单位的轮廓误差 1.241%→1.167%，整轨终点 22.54%→22.44%，走偏分叉 14→15/37。真实压力导出的符号速度给出比位置差更直接的监督，但目前幅度有限。训练和推理的界面推进仍由同一网络完成。[速度标签对照](results/supervised/runs/normal_velocity_labels/comparison.json)

## 数据工程与诊断

- **物理标签与划分**。从 8 个主导模态族的 64 个初形扩到 96 和 128，最终保留 112／8／8 的独立初形划分。另有 $T=1000$ 265 帧与 $T=100$ 177 帧两套时间表。旋转、镜像与周期重编号同步应用在同一轨迹内，增加可见方向而不改变独立形态数。长时新增标签从对应 $T=100$ 状态续算并按双分辨率、通量和自交检查验收。[长时数据](data/supervised/trajectories_t1000_128/summary.json) [早期加密数据](data/supervised/trajectories_t100_dense_128/summary.json)
- **搜索挑选初形**。8 个搜索候选对比 8 个随机候选，模型分歧 8/8 胜出，物理终点差异仅 5/8，预测终点难度仅 4/8。分歧可用来筛查有趣形态，但该先导没有证明搜索选样能提高重训后的泛化。[选样先导](data/supervised/trajectories_pilot/acquisition.json)
- **64→96 初形与早期时间加密**。在同一原评估网格上，直接模型 31.30%→27.29%，四帧 23.55%→20.84%，几何 22.13%→22.06%。该阶段同时改变了初形数量和标签时间间隔。改变滚动推理网格也会改变相对排名。[数据对照](results/supervised/runs/dense_t100/summary.json)
- **96→128 等曝光对照**。保持每个初形约 600 次训练抽样，几何模型两种子终点平均改善 0.83 个百分点。从 $T=5/10/20$ 给真值后，128 组下一帧比保持不动低 23%–56% MSE。增加初形覆盖较明显地改善了局部推进，闭环收益小得多。自交与分叉匹配的种子趋势不一致。[等曝光对照](results/supervised/runs/geometry_equal_exposure/summary.json)
- **后段分叉标签加密**。原先间隔 5 的四个试点分叉，在 0.5 网格上提早 0.5–4 个时间单位被识别，随后扩到 42 个训练窗口和 14 个来自 6 个独立留出初形的窗口。粗存档会混合分叉时间量化与模型失误，评估时应固定相同预测时距。[试点](data/supervised/late_tip_windows/summary.json) [分叉加密数据](data/supervised/late_tip_density/summary.json)
- **早期真值重启**。从 $T=5、10、20$ 的数值界面与历史起步，128 初形组的一步误差低于保持形状的基线。从 $T=20$ 给真值后的半步轮廓误差约 0.4%，但继续自由滚到 $T=40$ 仍达 8.5%–10.6%。正确起步只能消除此前的偏差，后续局部更新仍会积累误差。[真值重启诊断](results/supervised/runs/geometry_equal_exposure/summary.json)
- **新形态与同状态物理反事实**。另生成 8 个独立初形。对比模型走偏曲线和从同一曲线重新数值推进的轨迹时，0.5 单位的一步误差约 0.167% $R$，1.5 单位的自由误差约 0.490% $R$。原轨迹与物理走偏轨迹本身也会分离。使用相同状态的物理推进作目标可以区分真实敏感性和模型误差积累。[新形态](data/supervised/drift_diagnostic_fresh/summary.json) [同状态诊断](results/supervised/runs/drift_diagnostic/comparison.json)
- **有符号误差与幅值校准**。有符号法向误差的跨步一致性约 0.996–0.999，单个幅值增益可以降低部分一步全局误差，却未稳定改善尖端扇区。持续偏向同一方向的误差更容易累积。对齐法向和跟踪同一弧长坐标带来测量限制，不能据此断言所有误差仅来自速度幅值。[误差方向](results/supervised/runs/drift_diagnostic/normal_bias.json) [幅值控制](results/supervised/runs/drift_diagnostic/step_amplitude_calibration.json)
- **后段减小推理步长**。两个种子把后段改成半单位步后，独立形态终点分别为 24.46%→23.81% 和 21.78%→21.89%，分叉匹配没有增加。单纯增加反馈次数未稳定改善模型的长时误差。[步长对照](results/supervised/runs/drift_diagnostic/fresh_rollout_step_control.json)
- **高分辨率法向速度标签**。采用 1024／1536 点检查短轨迹，瞬时速度逐级验算至 3072 点，得到 252 个训练样本、48 个评估样本。短时位移差与瞬时速度特别是在走偏状态上有可测差异。压力求解得到的瞬时目标补充了轨迹坐标监督。[标签审计](data/supervised/normal_velocity_labels/summary.json)
- **截断、数值回退与重锚定**。历史 $T=20$ 的几何、预测分歧、物理和自适应组合检查可在部分案例提前报警，但实用策略明显缩短保留时间。真值重锚能缩短误差积累区间。重锚与数值回退是混合系统诊断，独立报告计算代价和纯模型表现。记录在 `c4cf2f9` 的 `results/supervised/world_t20/verification.json` 与 `results/supervised/rollout_t100/anchors.json`。
- **计划中或记录缺失的方向**。未在当前工作树与可用历史提交中找到独立的 仅到 $T=5、10、20$ 训练后外推到 $T=100$ 对照，也没有 AlphaZero 式在线搜索或学习端到端分叉校正器取得收益的记录。已完成的搜索是离线选样先导，已完成的同帧修补、走偏监督与物理输入实验各有不同的对照范围。
