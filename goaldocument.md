**实验规划总结书**

**DiffStateGrad 无训练动态谱门控**

基于 ReSample 的复现、解析改进与稳健性验证

**研究目标：**验证噪声感知的动态谱门控能否优于固定方差阈值

**当前范围：**第一阶段不训练任何网络，仅修改推理时的梯度投影

**计算资源：**4 × NVIDIA GeForce RTX 4090（24 GB）

**实验对象：**FFHQ 256 × 256；ReSample / DiffStateGrad-ReSample

**版本日期：**V1.0｜2026-07-16

> **核心结论｜**本项目第一阶段无需训练。研究工作集中于将 DiffStateGrad 的固定阈值投影替换为由扩散噪声水平决定的动态硬门控与软门控，并在相同测量、相同随机种子下完成配对验证。

# 1. 研究问题与可检验假设

研究问题：DiffStateGrad 使用固定方差保留阈值 τ=0.99。该阈值没有显式利用扩散时刻对应的噪声尺度。若依据当前扩散状态的噪声谱边界，自适应决定哪些奇异方向允许 measurement guidance 通过，是否能够减少伪影和极端失败，同时维持数据一致性？

- **H1｜**动态性：门控强度应随扩散时刻与状态谱自动变化，不再依赖全程固定的 τ。

- **H2｜**稳健性：相较固定阈值，动态软门控应改善 LPIPS 的尾部表现并降低失败率。

- **H3｜**保真性：动态门控不能以显著牺牲 measurement consistency 或平均 PSNR 为代价。

- **H4｜**效率：新增谱权重计算相对原 DiffStateGrad 的运行时间增幅应控制在 5% 以内。

# 2. 方法与对照组

设当前扩散状态

$$
Z_t=U_t\operatorname{diag}(s_t)V_t^\top,
$$

测量一致性梯度为 $G_t$。所有方法均冻结预训练扩散模型，仅改变 $G_t$ 在一次数据一致性更新前的投影方式。

| **编号** | **方法**               | **投影规则**         | **作用**                     |
|----------|------------------------|----------------------|------------------------------|
| M0       | ReSample               | 不投影；P=0          | 确认基础求解器性能与运行时间 |
| M1       | 固定阈值 DiffStateGrad | τ=0.99，P=5          | 复现论文设置，作为直接基线   |
| M2       | 动态硬门控             | 保留 $s_{t,i}>b_t$ 的方向 | 检验动态秩本身是否有效       |
| M3       | 动态软门控（主方法）   | 按谱置信度连续衰减   | 避免硬截断并保留边界方向     |

## 2.1 无训练动态谱规则

对第 t 个扩散时刻，使用扩散噪声标准差 σₜ 构造随机噪声矩阵的近似谱边界：

$$
b_t=\sigma_t\left(\sqrt{H}+\sqrt{W}\right).
$$

动态硬门控根据该边界直接选择秩：

$$
r_t=\#\{i:s_{t,i}>b_t\}.
$$

动态软门控对每个奇异方向赋予连续权重：

$$
a_{t,i}=\sqrt{\max\left(0,1-\left(\frac{b_t}{s_{t,i}}\right)^2\right)}.
$$

$$
G_t^{\mathrm{proj}}
=U_t\operatorname{diag}(a_t)U_t^\top
G_t
V_t\operatorname{diag}(a_t)V_t^\top.
$$

**实现约束：**$a_{t,i}\in[0,1]$，因此投影不会放大梯度范数。M2 与 M3 均不包含可学习参数，也不进行 test-time optimization。

# 3. 数据、任务与公平性控制

| **阶段**   | **图像**           | **任务**                                         | **目的**                       |
|------------|--------------------|--------------------------------------------------|--------------------------------|
| 快速检查   | 仓库自带 2 张      | Box inpainting                                   | 确认环境、权重、指标与保存路径 |
| Pilot      | 固定 20 张 FFHQ    | Box inpainting；Phase retrieval                  | 低成本判断主方法是否有趋势     |
| Core       | 固定 100 张 FFHQ   | Box inpainting；Gaussian deblur；Phase retrieval | 形成主要定量表与配对统计       |
| Robustness | Pilot 的 20 张子集 | 噪声、退化强度、guidance step size 扰动          | 验证稳定性与未调参表现         |

- 所有方法使用同一图像、同一 measurement、同一 mask/退化核、同一观测噪声和同一扩散初始 seed。

- 先固定 100 张测试图像 ID，并随代码发布 manifest；不根据测试结果更换样本。

- Pilot 只用于决定是否进入 Core；Core 完成前锁定方法公式和主要超参数。

- 仓库未公开论文原始 100 张图像清单，因此目标是复现趋势，不承诺逐位匹配论文表格。

# 4. 实验矩阵

## 4.1 Pilot：先回答“值不值得继续”

| **任务**        | **样本数** | **方法**          | **主要观察**                         |
|-----------------|------------|-------------------|--------------------------------------|
| Box inpainting  | 20         | M0 / M1 / M2 / M3 | LPIPS、PSNR、worst-10%、伪影与秩轨迹 |
| Phase retrieval | 20         | M0 / M1 / M2 / M3 | 失败率、LPIPS 分布、不同 seed 稳定性 |

总计 160 次 reconstruction。四卡独立运行，预计 3–4 小时完成，包括保存和指标计算。

## 4.2 Core：形成主要结果

| **任务**        | **基础设置**              | **样本数** | **比较** |
|-----------------|---------------------------|------------|----------|
| Box inpainting  | 128×128 中心/随机位置遮挡 | 100        | M0–M3    |
| Gaussian deblur | 61×61 kernel，强度 3.0    | 100        | M0–M3    |
| Phase retrieval | oversample=2.0            | 100        | M0–M3    |

总计 1,200 次 reconstruction。按论文报告的单张 4090 时间估算，理想计算时间约 20.7 小时；计入调度和 I/O 后按 24–30 小时安排。

## 4.3 Robustness：仅比较关键方法

- 观测噪声：σᵧ∈{0.01, 0.05, 0.10}。

- guidance step size：默认值的 {0.5×, 1×, 2×, 4×}。

- 退化严重程度：Box size 与 blur intensity 各设轻、中、重三档。

- 只比较 M1 与 M3，优先观察平均指标之外的 worst-10% 与 failure rate。

# 5. 指标与统计分析

| **维度**   | **指标**                           | **报告方式**                     |
|------------|------------------------------------|----------------------------------|
| 重建质量   | PSNR、SSIM、LPIPS                  | mean±std、median、配对差值       |
| 尾部稳健性 | worst-10% PSNR/LPIPS、failure rate | 按任务分别报告，并展示分布图     |
| 数据一致性 | ‖A(x̂)−y‖² 与归一化残差             | 确认改善不来自忽略 measurement   |
| 门控行为   | rₜ、均值(aₜ)、有效秩随 t 的轨迹    | 解释方法何时抑制或放行梯度       |
| 效率       | 秒/图、峰值显存、相对 M1 开销      | 同卡、同环境、去除首次加载后统计 |

**统计原则：**以图像为配对单位，对 M3−M1 的 PSNR、LPIPS 与 residual 做 paired bootstrap（建议 10,000 次），报告 95% 置信区间；同时给出原始散点或箱线图，避免仅依赖均值。

# 6. Go / No-Go 判定标准

- Pilot 进入 Core：M3 在至少一个任务上平均 LPIPS 相对 M1 下降 ≥3%，或平均 PSNR 提升 ≥0.2 dB。

- 尾部价值：Box inpainting 或 Phase retrieval 的 worst-10% LPIPS 相对下降 ≥8%，或失败率明显降低。

- 无明显副作用：任一 Pilot 任务平均 PSNR 相对 M1 的下降不超过 0.15 dB，measurement residual 不显著恶化。

- 效率可接受：M3 相对 M1 的推理时间增加不超过 5%。

- 若平均指标持平而尾部显著改善，仍进入 Core，并将论文定位为 robustness / failure suppression。

# 7. 实施步骤与时间安排

| **阶段**  | **工作内容**                      | **人力时间** | **4卡计算时间** | **退出条件**             |
|-----------|-----------------------------------|--------------|-----------------|--------------------------|
| S0 环境   | 适配 4090；下载权重；跑通仓库图像 | 0.5–1 天     | \<1 小时        | M0/M1 均能产出图像与指标 |
| S1 基线   | 固定 20 张；复现 M0/M1；锁定 seed | 0.5 天       | 1–2 小时        | 结果稳定、日志齐全       |
| S2 方法   | 实现 M2/M3；记录谱与门控轨迹      | 0.5–1 天     | \<1 小时        | 单元检查与极端时刻正常   |
| S3 Pilot  | 两任务、四方法、20 张             | 0.5 天       | 3–4 小时        | 通过 Go / No-Go          |
| S4 Core   | 三任务、四方法、100 张            | 1 天         | 24–30 小时      | 主要定量结果完成         |
| S5 稳健性 | 噪声/步长/严重度扫描              | 1 天         | 1–2 天          | 完成消融与尾部分析       |

> **推荐节奏｜**第 1 天跑通并复现基线；第 2 天实现动态门控；第 3 天完成 Pilot。只有通过 Pilot 判定标准，才投入 100 张 Core 与稳健性扫描。

# 8. 实现要点

1.  整理运行环境。官方 environment 较老；优先建立适配 RTX 4090 的现代 PyTorch 环境，同时保留原配置文件。

2.  修正复现入口。默认 image_id 在仓库中不存在；论文 ReSample 使用 P=5，而代码默认 P=1；四卡通过 CUDA_VISIBLE_DEVICES 分配独立 worker。

3.  抽象 projector 接口。统一支持 none、fixed、noise_hard、noise_soft 四种模式，确保除投影外其余代码路径完全一致。

4.  增加批量 runner。每个 GPU 只加载一次模型，连续处理分配到的 image IDs，避免逐图重新加载 checkpoint。

5.  统一日志。每张图保存 config、seed、PSNR/SSIM/LPIPS、measurement residual、runtime、rₜ 与权重摘要。

6.  先做数值检查。验证 aₜ,ᵢ∈\[0,1\]、‖Gₜᵖʳᵒʲ‖≤‖Gₜ‖，并检查早期高噪声时门控不会产生 NaN/Inf。

# 9. 风险与应对

| **风险**                             | **影响**               | **应对措施**                                                            |
|--------------------------------------|------------------------|-------------------------------------------------------------------------|
| 官方代码与论文配置不完全一致         | 难以逐位复现           | 记录 commit；显式设置 P=5；发布自己的 image manifest 与完整命令         |
| RTX 4090 与旧 CUDA/PyTorch 不兼容    | 环境安装失败或算子报错 | 优先升级 PyTorch/CUDA；先完成 2 张图 smoke test                         |
| Phase retrieval 默认配置不稳定       | 掩盖方法差异           | 先在 Box/Gaussian 上验明基线，再单独校准 phase 设置并固定               |
| 理论噪声边界与实际 latent 尺度有偏差 | 门控过强或过弱         | 记录实测谱；增加 schedule edge 与 empirical tail 两个无训练版本作为消融 |
| 动态门控只改善少数失败样本           | 均值提升不明显         | 将 worst-10%、failure rate 与配对分布设为主要稳健性证据                 |

# 10. 预期产出与后续决策

- 可复现的 4-GPU 批量推理入口与固定测试 manifest。

- 统一 projector 模块：none / fixed / noise_hard / noise_soft。

- Pilot 与 Core 的 CSV 指标、图像结果、谱门控轨迹和运行时间日志。

- 主结果表、稳健性分布图、失败案例可视化及实现消融。

- 若无训练 M3 已取得稳定优势，则其本身构成主要方法；只有在其提升有限且数据表明固定解析规则不足时，再考虑可学习门控作为第二阶段。

# 附录 A｜最小运行配置

以下参数用于第一轮对照；新增 projection_mode 为计划中的统一接口。

| **方法** | **关键参数**                                     |
|----------|--------------------------------------------------|
| M0       | projection_mode=none；period=0                   |
| M1       | projection_mode=fixed；var_cutoff=0.99；period=5 |
| M2       | projection_mode=noise_hard；period=5             |
| M3       | projection_mode=noise_soft；period=5             |

# 附录 B｜来源

- [Diffusion State-Guided Projected Gradient for Inverse Problems（ICLR 2025）](https://arxiv.org/abs/2410.03463)

- [DiffStateGrad 官方代码仓库](https://github.com/Anima-Lab/DiffStateGrad)

- [ReSample 官方代码仓库](https://github.com/soominkwon/resample)
