# SMPL Diffusion Lab 学习日志

## 项目目标

从头重写 DiffPose 的核心功能，通过实际开发学习：

- Python 基础
- NumPy 基础
- PyTorch 与张量计算
- DDPM / diffusion
- SMPL 人体模型
- 2D/3D 人体姿态估计
- 深度学习训练与评估
- Python 工程、测试与 Git 工作流

参考项目：

- DiffPose：https://github.com/gongjia0208/diffpose
- 当前仓库：https://github.com/Nyac0de/smpl-diffusion-lab

## 开发环境

- Python：3.11.15
- PyTorch：2.11.0+cu128
- GPU：8 × NVIDIA GeForce RTX 4090D
- Conda 环境：`smpldiff`
- 远程项目目录：`/data/rxxu/projects/smpl-diffusion-lab`

## 当前进度

### Diffusion 基础

- [x] 创建 linear beta schedule
- [x] 计算 `alpha_t = 1 - beta_t`
- [x] 计算累计乘积 `alpha_bar_t`
- [x] 实现 forward diffusion：`q_sample_torch`
- [x] 根据预测噪声恢复 `x0`
- [x] 计算反向分布均值
- [x] 计算 posterior variance
- [x] 实现单步反向采样 `x_t -> x_{t-1}`
- [ ] 实现完整反向采样循环 `x_T -> x_0`

### Diffusion 模型与训练

- [x] 创建 toy dataset
- [ ] 实现 timestep embedding
- [x] 实现最小 MLP 噪声预测模型
- [ ] 实现 DDPM 训练 loss
- [ ] 实现训练循环
- [ ] 在 toy 数据上完成训练和采样
- [ ] 保存与加载 checkpoint
- [ ] 可视化采样结果

### SMPL

- [ ] 理解 SMPL 输入和输出
- [ ] 理解 `betas`
- [ ] 理解 `global_orient`
- [ ] 理解 `body_pose`
- [ ] 理解 `transl`
- [ ] 理解 axis-angle、rotation matrix 和 6D rotation
- [ ] 调用 SMPL 生成 vertices 和 joints
- [ ] 可视化 SMPL mesh
- [ ] 将 diffusion 数据表示替换为人体 pose

### DiffPose 重写

- [ ] 阅读并梳理原始 DiffPose 的输入和输出
- [ ] 理解数据预处理
- [ ] 理解 condition 表示
- [ ] 实现 pose diffusion model
- [ ] 接入真实人体姿态数据
- [ ] 实现训练与评估
- [ ] 与原始 DiffPose 结果进行对照

## 学习记录

### 2026-07-10：完成 DDPM forward process 与反向均值

#### 今天完成了什么

- 检查 PyTorch 和 CUDA 环境
- 确认 PyTorch 可以识别 8 张 GPU
- 实现 linear diffusion schedule
- 实现 `q_sample_torch`
- 实现 `predict_x0_from_noise`
- 实现 `predict_previous_mean_from_noise`
- 全部测试通过：`25 passed`

#### 今天理解了什么

- `beta_t` 表示每一步加入的噪声强度
- `alpha_t = 1 - beta_t`
- `alpha_bar_t` 是从开始到时间步 `t` 的累计信号保留比例
- forward diffusion 可以直接从 `x0` 采样任意时间步的 `x_t`
- DDPM 的噪声预测参数化让模型预测加入的噪声 `epsilon`
- batch 中每个样本可以使用不同的 timestep
- schedule 系数需要从 `(B,)` reshape 成 `(B, 1, ...)` 才能按 batch 广播
- Python 使用缩进表示代码块
- Python 列表元素之间需要逗号
- `object.attribute` 表示访问属性
- `tensor.reshape(shape)` 表示调用方法
- pytest 会先 collection，再执行测试

#### 遇到的问题

- Python 缩进错误
- 多行列表遗漏逗号
- 文件名 `torch.reverse.py` 与 import 名称不匹配
- 函数参数名与测试关键字参数名不一致
- 把 `schedule.alphas` 错写成 `schedule, alphas`
- 把调用 `reshape` 错写成给 `reshape` 属性赋值

#### 如何解决

- 根据 traceback 从第一个错误开始处理
- 使用 `inspect.signature` 检查 Python 实际加载的函数签名
- 使用 `inspect.getsourcefile` 检查模块实际来源
- 使用测试逐步确认每一处修复

#### 下一步

1. 实现 posterior variance
2. 实现单步反向采样 `p_sample_torch`
3. 实现最小噪声预测网络
4. 在 toy 数据上开始训练

## 每次提交前的记录模板

复制下面的模板，在“学习记录”下方增加一节：

```text
### YYYY-MM-DD：本次学习主题

#### 本次完成了什么

-

#### 本次理解了什么

-

#### 遇到的问题

-

#### 如何解决

-

#### 测试结果

-

#### 下一步


 ## 2026-07-10

- 练习 PyTorch 版扩散反向过程实现。
- 学习 `torch.cat`、tensor 切片 `[:-1]`、`alpha_bars_prev` 的构造方式。
- 调试 `tests/test_torch_reverse.py`，修正 tensor 写法和反向采样相关测试。

 ### 2026-07-14：完成 DDPM 单步反向采样

#### 本次完成了什么

- 实现 DDPM posterior variance
- 实现单步反向采样 `p_sample_torch`
- 使用 `nonzero_mask` 控制最后一步是否加入采样噪声
- 为 posterior variance 和单步反向采样添加测试
- Reverse 测试全部通过：`4 passed`
- 项目全部测试通过：`27 passed`

#### 本次理解了什么

- 每个 diffusion 时间步都有一个 posterior variance。
- `posterior_variances` 包含全部时间步的方差，shape 是 `(T,)`。
- batch 中的不同样本可能位于不同时间步，因此要用 `timesteps`
取出每个样本对应的 `posterior_variances_t`。
- `posterior_variances_t` 从 `(B,)` reshape 为 `(B, 1, ...)` 后，
可以与形状为 `(B, ...)` 的数据按 batch 广播。
- `nonzero_mask` 在 `t=0` 时为 0，消去采样噪声项，避免污染最终
的干净输出；在 `t>0` 时为 1，允许正常的随机反向采样。
- `noise_pred` 是神经网络对 forward noise 的预测，将用于计算反向
均值。
- `sampling_noise` 是反向采样时新生成的随机噪声，用于从反向高斯
分布中取样。
- 目前对 `noise_pred` 和 `sampling_noise` 的区别已有初步理解，
后续将在训练循环和完整采样循环中继续巩固。

#### 遇到的问题

- 在 `torch.cat` 的输入列表中误放入了函数对象，导致
`expected Tensor ... but got function`。
- 最终公式中误用了包含全部时间步的 `posterior_variances`，
导致 shape `(2, 4)` 无法与 `(2, 3)` 相乘。
- 一度把 `reshape` 当成可以直接赋值的属性，而不是需要调用的方法。
- 对 `noise_pred` 和 `sampling_noise` 的来源及职责产生混淆。

#### 如何解决

- 从 traceback 的最后一行确定错误类型，再查看具体出错代码。
- 列出参与计算的每个 tensor 的 shape，逐维分析 broadcasting。
- 使用 `posterior_variances[timesteps]` 选取当前 batch 对应的方差。
- 将当前 timestep 的方差从 `(B,)` reshape 为 `(B, 1, ...)`。
- 用训练流程和采样流程分别理解 `noise_pred` 与 `sampling_noise`。

#### 测试结果

- 单步反向采样测试：`1 passed`
- Reverse 测试：`4 passed`
- 全部测试：`27 passed`

#### 下一步

1. 提交当前单步反向采样实现
2. 创建二维 toy dataset
3. 学习 `torch.nn.Module`
4. 实现最小 MLP 噪声预测器
5. 实现第一个 DDPM 训练循环

---
### 2026-07-15：完成二维圆环混合分布 toy dataset

#### 本次完成了什么

- 实现 `sample_ring_mixture`
- 为每个样本随机选择一个圆环 mode
- 将 mode 编号转换为圆周角度
- 使用 `cos` 和 `sin` 计算二维中心坐标
- 为每个二维中心加入独立的高斯扰动
- 使用固定随机种子生成可复现的测试数据
- 添加 toy dataset 的 shape、dtype、有限值和半径测试
- toy data 测试全部通过：`2 passed`
- 项目全部测试通过：`29 passed`

#### 本次理解了什么

- `torch.randint(low=0, high=8, size=(5,))` 会生成 5 个整数，
每个整数的取值范围是 `[0, 8)`，即 0 到 7。
- `num_modes` 控制每个 mode index 的可选范围，`size` 控制输出
tensor 的形状和元素数量。
- `torch.stack(..., dim=1)` 可以把两个 shape 为 `(B,)` 的坐标
tensor 组合成 shape 为 `(B, 2)` 的二维点。
- `samples.shape == (32, 2)` 中，32 是 batch size，2 是每个样本的
特征数，即 `[x, y]`。
- 对于 24 个三维人体关节，不含 batch 的 shape 可以是 `(24, 3)`；
含 batch 时应为 `(B, 24, 3)`。
- `samples = sample_ring_mixture` 保存的是函数对象；
`samples = sample_ring_mixture(...)` 才会调用函数并保存返回的
tensor。
- 当 `noise_std=0.0` 时，扰动为零，samples 与圆环中心相同，因此
每个点到原点的距离约等于指定半径。
- `noise` 与 `centers` 都使用 `(num_samples, 2)`，使每个样本的
x、y 坐标分别获得独立扰动。
- 固定 `torch.Generator` 的随机种子可以让实验和测试具有可复现性。

#### 遇到的问题

- 用非法的 `(,)` 作为 `torch.randn` 的 shape，导致 `SyntaxError`。
- 混淆 `num_modes` 与 `size` 的含义。
- 将 `torch.randint` 误写成不存在的 `torch.radiant`。
- 将变量名混写为 `mode_indice` 和 `mode_indices`。
- 在变量创建之前尝试访问它。
- 将函数对象直接赋给 `samples`，随后访问 `.shape`，导致
`AttributeError`。
- 将 `vector_norm` 拼错为 `vetcor_norm`。
- 多行函数参数之间遗漏逗号。
- 一开始不清楚 `torch.stack` 的 `dim=0` 和 `dim=1` 对输出 shape
的影响。

#### 如何解决

- 使用 `python -m py_compile` 先区分语法错误和运行时错误。
- 将实现拆成 mode index、angle、center、noise、sample 五个小步骤。
- 每一步打印 tensor 内容和 shape，确认后再继续下一步。
- 通过 `torch.Size([5])` 验证每个样本只有一个 mode index。
- 通过 `torch.Size([5, 2])` 验证每个样本拥有一对二维坐标。
- 用 `noise_std=0.0` 的特殊情况验证圆环半径公式。
- 将手动实验转化为 pytest 自动测试。
- 根据 traceback 最后一行区分函数对象、拼写和数值计算问题。

#### 测试结果

- Toy dataset 测试：`2 passed`
- 全部测试：`29 passed`

#### 下一步

1. 学习 `torch.nn.Module`
2. 理解神经网络中的输入层、隐藏层、激活函数和输出层
3. 实现最小 MLP 噪声预测器
4. 让模型接收 `x_t` 和 timestep
5. 实现 DDPM 噪声预测 loss


### 2026-07-15：实现第一个 PyTorch 噪声预测 MLP

  #### 本次完成了什么

  - 实现继承自 `nn.Module` 的 `NoisePredictMLP`
  - 使用两个隐藏层和 SiLU 激活函数构建 MLP
  - 将二维带噪点与归一化 timestep 拼接为模型输入
  - 让模型输出与带噪数据 shape 相同的预测噪声
  - 添加模型输出 shape、dtype 和有限值测试
  - 模型测试通过：`1 passed`
  - 项目全部测试通过：`30 passed`

  #### 本次理解了什么

  - `nn.Module` 是 PyTorch 神经网络模型的基础类。
  - `__init__` 用来创建并注册网络层，`forward` 用来定义数据通过网络
    时的计算过程。
  - 调用 `model(xt, timesteps)` 时，PyTorch 会执行模型的 `forward`。
  - timestep 原本的 shape 是 `(B,)`，reshape 为 `(B, 1)` 后才能与
    shape 为 `(B, data_dim)` 的数据沿 feature 维拼接。
  - 对二维数据，模型输入是 `[x, y, normalized_t]`，因此第一层输入
    维度为 3；模型预测二维噪声，因此输出维度为 2。
  - `nn.Linear(in_features=3, out_features=32)` 的权重 shape 是
    `(32, 3)`，因为 PyTorch 按 `(out_features, in_features)` 保存
    权重。
  - batch 维不包含在模型权重中；同一套权重会应用到 batch 中的所有
    样本。
  - 对 shape 为 `(B, 72)` 的 pose vector，`data_dim` 应为 72；
    拼接一个 timestep 后第一层输入维度为 73，输出仍为 72。
  - 本次一开始将 `data_dim` 和“加入 timestep 后的模型输入维度”
    混淆，经过 shape 分析后完成区分。

  #### 遇到的问题

  - 模型文件存在，但 class 名称与测试导入名称不一致，导致
    `ImportError`。
  - 一开始不清楚 `Linear` 权重为什么使用 `(32, 3)` 而不是
    `(3, 32)`。
  - 一开始把 72 维 pose 的 `data_dim` 错认为 73。

  #### 如何解决

  - 检查 class 定义与测试 import 的名称是否完全一致。
  - 打印 `model.named_parameters()`，观察每层权重和偏置的 shape。
  - 使用矩阵乘法 shape：
    `(B, 3) @ (3, 32) -> (B, 32)` 理解 Linear 层。
  - 区分原始数据维度 `data_dim` 和拼接 timestep 后的模型输入维度。

  #### 测试结果

  - Noise MLP 测试：`1 passed`
  - 全部测试：`30 passed`

  #### 下一步

  1. 实现 DDPM 噪声预测 loss
  2. 学习 MSE loss
  3. 学习梯度、`backward()` 和参数的 `.grad`
  4. 实现一个 optimizer step
  5. 将单步训练扩展为完整训练循环


### 2026-07-16：完成 DDPM 噪声预测 loss 和反向传播

  #### 本次完成了什么

  - 实现 `noise_prediction_loss`
  - 使用 `q_sample_torch` 根据 `x0`、timestep 和真实噪声生成 `xt`
  - 将 `xt` 和 timestep 输入噪声预测 MLP
  - 使用 MSE 比较 `noise_pred` 与真实 `noise`
  - 调用 `loss.backward()` 为模型参数计算梯度
  - 添加测试，验证 loss 使用的是带噪输入 `xt`
  - 添加测试，验证所有模型参数都获得形状正确且数值有限的梯度
  - Loss/gradient 测试通过：`2 passed`
  - 项目全部测试通过：`32 passed`

  #### 本次理解了什么

  - `noise` 是 forward diffusion 中主动生成的已知噪声，也是训练监督目标。
  - `noise_pred` 来自神经网络，是模型对真实噪声的预测。
  - 模型必须接收 `xt` 而不是 `x0`，因为实际生成阶段没有干净的 `x0`
    可以作为输入。
  - 默认的 `F.mse_loss` 会计算所有对应元素的平方误差并取平均，
    因此返回标量 tensor。
  - 对 shape 为 `(32, 2)` 的预测和目标，默认 MSE 会对全部 64 个元素
    的平方误差取平均。
  - `loss.backward()` 负责计算梯度，但不会直接修改模型参数。
  - `optimizer.step()` 才会利用梯度更新模型参数。
  - 每个参数元素都有对应梯度，因此 `parameter.grad.shape` 应与
    `parameter.shape` 相同。
  - `parameter.grad is None` 表示尚未计算出梯度；梯度 tensor 全为 0
    表示梯度存在，但当前梯度值为零。

  #### 遇到的问题

  - 终端从 `smpldiff` 环境切换到了 `base`，导致找不到 pytest。
  - 将 `functional` 拼错为 `funticonal`，导致 `ImportError`。
  - 使用大写 `torch.Tensor` 从列表创建数据并传入 `dtype`，导致
    `TypeError`。
  - 测试中使用了尚未定义的 `noise`，导致 `NameError`。
  - 一开始将 MSE 称为“均值方差”，后来明确其含义是“均方误差”。

  #### 如何解决

  - 通过终端提示符和 `which python` 确认当前 Conda 环境。
  - 区分 `py_compile` 的语法检查和 pytest 的运行时行为检查。
  - 从 Python 列表创建 tensor 时使用小写 `torch.tensor`。
  - 根据 traceback 定位未定义变量和拼写错误。
  - 在 backward 前后检查每个参数的 `.grad`。
  - 检查梯度 shape 与参数 shape 一致，并用 `torch.isfinite` 检查
    梯度中没有 NaN 或无穷值。

  #### 测试结果

  - Noise prediction loss 与梯度测试：`2 passed`
  - 全部测试：`32 passed`

  #### 下一步

  1. 创建 Adam optimizer
  2. 学习 `optimizer.zero_grad()`
  3. 学习梯度累积
  4. 使用 `optimizer.step()` 更新模型参数
  5. 验证一次训练步骤确实改变了参数
  6. 将单步训练扩展为完整训练循环
--以后重新 clone 仓库：需要再次执行一次：
git config core.hooksPath .githooks
