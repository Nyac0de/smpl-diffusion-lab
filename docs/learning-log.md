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
- [ ] 计算 posterior variance
- [ ] 实现单步反向采样 `x_t -> x_{t-1}`
- [ ] 实现完整反向采样循环 `x_T -> x_0`

### Diffusion 模型与训练

- [ ] 创建 toy dataset
- [ ] 实现 timestep embedding
- [ ] 实现最小 MLP 噪声预测模型
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

1.

保存文件。

---

--以后重新 clone 仓库：需要再次执行一次：
git config core.hooksPath .githooks
