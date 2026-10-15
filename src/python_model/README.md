# 手写数字识别 Python 模型

使用 PyTorch 实现 MNIST **0–9 单个手写数字**识别，包含训练、验证、测试和图片推理。不包含中文手写字符或多位数字的检测、分割。

## 使用 uv 创建环境

在工程根目录执行：

```sh
cd src/python_model
uv venv --python 3.12
uv sync --frozen
```

依赖由 `pyproject.toml` 声明、`uv.lock` 锁定，环境位于本目录 `.venv/`。默认从 PyTorch 官方 CPU 源安装，适用于没有 CUDA 的开发机。`uv run` 自动使用该环境，无需手动激活。首次同步依赖需要网络。

## 网络结构

| 层 | 输出形状（省略 batch） |
| --- | --- |
| 输入，灰度像素除以 255 | 1 × 28 × 28 |
| Conv2d，1→8，5×5，stride=1，VALID | 8 × 24 × 24 |
| ReLU + MaxPool2d，2×2 | 8 × 12 × 12 |
| Flatten + Linear + ReLU | 64 |
| Linear | 10 |

网络有 74,650 个可训练参数。训练输出 logits，配合交叉熵损失；推理时用 softmax 输出各数字概率。训练和推理统一使用黑底亮字、`[0, 1]` 输入。

## 训练与测试

```sh
# 首次运行下载 MNIST；后续已有数据时可省略 --download。
uv run train.py --download --epochs 10

# 使用保存的最佳模型，单独评估 10,000 张测试图片。
uv run train.py --evaluate
```

默认使用 CPU、4 个计算线程、batch size 128、Adam（学习率 0.001）及随机种子 42。训练集固定拆为 55,000 张训练和 5,000 张验证图片，按验证准确率选择最佳模型；测试集不参与模型选择。相同环境中固定随机种子并启用确定性算子。

默认输出：

- `data/MNIST/`：数据集。
- `checkpoints/best.pt`：最佳模型参数、架构版本和训练元数据。
- `checkpoints/best.metrics.json`：每轮训练/验证指标、最佳轮次及最终测试指标。

这些生成文件与 `.venv/` 均已加入 Git 忽略规则。可以使用 `--data-dir`、`--checkpoint`、`--batch-size`、`--learning-rate`、`--threads` 和 `--seed` 自定义配置。默认数据和模型路径相对于脚本目录，不依赖命令执行目录。再次训练会覆盖指定的输出文件；需要保留多个实验时，设置不同的 `--checkpoint`。

本机使用默认配置实际训练 5 轮（`uv run train.py --download --epochs 5`）：最佳模型为第 4 轮，验证准确率 **97.88%**，10,000 张测试图片准确率 **98.01%**。本地已生成该模型及指标文件；生成文件不随源码提交，其他环境需自行训练。

## 识别一张图片

```sh
uv run predict.py /path/to/digit.png

# 指定前景为黑色，适用于白底黑字。
uv run predict.py /path/to/digit.png --polarity dark

# 原始 28×28 MNIST 图片保留像素布局。
uv run predict.py /path/to/mnist.png --mnist-layout --polarity light
```

输入应只包含一个数字。普通图片自动转灰度，依据边框估计黑白极性，裁剪前景，保持比例缩放到 20×20 区域，然后在 28×28 画布中按灰度重心居中。透明背景按白色合成。复杂背景、阴影和多个数字需要额外的图像处理；可通过 `--polarity light|dark` 覆盖自动判断。纯空白输入会报错。

输出 JSON 包含 `digit`、`confidence` 和全部 10 类概率；`confidence` 是 softmax 分数。预测必须先有训练生成的 checkpoint。

## 验证代码

```sh
uv run pytest -q
```

测试覆盖合成图案的实际学习、保存/重载后预测一致性、图片极性、重心居中、透明背景、原始 MNIST 像素保持及无效输入。单元测试无需下载 MNIST。

## 与 Zynq 卷积加速器的关系

第一层的每个输出通道都是对同一张单通道图像做 5×5 VALID 互相关，与现有 RTL 的核方向及尺寸规则一致，8 个输出通道可分别提交任务。ReLU、池化和全连接层可在 PS 上执行。

`brevitas_model.py` 使用 [Brevitas](https://github.com/Xilinx/brevitas) 实现 16 位量化模型；`quantize.py` 负责校准、可选 QAT、参数导出和评估；`fixed_point.py` 作为整数推理与硬件比对参考。当前尚未接入 AXI；上板时仍需实现任务提交、缓存维护及缓冲区地址对齐。卷积整数计算已与现有 `app_component/src/conv_cpu.c` 逐元素对比验证，尚未完成板级运行验证。

## 16 位定点量化

使用 **Brevitas 0.12.1** 的 `QuantConv2d`、`QuantLinear` 和 `QuantIdentity`，权重/激活均为 16 位（W16A16），支持 PTQ 和 QAT。依赖由 uv 管理并写入 `uv.lock`；当前兼容环境为 PyTorch 2.12.1 CPU、torchvision 0.27.1、NumPy 1.26.4。

```sh
uv sync --frozen

# PTQ：直接读取已有浮点权重，校准后导出，无需重新训练。
uv run quantize.py

# 自定义权重、校准图片数量和导出目录。
uv run quantize.py --checkpoint checkpoints/best.pt \
  --calibration-samples 1024 --output-dir checkpoints/int16

# QAT：在量化模型上额外微调 1 轮，并按验证准确率选择最佳状态。
uv run quantize.py --qat-epochs 1 --qat-learning-rate 1e-5 \
  --output-dir checkpoints/int16_qat

# 使用量化后的模型识别单张图片。
uv run predict.py /path/to/digit.png \
  --int16-model checkpoints/int16/model_int16.npz
```

默认从原 checkpoint 对应的 55,000 张训练图片中抽取 1,024 张校准，随机种子为 42。`brevitas.graph.calibrate.calibration_mode` 收集激活绝对值的批次 99.99 百分位统计，初始化可学习尺度；权重采用库内置 MSE 固定点量化器。两者均约束为 2 的整数次幂，导出时直接读取 Brevitas 已生成的整数和尺度，不再执行另一套手工量化。代码中的基类名称含 `Int8`，但派生配置明确设置 `bit_width=16`，导出也检查实际位宽。

`--qat-epochs` 默认为 0。启用 QAT 后，在全部 55,000 张训练图片上同时微调权重和可学习尺度，使用独立的 5,000 张验证图片选择最佳状态，并将校准后的初始模型纳入比较。测试集始终只用于最终评估。`metrics.json` 记录 QAT 各轮指标及最佳轮次；QAT 后与原模型的 logits 差异也包含微调带来的权重变化。

本工程的硬件适配只有舍入规则和导出格式。Brevitas 默认中点取偶，与当前 RTL 不一致，因此通过库的 `float_to_int_impl` 接口注入 `HardwareRoundSTE`，前向执行中点远离零，反向使用直通梯度。Brevitas 仿真使用 float64 容器，以准确表示 W16A16 的乘加结果；量化参数和部署激活仍为 int16。导出后在完整测试集比较 Brevitas 与整数参考的每个 logits 编码，出现任何差异时命令返回失败。

### 数值约定

- 权重、层间激活和最终 logits 为有符号 **int16**，范围 `[-32768, 32767]`，零点为 0。
- 每层独立选择二进制小数位 `F`，实数值为 `integer × 2^(-F)`。这属于整数定点运算，不是 FP16 浮点。
- 输入小数位固定为 14；灰度 0–255 编码为 `round(pixel / 255 × 16384)`，白色 1.0 可精确表示。若直接处理 uint8，可用整数式 `(pixel * 16384 + 127) // 255`，乘法需先扩展到 int32。
- 偏置使用 Brevitas 的 **Int32Bias**，小数位为 `Fx + Fw`，与现有硬件 `BIAS` 寄存器一致。导出检查偏置是否溢出 int32，以及尺度和右移是否符合硬件约束；不满足时明确报错。
- 乘加使用宽累加器；Python 参考实现为 int64，卷积结果的保守范围检查保证可放入 RTL 的 40 位累加器，全连接在 PS 上使用 int64。
- `OUT_SHIFT = Fx + Fw - Fy`，满足硬件的 0–31 范围。右移时就近舍入，中点远离零，之后饱和到 int16。负数不能直接用加正偏置后算术右移替代此规则。
- ReLU、2×2 最大池化和两层全连接均在整数域执行。各类别 logits 共享相同尺度，可直接 `argmax` 得到类别；预测命令只在显示 softmax 概率时转换回浮点。

整数层计算为：

```text
A = bias_int32 + Σ(input_int16 × weight_int16)
S = Fx + Fw - Fy
S = 0: Y = saturate_int16(A)
S > 0: Y = saturate_int16(sign(A) × ((abs(A) + 2^(S-1)) >> S))
```

本次第 10 轮浮点 checkpoint 的实际量化配置如下。重新训练后应以导出的 `manifest.json` 为准。

| 层 | 输入小数位 Fx | 权重小数位 Fw | 偏置小数位 | 输出小数位 Fy | OUT_SHIFT |
| --- | --- | --- | --- | --- | --- |
| conv | 14 | 15 | 29 | 12 | 17 |
| fc1 | 12 | 15 | 27 | 10 | 17 |
| fc2 | 10 | 15 | 25 | 10 | 15 |

### 导出文件与验证结果

默认导出到 `checkpoints/int16/`，不会覆盖浮点 `best.pt`：

| 文件 | 内容 |
| --- | --- |
| `brevitas.pt` | Brevitas 模型权重、学习到的尺度和元数据，可用 `load_brevitas()` 恢复 |
| `model_int16.npz` | 全部 int16 权重、int32 偏置和量化元数据，供 Python 整数推理加载 |
| `manifest.json` | Brevitas 版本及量化器、各层小数位、右移量、数据布局、累加范围、源权重 SHA-256、校准和 QAT 配置 |
| `conv_weight.bin`、`fc1_weight.bin`、`fc2_weight.bin` | 小端 int16 原始权重，无文件头；卷积为 OIHW，全连接为 OI，按 C 行优先存储 |
| `conv_bias.bin`、`fc1_bias.bin`、`fc2_bias.bin` | 小端 int32 原始偏置，按输出通道排列 |
| `metrics.json` | 浮点/Brevitas/整数参考准确率、逐值一致性、logits 误差、各层饱和计数和 QAT 训练记录 |

`manifest.json` 的 `conv_hardware` 提供 8 次单输出通道任务的公共配置和每通道偏置：`CONFIG=7`（5×5、16 位、启用舍入），输入 28×28、输出 24×24，源/目标行跨度分别为 56/48 字节。每次装载该通道连续的 25 个权重，16 位有符号值写入权重寄存器低 16 位；输出按通道保存，随后执行 ReLU 和池化。DMA 缓冲区首地址仍需 8 字节对齐。

当前权重（SHA-256 前缀 `ac4bd278e14e`）在 10,000 张 MNIST 测试图片上的 **Brevitas PTQ** 实测结果：

| 指标 | 结果 |
| --- | --- |
| 浮点准确率 | 98.45% |
| Brevitas / int16 整数参考准确率 | 98.45% / 98.45% |
| 相比浮点准确率变化 | 0 个百分点 |
| 量化前后预测一致率 | 100% |
| logits 平均绝对误差 | 0.000580 |
| Brevitas 与整数参考 logits 差异 | 0 / 100,000 个元素 |
| 输出饱和计数 | conv=0、fc1=0、fc2=1 / 100,000 |

额外运行 **1 轮 QAT**（学习率 `1e-5`）后，验证准确率为 **98.48%**，测试准确率为 **98.65%**；Brevitas 与整数参考的 100,000 个 logits 仍逐值一致。该结果包含额外微调训练，产物位于 `checkpoints/int16_qat/`。百分位校准允许尾部激活发生少量饱和；PTQ 与 QAT 的全部计数分别保存在各自的 `metrics.json`。

`uv run pytest -q` 共通过 31 项测试，覆盖正负中点舍入及 STE 梯度、移位边界、饱和、偏置尺度、Brevitas 与整数网络逐值比较、库 checkpoint 的尺度恢复、QAT 的参数/尺度更新、导出重载和小端二进制文件。具备 C 编译器时，还会编译项目现有 C 参考实现，覆盖 3×3/5×5、极值输入/偏置、不同右移量及带对齐填充的行布局；缺少 C 编译器时仅跳过这些 C 对比测试。
