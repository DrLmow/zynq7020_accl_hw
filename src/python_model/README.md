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

当前代码是浮点训练/推理参考模型，尚未接入 AXI 或实现定点部署。上板前需要量化输入、权重、偏置，确定 `OUT_SHIFT`，验证 RTL 的舍入/饱和行为及量化后准确率，并满足 DMA 缓冲区对齐要求。浮点 checkpoint 不能直接写入硬件寄存器。
