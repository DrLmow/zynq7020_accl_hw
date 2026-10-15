"""Brevitas W16A16 model, PTQ calibration, QAT and hardware export."""

import math
from pathlib import Path

import brevitas
import brevitas.nn as qnn
from brevitas.core.scaling import ScalingImplType
from brevitas.graph.calibrate import calibration_mode
from brevitas.quant import Int32Bias
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8ActPerTensorFixedPointMinMaxInit,
    Int8WeightPerTensorFixedPointMSE,
)
from brevitas.quant_tensor import QuantTensor
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.types import Tensor

from fixed_point import build_fixed_model

ARCHITECTURE = "digit-cnn-brevitas-w16a16-v1"


class HardwareRoundSTE(nn.Module):
    """提供与 RTL 一致的舍入规则，并让 QAT 能够传递梯度。

    前向：就近取整，恰好在中间时远离零，如 1.5 -> 2、-1.5 -> -2。
    反向：用 STE（直通估计器）把舍入操作的梯度近似为 1，允许继续训练。
    Brevitas 默认中点取偶，例如 2.5 -> 2；当前 RTL 要求 2.5 -> 3，
    因此需要通过 float_to_int_impl 配置这个适配模块。
    """

    def __init__(self):
        super().__init__()

    def forward(self, values):
        # 先算出符合硬件规则的整数值。
        rounded = torch.sign(values) * torch.floor(torch.abs(values) + 0.5)
        # 前向结果等于 rounded；detach() 阻断括号内的梯度，
        # 反向只通过外面的 values 传递梯度，而不是对 floor() 求导。
        return values + (rounded - values).detach()


# 以下四个类是传给 Brevitas 的“量化配置”，不是四种新的神经网络层。
# class Weight16(某个库内类) 表示继承库的现成配置，只修改下面列出的选项；
# 没写出来的选项仍使用父类的设置。父类名称中的 Int8 是默认 8 位的意思，
# 子类设置 bit_width = 16 后，实际使用 16 位，不会先量化成 8 位再转 16 位。
#
# 用一层计算 y = sum(x * w) + b 理解它们的分工：
#   Input16      -> 第一层收到的图片像素 x。
#   Weight16     -> 每一层训练得到的权重 w。
#   Bias32       -> 每一层训练得到的偏置 b。
#   Activation16 -> 每一层算出的结果 y（后续层继续使用这些结果）。
#
# 量化时大致执行：整数编码 q = 饱和(舍入(浮点值 / scale))，还原值 = q * scale。
# bit_width 决定 q 能用多少位存储；scale 决定每个整数单位代表多少实际数值。
# 这两个概念不同：16 位存储不代表 16 位小数。这里的定点 scale 都是 2 的幂。
# QAT 时库用浮点容器模拟这些数值，导出时才将整数编码存成 int16/int32。


class Weight16(Int8WeightPerTensorFixedPointMSE):
    """权重量化配置：每层权重使用一个尺度，目标为有符号 16 位定点。

    父类名称可拆为：Weight=权重，PerTensor=整个权重张量共享一个尺度，
    FixedPoint=尺度限制为 2 的整数次幂，MSE=用均方误差选择初始尺度。
    这里的误差比较“原权重”和“量化后还原的权重”，不是分类训练损失。
    不同层仍各自拥有尺度；启用 QAT 后，尺度参数也可以参与学习。
    """

    # 使用 16 位表示权重整数编码，小数位数由库选择的 scale 决定。
    bit_width = 16
    # False：允许完整的 signed16 范围 [-32768, 32767]，匹配当前硬件。
    # True 则不使用最小负数，范围为 [-32767, 32767]。
    narrow_range = False
    # 指定“从浮点数舍入到整数”的实现，使用上面的硬件舍入规则和 STE。
    float_to_int_impl = HardwareRoundSTE


class Activation16(Int8ActPerTensorFixedPoint):
    """层输出量化配置：把卷积/全连接计算出的激活限制为 16 位定点。

    Act 是 activation（激活）的缩写，这里指层的输出数据，包括最终 logits；
    它不是权重，也不等同于 ReLU。此配置的量化发生在 ReLU/池化之前。
    激活范围随输入图片变化，因此先用校准图片统计范围、初始化尺度，
    启用 QAT 后再允许尺度参数随训练调整。
    """

    # 输出仍有正有负，使用 signed16；这里没有配置成无符号整数。
    bit_width = 16
    # 用激活绝对值的 99.99 百分位统计初始化尺度，降低极少数大值的影响。
    # 这是人工设置的校准参数，不是准确率目标，也不是保留 99.99% 的位数。
    # 库汇总批次统计并约束为 2 的幂；超出最终表示范围的值会被饱和截断，
    # 因此实际截断比例不一定恰好为 0.01%，新图片的分布也可能不同。
    high_percentile_q = 99.99
    # 与权重量化和 RTL 使用相同的舍入方式，QAT 时通过 STE 传递梯度。
    float_to_int_impl = HardwareRoundSTE


class Input16(Int8ActPerTensorFixedPointMinMaxInit):
    """网络入口的量化配置：按固定尺度处理已经归一化到 [0, 1] 的像素。

    MinMaxInit 表示可用指定的 min_val/max_val 初始化表示范围。
    这里进一步选择 CONST 固定尺度，使图片预处理和硬件输入编码一致。
    """

    # 对齐硬件支持的 signed16 输入格式。
    bit_width = 16
    # CONST = 常量：输入尺度不通过校准估计，也不随 QAT 学习而变化。
    scaling_impl_type = ScalingImplType.CONST
    # 人工指定名义范围 [-2, 2]，据此得到 scale = 2 / 32768 = 2**(-14)。
    # 因而使用 14 位小数；1.0 编码成 16384，能够精确表示且不会饱和。
    # 这是表示范围配置，不会把原图片的 [0, 1] 重新映射成 [-2, 2]。
    # 正端实际最大可表示值为 32767 / 16384，略小于 2.0。
    min_val = -2.0
    max_val = 2.0
    # 例如像素 0.3 编码为 round(0.3 * 16384) = 4915。
    float_to_int_impl = HardwareRoundSTE


class Bias32(Int32Bias):
    """偏置量化配置：继承库内 Int32Bias，使用有符号 32 位整数。

    bit_width = 32 已在父类定义，因此这里不用再写一遍。
    偏置 b 要与 sum(x * w) 相加，必须使用乘积的尺度：
        scale_bias = scale_input * scale_weight
    例如输入 14 位小数、权重 15 位小数，偏置就需要 29 位小数。
    偏置使用 32 位既能容纳这种表示，也与当前 RTL 的 BIAS 寄存器一致；
    “W16A16”仅指定权重/激活为 16 位，不要求偏置和累加器也是 16 位。
    """

    # 保留父类提供的位宽和尺度计算，只替换成当前硬件要求的舍入规则。
    float_to_int_impl = HardwareRoundSTE


def map_values(value, operation):
    # calibration_mode temporarily replaces QuantTensors with plain tensors.
    return (
        value.set(value=operation(value.value))
        if isinstance(value, QuantTensor)
        else operation(value)
    )


class BrevitasDigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(
            act_quant=Input16, return_quant_tensor=True
        )
        options = dict(
            weight_quant=Weight16,
            bias_quant=Bias32,
            output_quant=Activation16,
            return_quant_tensor=True,
        )
        self.conv = qnn.QuantConv2d(1, 8, 5, **options)
        self.fc1 = qnn.QuantLinear(1152, 64, **options)
        self.fc2 = qnn.QuantLinear(64, 10, **options)
        for layer in (self.conv, self.fc1, self.fc2):
            layer.bias_quant.cache_inference_quant_bias = True
        # W16A16 MACs exceed float32's 24-bit integer precision. Float64 makes
        # the fake-quant forward exactly comparable to the int64 reference.
        self.double()

    def quant_forward(self, images):
        values = self.input_quant(images.to(dtype=self.conv.weight.dtype))
        values = self.conv(values)
        values = map_values(values, lambda x: F.max_pool2d(F.relu(x), 2).flatten(1))
        values = self.fc1(values)
        values = map_values(values, F.relu)
        return self.fc2(values)

    def forward(self, images):
        output = self.quant_forward(images)
        return output.value if isinstance(output, QuantTensor) else output

    @classmethod
    def from_float(cls, model):
        quantized = cls()
        with torch.no_grad():
            for target, source in zip(
                (quantized.conv, quantized.fc1, quantized.fc2),
                (model.features[0], model.classifier[1], model.classifier[3]),
            ):
                target.weight.copy_(source.weight)
                target.bias.copy_(source.bias)
        return quantized.eval()


def calibrate_brevitas(model, loader):
    """Brevitas collects percentile statistics and initializes learned scales."""
    samples = 0
    model.eval()
    with torch.no_grad(), calibration_mode(model):
        for images, _ in loader:
            model(images)
            samples += len(images)
    if not samples:
        raise ValueError("Calibration dataset is empty")
    model.eval()
    # Initialize the library's MSE weight scales and cached bias scales.
    with torch.no_grad():
        model(torch.zeros(1, 1, 28, 28))
    return samples


def fraction_bits(scale):
    values = scale.detach().cpu().numpy().reshape(-1)
    if len(values) != 1 or not np.isfinite(values[0]) or values[0] <= 0:
        raise ValueError("Hardware export requires a finite positive per-tensor scale")
    exponent = -math.log2(float(values[0]))
    if exponent != round(exponent) or not 0 <= exponent <= 30:
        raise ValueError(
            "Hardware export requires a power-of-two scale with F in [0, 30]"
        )
    return int(exponent)


def integer_codes(tensor: Tensor):
    """Extract already-quantized codes, also for float64 QuantTensors.

    Brevitas 0.12.1's QuantTensor.int() tolerance table omits float64.
    Validate the integer grid explicitly instead of changing library internals.
    """
    codes = tensor.value.detach() / tensor.scale.detach() + tensor.zero_point.detach()
    rounded = torch.round(codes)
    if not torch.isfinite(codes).all() or not torch.allclose(
        codes, rounded, rtol=0, atol=1e-6
    ):
        raise ValueError("Brevitas output is not on its declared integer grid")
    bits = int(tensor.bit_width.item())
    if (
        not tensor.signed
        or torch.any(rounded < -(2 ** (bits - 1)))
        or torch.any(rounded >= 2 ** (bits - 1))
    ):
        raise ValueError("Brevitas output exceeds its signed integer range")
    return rounded.to(torch.int64).cpu().numpy()


def export_fixed_model(model):
    """将已校准或完成 QAT 的 Brevitas 模型转换为整数推理模型。

    直接使用库内已有的量化参数和尺度，不在这里重新校准或选择尺度。
    返回 FixedPointCNN；调用方再通过其 save() 方法写入参数文件。
    下文的 input_bits、weight_bits、output_bits 均表示小数位数 F，
    不是存储位宽；定点数的实际值为 integer * 2**(-F)。
    """
    # 1. 切换到推理模式，使用校准/QAT 后的尺度；导出时不构建梯度图。
    model.eval()
    # arrays 保存各层整数权重/偏置；scales 保存各层的 (Fx, Fw, Fy)。
    arrays, scales = {}, {}
    with torch.no_grad():
        # 2. 用一张零输入完成前向，刷新各层偏置量化所需的尺度缓存。
        # 偏置尺度依赖输入尺度与权重尺度，不能只读取偏置参数就确定。
        # 这里没有进入 calibration_mode，零输入不用于重新统计激活范围。
        model(torch.zeros(1, 1, 28, 28))

        # 3. 读取输入量化器的 scale，并通过 F = -log2(scale) 得到小数位。
        # fraction_bits() 同时检查尺度为单个正数、2 的整数次幂且 F 在 0..30。
        # 当前输入编码约定 F=14，即像素归一化后的 1.0 对应整数 16384。
        input_bits = fraction_bits(model.input_quant.act_quant.scale())
        if input_bits != 14:
            raise ValueError(
                "Input quantizer must retain F=14 for this hardware format"
            )

        # 4. 按前向顺序处理卷积和两层全连接，以便逐层传递输入尺度。
        for name in ("conv", "fc1", "fc2"):
            layer = getattr(model, name)
            # 调用 Brevitas 的量化接口得到 QuantTensor，而非读取原始浮点参数。
            # QuantTensor 携带量化后还原的数值、scale、zero_point 和 bit_width。
            weight, bias = layer.quant_weight(), layer.quant_bias()
            weight_bits = fraction_bits(weight.scale)
            output_bits = fraction_bits(layer.output_quant.scale())

            # 5. 核对硬件格式：零点必须为 0，避免引入 RTL 未实现的零点补偿；
            # 权重为 signed16，偏置为 signed32，与 WEIGHT/BIAS 寄存器一致。
            # 此处检查声明的位宽，后续 integer_codes() 还会检查符号和数值范围。
            if not torch.all(weight.zero_point == 0) or not torch.all(
                bias.zero_point == 0
            ):
                raise ValueError("Hardware export requires zero-point=0")
            if int(weight.bit_width.item()) != 16 or int(bias.bit_width.item()) != 32:
                raise ValueError("Hardware export requires W16/B32")

            # 6. 偏置要与乘积直接相加，必须共享乘积的尺度：
            # scale_bias = scale_input * scale_weight = 2**(-(Fx + Fw))。
            expected_bias_scale = 2.0 ** -(input_bits + weight_bits)
            if not torch.all(bias.scale == expected_bias_scale):
                raise ValueError(f"{name} bias scale does not match the accumulator")

            # 7. 单独检查原始偏置是否在量化时发生 int32 饱和。
            # 只检查量化后的 bias 会漏掉溢出，因为库可能已把它截断到边界。
            # 按硬件的“中点远离零”规则计算未饱和编码，越界则拒绝导出。
            raw_bias = layer.bias.detach() / expected_bias_scale
            rounded_bias = torch.sign(raw_bias) * torch.floor(raw_bias.abs() + 0.5)
            if torch.any(rounded_bias < -(2**31)) or torch.any(
                rounded_bias > 2**31 - 1
            ):
                raise ValueError(
                    f"{name} bias would saturate int32; adjust the quantizer before export"
                )

            # 8. 从已量化数值提取整数编码：code = value / scale + zero_point。
            # integer_codes() 先验证整数网格和范围，再转换为实际存储类型；
            # 此处 astype 负责改变存储类型，不负责选择尺度、舍入或饱和。
            arrays[name + "_weight"] = integer_codes(weight).astype(np.int16)
            arrays[name + "_bias"] = integer_codes(bias).astype(np.int32)
            scales[name] = (input_bits, weight_bits, output_bits)

            # ReLU、最大池化和展平均不改变尺度，当前层 Fy 就是下一层 Fx。
            input_bits = output_bits

    # 9. 封装整数参数和量化来源信息，方便复现及核对导出结果。
    # build_fixed_model() 根据 Fx/Fw/Fy 计算 OUT_SHIFT=Fx+Fw-Fy，
    # 生成布局与卷积寄存器配置，并通过 FixedPointCNN 校验尺度链、
    # 右移范围和累加器上界；该步骤不再执行量化，也不写文件。
    return build_fixed_model(
        arrays,
        scales,
        {
            "library": "brevitas",
            "version": brevitas.__version__,
            "weight_quantizer": "Int8WeightPerTensorFixedPointMSE overridden to 16 bits",
            "activation_quantizer": "Int8ActPerTensorFixedPoint overridden to 16 bits",
            "activation_percentile": 99.99,
            "scale_restriction": "power_of_two",
            "rounding_adapter": "HardwareRoundSTE",
            "simulation_dtype": "float64",
        },
    )


def save_brevitas(path, model, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "brevitas_version": brevitas.__version__,
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_brevitas(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Unsupported Brevitas checkpoint architecture")
    model = BrevitasDigitCNN()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), checkpoint["metadata"]
