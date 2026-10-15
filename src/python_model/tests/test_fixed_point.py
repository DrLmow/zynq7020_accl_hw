import copy
import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from brevitas_model import BrevitasDigitCNN, calibrate_brevitas, export_fixed_model
from fixed_point import (
    FixedPointCNN, conv_accumulate,
    quantize_values, requantize,
)
from model import DigitCNN


def scalar_quantize(value, shift):
    value = int(value)
    if shift:
        magnitude = (abs(value) + 2 ** (shift - 1)) // 2**shift
        value = -magnitude if value < 0 else magnitude
    return max(-32768, min(32767, value))


@pytest.mark.parametrize("shift", [0, 1, 17, 31])
def test_requantization_matches_scalar_rounding_and_saturation(shift):
    values = np.array([
        -(2**39), -100000, -65537, -65536, -32768, -7, -5, -3, -1,
        0, 1, 3, 5, 7, 32767, 65534, 65535, 100000, 2**39 - 1,
        -(2**30), 2**30,
    ], dtype=np.int64)
    expected = np.array([scalar_quantize(v, shift) for v in values], dtype=np.int16)
    np.testing.assert_array_equal(requantize(values, shift), expected)


def test_float_encoding_uses_ties_away_and_saturates():
    values = [-40000, -32768.5, -1.5, -0.5, 0, 0.5, 1.5, 32767.5, 40000]
    expected = [-32768, -32768, -2, -1, 0, 1, 2, 32767, 32767]
    np.testing.assert_array_equal(quantize_values(values, 0), np.array(expected, dtype=np.int16))
    np.testing.assert_array_equal(quantize_values([0, 0.5, 1], 14), [0, 8192, 16384])
    with pytest.raises(ValueError, match="non-finite"):
        quantize_values([float("nan")], 10)


@pytest.fixture
def fixed_model():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    quantized = BrevitasDigitCNN.from_float(DigitCNN())
    data = TensorDataset(torch.rand(8, 1, 28, 28), torch.zeros(8, dtype=torch.long))
    calibrate_brevitas(quantized, DataLoader(data, batch_size=4))
    return export_fixed_model(quantized)


def test_entire_integer_network_matches_independent_reference(fixed_model):
    # Float64 represents these bounded integer MACs exactly (< 2**53).
    # Independent scalar rounding checks scales, ReLU, pooling and flatten order.
    rng = np.random.default_rng(42)
    images = rng.integers(0, 16385, (2, 1, 28, 28), dtype=np.int16)
    actual = fixed_model.forward_int16(images)
    reference = torch.from_numpy(images.astype(np.float64))
    for name in ("conv", "fc1", "fc2"):
        weight = torch.from_numpy(fixed_model.arrays[name + "_weight"].astype(np.float64))
        bias = torch.from_numpy(fixed_model.arrays[name + "_bias"].astype(np.float64))
        reference = F.conv2d(reference, weight, bias) if name == "conv" else F.linear(reference, weight, bias)
        values = reference.numpy()
        shift = fixed_model.metadata["layers"][name]["out_shift"]
        rounded = np.array([scalar_quantize(v, shift) for v in values.flat], dtype=np.int16).reshape(values.shape)
        reference = torch.from_numpy(rounded.astype(np.float64))
        if name == "conv":
            reference = F.max_pool2d(F.relu(reference), 2).flatten(1)
        elif name == "fc1":
            reference = F.relu(reference)
    assert actual.dtype == np.int16
    np.testing.assert_array_equal(actual, reference.numpy().astype(np.int16))


def test_export_roundtrip_and_raw_little_endian_parameters(fixed_model, tmp_path):
    fixed_model.save(tmp_path)
    loaded = FixedPointCNN.load(tmp_path / "model_int16.npz")
    image = fixed_model.encode_input(np.ones((1, 1, 28, 28)))
    np.testing.assert_array_equal(loaded.forward_int16(image), fixed_model.forward_int16(image))
    assert loaded.metadata == fixed_model.metadata
    for name, values in fixed_model.arrays.items():
        dtype = "<i2" if name.endswith("_weight") else "<i4"
        raw = np.fromfile(tmp_path / f"{name}.bin", dtype=dtype)
        np.testing.assert_array_equal(raw.reshape(values.shape), values)


def test_export_rejects_bias_overflow():
    model = BrevitasDigitCNN.from_float(DigitCNN())
    data = TensorDataset(torch.rand(8, 1, 28, 28), torch.zeros(8, dtype=torch.long))
    calibrate_brevitas(model, DataLoader(data, batch_size=4))
    with torch.no_grad():
        model.conv.bias.fill_(1e10)
    with pytest.raises(ValueError, match="saturate int32"):
        export_fixed_model(model)


def test_invalid_scales_and_inputs_are_rejected(fixed_model):
    metadata = copy.deepcopy(fixed_model.metadata)
    metadata["layers"]["conv"]["out_shift"] += 1
    with pytest.raises(ValueError, match="scales"):
        FixedPointCNN(metadata, fixed_model.arrays)
    with pytest.raises(ValueError, match="int16"):
        fixed_model.forward_int16(np.zeros((1, 1, 28, 28), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fixed_model.encode_input(np.array([1.1]))
    with pytest.raises(ValueError, match="OUT_SHIFT"):
        requantize(np.array([1], dtype=np.int64), 32)


def test_saturation_counts_clipping_only():
    stats = {}
    requantize(np.array([-32769, -32768, 32767, 32768], dtype=np.int64), 0, stats, "conv")
    assert stats == {"conv": {"clipped": 2, "total": 4}}


class ConvConfig(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "width", "height", "src_stride", "dst_stride", "kernel_size", "mode16",
        "round_enable", "shift",
    )] + [("bias", ctypes.c_int32), ("weights", ctypes.c_int16 * 25)]


@pytest.fixture(scope="module")
def c_convolution(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler needed for existing hardware-reference cross-check")
    source = Path(__file__).resolve().parents[3] / "app_component/src/conv_cpu.c"
    library = tmp_path_factory.mktemp("conv_reference") / "conv_cpu.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-O2", str(source), "-o", str(library)], check=True)
    reference = ctypes.CDLL(str(library))
    reference.conv_cpu_run.argtypes = [ctypes.POINTER(ConvConfig), ctypes.c_void_p, ctypes.c_void_p]
    reference.conv_cpu_run.restype = None
    return reference.conv_cpu_run


@pytest.mark.parametrize("kernel", [3, 5])
@pytest.mark.parametrize("shift", [0, 1, 17, 31])
def test_convolution_matches_existing_c_hardware_reference(c_convolution, kernel, shift):
    rng = np.random.default_rng(123 + kernel)
    width, height = 11, 9
    images = rng.integers(-32768, 32768, (1, 1, height, width), dtype=np.int16)
    weights = rng.integers(-32768, 32768, (3, 1, kernel, kernel), dtype=np.int16)
    bias = np.array([-(2**31), 12345, 2**31 - 1], dtype=np.int32)
    expected = requantize(conv_accumulate(images, weights, bias), shift)[0]
    ow, oh = width - kernel + 1, height - kernel + 1
    src_stride = ((width * 2 + 7) // 8) * 8
    dst_stride = ((ow * 2 + 7) // 8) * 8
    source = np.zeros((height, src_stride // 2), dtype=np.int16)
    source[:, :width] = images[0, 0]
    for channel in range(3):
        output = np.full((oh, dst_stride // 2), 0x1234, dtype=np.int16)
        config = ConvConfig(width, height, src_stride, dst_stride, kernel, 1, 1, shift, int(bias[channel]))
        for i, value in enumerate(weights[channel].flat):
            config.weights[i] = int(value)
        c_convolution(ctypes.byref(config), source.ctypes.data, output.ctypes.data)
        np.testing.assert_array_equal(output[:, :ow], expected[channel])
        assert np.all(output[:, ow:] == 0x1234)
