"""Signed int16, power-of-two fixed-point inference for DigitCNN.

Real value = stored integer * 2**(-fraction_bits), zero point = 0.
All MACs use int64 here. The convolution fits the RTL's signed 40-bit MAC;
fully connected layers require int64 accumulators on the PS.
"""

import json
from pathlib import Path

import numpy as np

FORMAT = "digit-cnn-int16-v1"
LAYERS = ("conv", "fc1", "fc2")
SHAPES = {"conv": (8, 1, 5, 5), "fc1": (64, 1152), "fc2": (10, 64)}


def round_away(values):
    values = np.asarray(values, dtype=np.float64)
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def quantize_values(values, fraction_bits, bits=16):
    """Round before saturating; never allow narrow integer wraparound."""
    if bits not in (16, 32) or not 0 <= fraction_bits <= 60:
        raise ValueError("Invalid fixed-point format")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Cannot quantize non-finite values")
    rounded = round_away(values * (2.0**fraction_bits))
    return np.clip(rounded, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1).astype(
        np.int16 if bits == 16 else np.int32
    )


def requantize(accumulator, shift, stats=None, name=None):
    """Match RTL ROUND_EN=1: ties away from zero, then signed16 saturation."""
    if not 0 <= shift <= 31:
        raise ValueError("OUT_SHIFT must be in [0, 31]")
    values = np.asarray(accumulator)
    if values.dtype != np.int64:
        raise ValueError("Accumulators must use int64")
    # Network bounds are checked at construction, well inside int64 limits.
    if shift:
        magnitude = (np.abs(values) + (1 << (shift - 1))) >> shift
        values = np.where(values < 0, -magnitude, magnitude)
    if stats is not None:
        entry = stats.setdefault(name, {"clipped": 0, "total": 0})
        entry["clipped"] += int(np.count_nonzero((values < -32768) | (values > 32767)))
        entry["total"] += values.size
    return np.clip(values, -32768, 32767).astype(np.int16)


def conv_accumulate(images, weights, bias):
    """Single input channel VALID cross-correlation; no kernel reversal."""
    images = np.asarray(images)
    weights = np.asarray(weights)
    bias = np.asarray(bias)
    if images.dtype != np.int16 or weights.dtype != np.int16 or bias.dtype != np.int32:
        raise ValueError("Convolution requires int16 input/weight and int32 bias")
    if (
        images.ndim != 4
        or images.shape[1] != 1
        or weights.ndim != 4
        or weights.shape[1] != 1
    ):
        raise ValueError("Convolution requires NCHW with one input channel")
    kernel = weights.shape[-1]
    if (
        kernel not in (3, 5)
        or weights.shape[-2] != kernel
        or bias.shape != (len(weights),)
    ):
        raise ValueError(
            "Expected square 3x3/5x5 kernels and one bias per output channel"
        )
    windows = np.lib.stride_tricks.sliding_window_view(
        images[:, 0], (kernel, kernel), axis=(1, 2)
    )
    batch, height, width = windows.shape[:3]
    patches = windows.reshape(batch, height * width, kernel * kernel).astype(np.int64)
    kernels = weights.reshape(len(weights), kernel * kernel).astype(np.int64)
    result = patches @ kernels.T + bias.astype(np.int64)
    return result.transpose(0, 2, 1).reshape(batch, len(weights), height, width)


class FixedPointCNN:
    """Integer-only convolution, ReLU, pooling and dense layers on CPU."""

    def __init__(self, metadata, arrays):
        if metadata.get("format") != FORMAT:
            raise ValueError("Unsupported fixed-point model format")
        self.metadata = metadata
        self.arrays = arrays
        previous_bits = metadata["input_fraction_bits"]
        if previous_bits != 14:
            raise ValueError("Expected input fraction_bits=14")
        for name in LAYERS:
            config = metadata["layers"][name]
            weight, bias = arrays[name + "_weight"], arrays[name + "_bias"]
            if weight.dtype != np.dtype("int16") or weight.shape != SHAPES[name]:
                raise ValueError(f"Invalid {name} weight dtype/shape")
            if bias.dtype != np.dtype("int32") or bias.shape != (SHAPES[name][0],):
                raise ValueError(f"Invalid {name} bias dtype/shape")
            fw, fy = config["weight_fraction_bits"], config["output_fraction_bits"]
            if not all(isinstance(v, int) and 0 <= v <= 30 for v in (fw, fy)):
                raise ValueError(f"Invalid {name} fraction bits")
            shift = previous_bits + fw - fy
            if (
                config["input_fraction_bits"] != previous_bits
                or config["bias_fraction_bits"] != previous_bits + fw
                or config["out_shift"] != shift
                or not 0 <= shift <= 31
            ):
                raise ValueError(f"Inconsistent {name} fixed-point scales")
            bound = int(
                (
                    np.abs(weight.astype(np.int64)).reshape(len(weight), -1).sum(1)
                    * 32768
                    + np.abs(bias.astype(np.int64))
                ).max()
            )
            limit = 2**39 if name == "conv" else 2**63
            if bound >= limit:
                raise ValueError(f"{name} accumulator would overflow")
            previous_bits = fy

    @property
    def output_fraction_bits(self):
        return self.metadata["layers"]["fc2"]["output_fraction_bits"]

    def encode_input(self, images):
        images = np.asarray(images)
        if not np.isfinite(images).all() or np.any(images < 0) or np.any(images > 1):
            raise ValueError("Input pixels must be finite and within [0, 1]")
        return quantize_values(images, self.metadata["input_fraction_bits"])

    def forward_int16(self, images, stats=None):
        images = np.asarray(images)
        if (
            images.dtype != np.int16
            or images.ndim != 4
            or images.shape[1:] != (1, 28, 28)
        ):
            raise ValueError("Expected int16 input with shape [N, 1, 28, 28]")
        arrays = self.arrays
        layers = self.metadata["layers"]
        accumulator = conv_accumulate(
            images, arrays["conv_weight"], arrays["conv_bias"]
        )
        conv = requantize(accumulator, layers["conv"]["out_shift"], stats, "conv")
        # Pooling preserves the convolution scale and NCHW flatten order.
        pooled = np.maximum(conv, 0).reshape(-1, 8, 12, 2, 12, 2).max(axis=(3, 5))
        values = pooled.reshape(len(images), 1152)
        for name in ("fc1", "fc2"):
            accumulator = values.astype(np.int64) @ arrays[name + "_weight"].astype(
                np.int64
            ).T + arrays[name + "_bias"].astype(np.int64)
            values = requantize(accumulator, layers[name]["out_shift"], stats, name)
            if name == "fc1":
                values = np.maximum(values, 0)
        return values

    def dequantize_logits(self, logits):
        return np.asarray(logits, dtype=np.float64) * (2.0**-self.output_fraction_bits)

    def save(self, directory):
        """Portable NPZ plus little-endian raw parameters and readable manifest."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self.metadata, indent=2, allow_nan=False)
        np.savez(directory / "model_int16.npz", metadata=serialized, **self.arrays)
        (directory / "manifest.json").write_text(serialized + "\n", encoding="utf-8")
        for name, values in self.arrays.items():
            dtype = "<i2" if name.endswith("_weight") else "<i4"
            values.astype(dtype).tofile(directory / f"{name}.bin")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            arrays = {
                name: archive[name] for name in archive.files if name != "metadata"
            }
        return cls(metadata, arrays)


def build_fixed_model(arrays, scales, quantizer):
    """Package library-exported parameters; this function does no quantization."""
    metadata = {
        "format": FORMAT,
        "weight_bits": 16,
        "activation_bits": 16,
        "bias_bits": 32,
        "zero_point": 0,
        "rounding": "nearest_ties_away_from_zero",
        "saturation": [-32768, 32767],
        "input_fraction_bits": 14,
        "quantizer": quantizer,
        "layers": {},
    }
    for name in LAYERS:
        input_bits, weight_bits, output_bits = scales[name]
        weight, bias = arrays[name + "_weight"], arrays[name + "_bias"]
        bound = int(
            (
                np.abs(weight.astype(np.int64)).reshape(len(weight), -1).sum(1) * 32768
                + np.abs(bias.astype(np.int64))
            ).max()
        )
        metadata["layers"][name] = {
            "input_fraction_bits": input_bits,
            "weight_fraction_bits": weight_bits,
            "bias_fraction_bits": input_bits + weight_bits,
            "output_fraction_bits": output_bits,
            "out_shift": input_bits + weight_bits - output_bits,
            "accumulator_bits": 40 if name == "conv" else 64,
            "accumulator_abs_bound": bound,
            "weight_shape": list(weight.shape),
            "weight_layout": "OIHW" if name == "conv" else "OI",
            "weight_file": f"{name}_weight.bin",
            "weight_dtype": "little-endian int16",
            "bias_file": f"{name}_bias.bin",
            "bias_dtype": "little-endian int32",
        }
    metadata["conv_hardware"] = {
        "config": 7,
        "kernel_size": 5,
        "mode16": 1,
        "round_enable": 1,
        "width": 28,
        "height": 28,
        "src_stride_bytes": 56,
        "output_width": 24,
        "output_height": 24,
        "dst_stride_bytes": 48,
        "jobs": 8,
        "weight_elements_per_job": 25,
        "out_shift": metadata["layers"]["conv"]["out_shift"],
        "bias_per_job": arrays["conv_bias"].tolist(),
    }
    return FixedPointCNN(metadata, arrays)
