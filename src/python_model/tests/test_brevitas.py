import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from brevitas_model import (
    BrevitasDigitCNN, HardwareRoundSTE, calibrate_brevitas, export_fixed_model,
    integer_codes, load_brevitas, save_brevitas,
)
from model import DigitCNN
from quantize import fine_tune


@pytest.fixture
def model_and_data():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    images = torch.rand(8, 1, 28, 28)
    images[0] = 0
    images[1] = 1
    data = TensorDataset(images, torch.arange(8))
    loader = DataLoader(data, batch_size=4)
    model = BrevitasDigitCNN.from_float(DigitCNN())
    assert calibrate_brevitas(model, loader) == 8
    return model, loader


def test_hardware_rounding_has_ste_gradients():
    values = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], requires_grad=True)
    rounded = HardwareRoundSTE()(values)
    torch.testing.assert_close(rounded, torch.tensor([-3., -2., -1., 1., 2., 3.]))
    rounded.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones(6))


def test_export_matches_brevitas_codes_for_all_logits(model_and_data):
    model, loader = model_and_data
    exported = export_fixed_model(model)
    assert exported.metadata["quantizer"]["library"] == "brevitas"
    with torch.no_grad():
        for images, _ in loader:
            expected = integer_codes(model.quant_forward(images))
            actual = exported.forward_int16(exported.encode_input(images.numpy()))
            np.testing.assert_array_equal(actual, expected)
        for name in ("conv", "fc1", "fc2"):
            layer = getattr(model, name)
            np.testing.assert_array_equal(exported.arrays[name + "_weight"], integer_codes(layer.quant_weight()))
            np.testing.assert_array_equal(exported.arrays[name + "_bias"], integer_codes(layer.quant_bias()))


def test_library_checkpoint_retains_weights_and_calibrated_scales(model_and_data, tmp_path):
    model, loader = model_and_data
    images, _ = next(iter(loader))
    with torch.no_grad():
        before = model(images)
    path = tmp_path / "brevitas.pt"
    save_brevitas(path, model, purpose="roundtrip")
    loaded, metadata = load_brevitas(path)
    with torch.no_grad():
        torch.testing.assert_close(loaded(images), before, rtol=0, atol=0)
    assert metadata == {"purpose": "roundtrip"}
    assert export_fixed_model(model).metadata == export_fixed_model(loaded).metadata


def test_qat_updates_weights_and_scales_and_preserves_export_equivalence(model_and_data):
    model, loader = model_and_data
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    images, labels = next(iter(loader))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    for _ in range(2):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
    assert not torch.equal(model.fc2.weight, before["fc2.weight"])
    scale_parameters = [(name, value) for name, value in model.named_parameters() if "scaling_impl.value" in name]
    assert scale_parameters
    assert any(not torch.equal(value, before[name]) for name, value in scale_parameters)
    model.eval()
    exported = export_fixed_model(model)
    with torch.no_grad():
        np.testing.assert_array_equal(
            exported.forward_int16(exported.encode_input(images.numpy())),
            integer_codes(model.quant_forward(images)),
        )


def test_qat_training_loop_uses_validation_and_restores_best(model_and_data):
    model, loader = model_and_data
    report = fine_tune(model, loader, loader, epochs=1, learning_rate=1e-4)
    assert len(report["history"]) == 2
    assert report["best_epoch"] in (0, 1)
    assert not model.training
    export_fixed_model(model)
