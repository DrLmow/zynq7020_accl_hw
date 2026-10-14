import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, TensorDataset

from model import DigitCNN, load_checkpoint, save_checkpoint
from predict import preprocess
from train import run_epoch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(123)


def test_network_learns_and_checkpoint_preserves_predictions(tmp_path):
    # Distinct spatial patterns exercise convolution, gradients and the classifier.
    images = torch.zeros(16, 1, 28, 28)
    images[:8, :, 3:10, 3:10] = 1
    images[8:, :, 18:25, 18:25] = 1
    labels = torch.tensor([0] * 8 + [1] * 8)
    loader = DataLoader(TensorDataset(images, labels), batch_size=16)
    model = DigitCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = run_epoch(model, loader, "cpu")
    for _ in range(12):
        run_epoch(model, loader, "cpu", optimizer)
    after = run_epoch(model, loader, "cpu")
    assert after["loss"] < before["loss"] * 0.2
    assert after["accuracy"] == 1.0

    path = tmp_path / "nested/model.pt"
    save_checkpoint(path, model, epoch=12)
    loaded, metadata = load_checkpoint(path)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(images), model(images), rtol=0, atol=0)
    assert metadata["epoch"] == 12
    assert not loaded.training


def test_checkpoint_rejects_unknown_architecture(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"architecture": "unknown"}, path)
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(path)


def test_preprocessing_handles_polarity_and_centers_digit():
    image = Image.new("L", (70, 100))
    ImageDraw.Draw(image).rectangle((5, 10, 14, 60), fill=255)
    light = preprocess(image)
    dark = preprocess(ImageOps.invert(image))
    torch.testing.assert_close(light, dark)
    assert light.shape == (1, 1, 28, 28)
    assert light.dtype == torch.float32
    assert 0 <= light.min() <= light.max() <= 1
    values = light[0, 0].numpy()
    yy, xx = np.indices((28, 28))
    assert abs((values * xx).sum() / values.sum() - 13.5) <= 0.51
    assert abs((values * yy).sum() / values.sum() - 13.5) <= 0.51


def test_mnist_layout_keeps_original_pixels():
    values = np.zeros((28, 28), dtype=np.uint8)
    values[4:20, 8:12] = 127
    actual = preprocess(Image.fromarray(values), mnist_layout=True)
    np.testing.assert_allclose(actual[0, 0].numpy(), values / 255, atol=1e-7)


@pytest.mark.parametrize("color", [0, 255])
def test_blank_images_are_rejected(color):
    with pytest.raises(ValueError, match="foreground"):
        preprocess(Image.new("L", (40, 40), color))


def test_mnist_layout_requires_correct_size():
    with pytest.raises(ValueError, match="28x28"):
        preprocess(Image.new("L", (32, 32)), mnist_layout=True)


def test_transparent_image_matches_white_background():
    transparent = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    ImageDraw.Draw(transparent).rectangle((20, 10, 25, 40), fill=(0, 0, 0, 255))
    opaque = Image.new("RGB", (50, 50), "white")
    ImageDraw.Draw(opaque).rectangle((20, 10, 25, 40), fill="black")
    torch.testing.assert_close(preprocess(transparent), preprocess(opaque))
