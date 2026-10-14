"""Small floating-point MNIST reference network."""

from pathlib import Path

import torch
from torch import nn

MODEL_NAME = "digit-cnn-v1"


class DigitCNN(nn.Module):
    """[N, 1, 28, 28] pixels in [0, 1] -> [N, 10] logits.

    Each 5x5 filter operates on one input channel, so its VALID correlation
    can later be mapped to one accelerator job after fixed-point conversion.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 12 * 12, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def save_checkpoint(path, model, **metadata):
    """Save CPU weights and plain metadata; no pickled model object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": MODEL_NAME,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != MODEL_NAME:
        raise ValueError("Checkpoint architecture does not match DigitCNN")
    model = DigitCNN()
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint.get("metadata", {})
