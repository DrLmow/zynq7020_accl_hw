"""Recognize a single handwritten digit in an image."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from model import load_checkpoint

ROOT = Path(__file__).resolve().parent


def preprocess(image, polarity="auto", mnist_layout=False):
    """Return [1, 1, 28, 28], with white foreground on black in [0, 1].

    Arbitrary images are cropped, fitted into a 20x20 box, then centered by
    intensity. mnist_layout preserves already prepared 28x28 MNIST pixels.
    """
    image = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    image = Image.alpha_composite(background, image).convert("L")
    pixels = np.asarray(image, dtype=np.uint8)
    if polarity not in {"auto", "light", "dark"}:
        raise ValueError("polarity must be auto, light, or dark")
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if polarity == "dark" or (polarity == "auto" and np.median(border) > 127):
        image = ImageOps.invert(image)
        pixels = np.asarray(image, dtype=np.uint8)
    if mnist_layout:
        if image.size != (28, 28):
            raise ValueError("--mnist-layout requires a 28x28 image")
    else:
        ys, xs = np.nonzero(pixels > 20)
        if len(xs) == 0:
            raise ValueError("No foreground digit found in image")
        image = image.crop(
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        )
        width, height = image.size
        scale = 20 / max(width, height)
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new("L", (28, 28))
        canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
        values = np.asarray(canvas, dtype=np.float32)
        mass = values.sum()
        if mass == 0:
            raise ValueError("No foreground digit remains after resizing")
        yy, xx = np.indices(values.shape)
        dx = round(13.5 - float((values * xx).sum() / mass))
        dy = round(13.5 - float((values * yy).sum() / mass))
        # Paste rather than roll: pixels leaving the canvas must not wrap around.
        image = Image.new("L", (28, 28))
        image.paste(canvas, (dx, dy))
    values = np.array(image, dtype=np.float32) / 255.0
    if not values.any():
        raise ValueError("No foreground digit found in image")
    return torch.from_numpy(values).unsqueeze(0).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best.pt")
    models.add_argument("--int16-model", type=Path, help="Exported model_int16.npz")
    parser.add_argument(
        "--polarity",
        choices=["auto", "light", "dark"],
        default="auto",
        help="Foreground color; auto estimates background from the border",
    )
    parser.add_argument(
        "--mnist-layout", action="store_true", help="Preserve prepared 28x28 geometry"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    with Image.open(args.image) as image:
        inputs = preprocess(image, args.polarity, args.mnist_layout)
    if args.int16_model:
        from fixed_point import FixedPointCNN

        model = FixedPointCNN.load(args.int16_model)
        integer_logits = model.forward_int16(model.encode_input(inputs.numpy()))
        # Only the display probabilities use floating point; classification uses
        # the fixed-point network's logits (all classes share the same scale).
        probabilities = torch.from_numpy(
            model.dequantize_logits(integer_logits)
        ).softmax(dim=1)[0]
    else:
        model, _ = load_checkpoint(args.checkpoint)
        with torch.inference_mode():
            probabilities = model(inputs).softmax(dim=1)[0]
    digit = int(probabilities.argmax())
    print(
        json.dumps(
            {
                "digit": digit,
                "confidence": float(probabilities[digit]),
                "probabilities": {
                    str(i): float(value) for i, value in enumerate(probabilities)
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
