"""Train on MNIST, select by validation accuracy, evaluate the test set once."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import DigitCNN, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parent


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return {"loss": total_loss / count, "accuracy": correct / count, "samples": count}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best.pt")
    parser.add_argument(
        "--download", action="store_true", help="Download missing MNIST files"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Only evaluate the saved checkpoint"
    )
    parser.add_argument("--epochs", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if not 0 < args.learning_rate < float("inf"):
        parser.error("--learning-rate must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; the default uv environment uses CPU PyTorch")

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    transform = transforms.ToTensor()  # MNIST: bright digit, dark background, [0, 1].
    test_data = datasets.MNIST(
        args.data_dir, train=False, transform=transform, download=args.download
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    if args.evaluate:
        model, _ = load_checkpoint(args.checkpoint, args.device)
        print(
            json.dumps({"test": run_epoch(model, test_loader, args.device)}, indent=2)
        )
        return

    full_train = datasets.MNIST(
        args.data_dir, train=True, transform=transform, download=args.download
    )
    train_data, validation_data = random_split(
        full_train, [55000, 5000], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    model = DigitCNN().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_accuracy = -1.0
    history = []
    print(
        f"Device: {args.device}; parameters: {sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args.device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, args.device)
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs}: loss={train_metrics['loss']:.4f}, "
            f"train={train_metrics['accuracy']:.2%}, validation={validation_metrics['accuracy']:.2%}",
            flush=True,
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            save_checkpoint(
                args.checkpoint,
                model,
                epoch=epoch,
                seed=args.seed,
                validation=validation_metrics,
                input_scale="uint8 / 255",
            )

    model, metadata = load_checkpoint(args.checkpoint, args.device)
    test_metrics = run_epoch(model, test_loader, args.device)
    report = {
        "best_epoch": metadata["epoch"],
        "seed": args.seed,
        "history": history,
        "test": test_metrics,
    }
    report_path = args.checkpoint.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Best epoch: {metadata['epoch']}; test accuracy: {test_metrics['accuracy']:.2%}"
    )
    print(f"Checkpoint: {args.checkpoint}\nMetrics: {report_path}")


if __name__ == "__main__":
    main()
