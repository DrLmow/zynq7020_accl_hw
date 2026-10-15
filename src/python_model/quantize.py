"""Brevitas W16A16 PTQ/QAT, hardware export and integer equivalence checks."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from brevitas_model import (
    BrevitasDigitCNN,
    calibrate_brevitas,
    export_fixed_model,
    integer_codes,
    save_brevitas,
)
from fixed_point import FixedPointCNN
from model import load_checkpoint
from train import positive_int, run_epoch

ROOT = Path(__file__).resolve().parent


def evaluate(float_model, fixed_model, loader, brevitas_model):
    count = float_correct = fixed_correct = agreement = 0
    absolute_error = 0.0
    maximum_error = 0.0
    saturation = {}
    library_correct = library_mismatches = 0
    with torch.inference_mode():
        for images, labels in loader:
            floating = float_model(images).numpy()
            integer = fixed_model.forward_int16(
                fixed_model.encode_input(images.numpy()), saturation
            )
            library_integer = integer_codes(brevitas_model.quant_forward(images))
            library_mismatches += int(np.count_nonzero(library_integer != integer))
            library_correct += int(
                np.count_nonzero(library_integer.argmax(1) == labels.numpy())
            )
            restored = fixed_model.dequantize_logits(integer)
            labels = labels.numpy()
            float_predictions = floating.argmax(1)
            fixed_predictions = integer.argmax(1)
            float_correct += int(np.count_nonzero(float_predictions == labels))
            fixed_correct += int(np.count_nonzero(fixed_predictions == labels))
            agreement += int(np.count_nonzero(float_predictions == fixed_predictions))
            error = np.abs(floating - restored)
            absolute_error += float(error.sum())
            maximum_error = max(maximum_error, float(error.max()))
            count += len(labels)
    if count == 0:
        raise ValueError("Evaluation dataset is empty")
    for entry in saturation.values():
        entry["fraction"] = entry["clipped"] / entry["total"]
    return {
        "samples": count,
        "float_accuracy": float_correct / count,
        "int16_accuracy": fixed_correct / count,
        "accuracy_change_percentage_points": 100
        * (fixed_correct - float_correct)
        / count,
        "prediction_agreement": agreement / count,
        "logit_mean_absolute_error": absolute_error / (count * 10),
        "logit_max_absolute_error": maximum_error,
        "saturation": saturation,
        "brevitas_accuracy": library_correct / count,
        "brevitas_integer_logit_mismatches": library_mismatches,
    }


def fine_tune(model, train_loader, validation_loader, epochs, learning_rate):
    """QAT with Brevitas STEs/learned scales; select using validation only."""
    baseline = run_epoch(model, validation_loader, "cpu")
    best_accuracy, best_epoch = baseline["accuracy"], 0
    best_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = [{"epoch": 0, "validation": baseline}]
    for epoch in range(1, epochs + 1):
        training = run_epoch(model, train_loader, "cpu", optimizer)
        validation = run_epoch(model, validation_loader, "cpu")
        history.append({"epoch": epoch, "train": training, "validation": validation})
        print(
            f"QAT {epoch}/{epochs}: train={training['accuracy']:.2%}, "
            f"validation={validation['accuracy']:.2%}",
            flush=True,
        )
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/int16")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--calibration-samples", type=positive_int, default=1024)
    parser.add_argument(
        "--qat-epochs", type=int, default=0, help="0 = PTQ; >0 = QAT fine-tuning epochs"
    )
    parser.add_argument("--qat-learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument(
        "--seed", type=int, default=42, help="Calibration sampling seed"
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.qat_epochs < 0:
        parser.error("--qat-epochs must be >= 0")
    if not np.isfinite(args.qat_learning_rate) or args.qat_learning_rate <= 0:
        parser.error("--qat-learning-rate must be finite and positive")
    if args.calibration_samples > 55000:
        parser.error("--calibration-samples cannot exceed the 55,000 training images")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, source_metadata = load_checkpoint(args.checkpoint)
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    train = datasets.MNIST(
        args.data_dir,
        train=True,
        transform=transforms.ToTensor(),
        download=args.download,
    )
    # Reproduce the checkpoint's training partition; neither validation nor test
    # images are used to select fixed-point ranges.
    split_seed = source_metadata.get("seed", 42)
    train, validation = random_split(
        train, [55000, 5000], generator=torch.Generator().manual_seed(split_seed)
    )
    indices = torch.randperm(
        len(train), generator=torch.Generator().manual_seed(args.seed)
    )
    calibration_data = Subset(train, indices[: args.calibration_samples].tolist())
    quantized = BrevitasDigitCNN.from_float(model)
    samples = calibrate_brevitas(
        quantized, DataLoader(calibration_data, batch_size=args.batch_size)
    )
    print(
        f"Brevitas W16A16 calibration complete: {samples} training images", flush=True
    )
    qat = None
    if args.qat_epochs:
        qat = fine_tune(
            quantized,
            DataLoader(
                train,
                batch_size=args.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(args.seed),
            ),
            DataLoader(validation, batch_size=args.batch_size),
            args.qat_epochs,
            args.qat_learning_rate,
        )
    library_path = args.output_dir / "brevitas.pt"
    save_brevitas(library_path, quantized, source_sha256=source_hash, qat=qat)
    fixed = export_fixed_model(quantized)
    fixed.metadata["source"] = {
        "checkpoint": args.checkpoint.name,
        "sha256": source_hash,
        "training_metadata": source_metadata,
    }
    fixed.metadata["calibration"] = {
        "dataset": "MNIST training partition",
        "samples": samples,
        "sampling_seed": args.seed,
        "training_split_seed": split_seed,
    }
    fixed.metadata["qat"] = qat
    fixed.save(args.output_dir)
    # Evaluate the exported/reloaded artifact, rather than only the in-memory copy.
    fixed = FixedPointCNN.load(args.output_dir / "model_int16.npz")
    print(f"Exported int16 model to {args.output_dir}", flush=True)
    for name, config in fixed.metadata["layers"].items():
        print(
            f"{name}: Fx={config['input_fraction_bits']}, Fw={config['weight_fraction_bits']}, "
            f"Fy={config['output_fraction_bits']}, OUT_SHIFT={config['out_shift']}",
            flush=True,
        )
    test = datasets.MNIST(
        args.data_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=args.download,
    )
    report = evaluate(
        model, fixed, DataLoader(test, batch_size=args.batch_size), quantized
    )
    report["checkpoint_sha256"] = source_hash
    report["calibration"] = fixed.metadata["calibration"]
    report["quantizer"] = fixed.metadata["quantizer"]
    report["qat"] = qat
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if report["brevitas_integer_logit_mismatches"]:
        raise RuntimeError(
            "Brevitas and integer reference logits differ; see metrics.json"
        )


if __name__ == "__main__":
    main()
