"""
06_evaluate_baseline.py

Final evaluation of the trained ResNet18 baseline on the unseen target domain.

Protocol
--------
1. Load the BEST checkpoint selected during source-only training.
2. Run inference on SOURCE validation data.
3. Select ONE classification threshold using SOURCE validation only.
   Current rule: pooled source-validation EER threshold.
4. Freeze that threshold.
5. Only then load and evaluate the unseen TARGET test set.

Labels
------
0 = bona_fide
1 = spoof

Metrics
-------
Threshold-free:
    AUC

At the frozen source-validation threshold:
    APCER
    BPCER
    ACER
    HTER
    balanced accuracy
    accuracy
    confusion matrix counts

IMPORTANT
---------
The target test set is never used to:
- train the model;
- select the checkpoint;
- tune hyperparameters;
- select the classification threshold.

This script intentionally creates the target test dataset only AFTER the
source-validation threshold has been selected.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.metrics import roc_auc_score, roc_curve

from fas_dataset import (
    FASDataset,
    find_tese_root,
    experiment_csv_for_target,
)


DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select threshold on source validation and evaluate "
            "the ResNet18 baseline on the unseen target."
        )
    )

    parser.add_argument(
        "--target",
        required=True,
        choices=["nuaa", "lcc_fasd", "casia_fasd"],
        help="Unseen target domain.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="PyTorch DataLoader workers. Keep 0 initially on Windows.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional explicit checkpoint path. "
            "Default: checkpoints/baseline/target_<target>/best_resnet18.pt"
        ),
    )

    return parser.parse_args()


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def build_model():
    """
    Recreate exactly the ResNet18 architecture used during training.

    Pretrained ImageNet weights are NOT downloaded here because the trained
    checkpoint replaces the model parameters immediately afterwards.
    """
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = build_model().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device,
):
    """
    Return labels, spoof probabilities, domain ids and sample metadata.

    Class 1 = spoof, therefore scores[:, 1] is the score used for ROC/AUC
    and threshold-based decisions.
    """
    all_labels = []
    all_scores = []
    all_domain_ids = []
    all_sample_ids = []
    all_datasets = []
    all_native_splits = []
    all_attack_types = []
    all_qualities = []
    all_subject_ids = []

    use_amp = device.type == "cuda"

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        if use_amp:
            with torch.amp.autocast(
                device_type="cuda",
                enabled=True,
            ):
                logits = model(images)
        else:
            logits = model(images)

        probabilities = torch.softmax(
            logits.float(),
            dim=1,
        )

        spoof_scores = probabilities[:, 1]

        all_labels.extend(
            batch["label"].cpu().numpy().astype(int).tolist()
        )
        all_scores.extend(
            spoof_scores.cpu().numpy().astype(float).tolist()
        )
        all_domain_ids.extend(
            batch["domain_id"].cpu().numpy().astype(int).tolist()
        )

        all_sample_ids.extend(list(batch["sample_id"]))
        all_datasets.extend(list(batch["dataset"]))
        all_native_splits.extend(list(batch["native_split"]))
        all_attack_types.extend(list(batch["attack_type"]))
        all_qualities.extend(list(batch["quality"]))
        all_subject_ids.extend(list(batch["subject_id"]))

    return {
        "labels": np.asarray(all_labels, dtype=np.int64),
        "scores": np.asarray(all_scores, dtype=np.float64),
        "domain_ids": np.asarray(all_domain_ids, dtype=np.int64),
        "sample_ids": all_sample_ids,
        "datasets": all_datasets,
        "native_splits": all_native_splits,
        "attack_types": all_attack_types,
        "qualities": all_qualities,
        "subject_ids": all_subject_ids,
    }


def select_eer_threshold(labels: np.ndarray, scores: np.ndarray):
    """
    Select threshold from SOURCE validation only.

    With 1=spoof:
      FPR = bona_fide -> spoof error = BPCER
      FNR = spoof -> bona_fide error = APCER

    EER threshold is approximated by the ROC threshold where |FPR - FNR|
    is minimal.
    """
    if len(np.unique(labels)) < 2:
        raise RuntimeError(
            "São necessárias ambas as classes para calcular EER."
        )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    finite_mask = np.isfinite(thresholds)

    if not finite_mask.any():
        raise RuntimeError(
            "Não foi encontrado nenhum threshold ROC finito."
        )

    fpr_f = fpr[finite_mask]
    fnr_f = fnr[finite_mask]
    thresholds_f = thresholds[finite_mask]

    idx = int(
        np.argmin(
            np.abs(fpr_f - fnr_f)
        )
    )

    threshold = float(thresholds_f[idx])
    eer = float(
        (fpr_f[idx] + fnr_f[idx]) / 2.0
    )

    return threshold, eer


def threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
):
    """
    Prediction rule:
        score >= threshold -> spoof (1)
        score <  threshold -> bona_fide (0)
    """
    predictions = (
        scores >= threshold
    ).astype(np.int64)

    bona_mask = labels == 0
    spoof_mask = labels == 1

    if not bona_mask.any() or not spoof_mask.any():
        raise RuntimeError(
            "APCER/BPCER requerem ambas as classes."
        )

    # Confusion matrix for positive class = spoof.
    tn = int(
        np.sum(
            (labels == 0) & (predictions == 0)
        )
    )
    fp = int(
        np.sum(
            (labels == 0) & (predictions == 1)
        )
    )
    fn = int(
        np.sum(
            (labels == 1) & (predictions == 0)
        )
    )
    tp = int(
        np.sum(
            (labels == 1) & (predictions == 1)
        )
    )

    # ISO PAD terminology:
    # APCER = attacks incorrectly classified as bona fide.
    # BPCER = bona fide incorrectly classified as attacks.
    apcer = fn / (tp + fn)
    bpcer = fp / (tn + fp)
    acer = (apcer + bpcer) / 2.0

    # Under this binary protocol, HTER uses the same two class-error rates.
    hter = acer

    tpr_spoof = tp / (tp + fn)
    tnr_bona = tn / (tn + fp)

    balanced_accuracy = (
        tpr_spoof + tnr_bona
    ) / 2.0

    accuracy = (
        tp + tn
    ) / len(labels)

    auc = roc_auc_score(
        labels,
        scores,
    )

    return {
        "auc": float(auc),
        "threshold": float(threshold),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "acer": float(acer),
        "hter": float(hter),
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float(accuracy),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "samples": int(len(labels)),
        "bona_fide_samples": int(np.sum(bona_mask)),
        "spoof_samples": int(np.sum(spoof_mask)),
    }


def metrics_by_domain(
    labels,
    scores,
    domain_ids,
    threshold,
):
    result = {}

    for domain_id in sorted(
        np.unique(domain_ids).tolist()
    ):
        mask = domain_ids == domain_id

        result[str(int(domain_id))] = (
            threshold_metrics(
                labels[mask],
                scores[mask],
                threshold,
            )
        )

    return result


def write_predictions_csv(
    path: Path,
    prediction_data: dict,
    threshold: float,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = prediction_data["labels"]
    scores = prediction_data["scores"]

    predictions = (
        scores >= threshold
    ).astype(np.int64)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "sample_id",
            "dataset",
            "native_split",
            "domain_id",
            "subject_id",
            "attack_type",
            "quality",
            "label",
            "spoof_probability",
            "threshold",
            "prediction",
            "correct",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for i in range(len(labels)):
            writer.writerow(
                {
                    "sample_id": prediction_data["sample_ids"][i],
                    "dataset": prediction_data["datasets"][i],
                    "native_split": prediction_data["native_splits"][i],
                    "domain_id": int(prediction_data["domain_ids"][i]),
                    "subject_id": prediction_data["subject_ids"][i],
                    "attack_type": prediction_data["attack_types"][i],
                    "quality": prediction_data["qualities"][i],
                    "label": int(labels[i]),
                    "spoof_probability": float(scores[i]),
                    "threshold": float(threshold),
                    "prediction": int(predictions[i]),
                    "correct": int(predictions[i] == labels[i]),
                }
            )


def save_json(path: Path, data: dict):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def fmt_pct(value):
    return f"{100.0 * value:.2f}%"


def main():
    args = parse_args()

    tese_root = find_tese_root()

    experiment_csv = experiment_csv_for_target(
        tese_root,
        args.target,
    )

    output_key = f"target_{args.target}"

    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint is not None
        else (
            tese_root
            / "checkpoints"
            / "baseline"
            / output_key
            / "best_resnet18.pt"
        )
    )

    results_dir = (
        tese_root
        / "results"
        / "baseline"
        / output_key
    )

    summary_path = (
        results_dir
        / "final_evaluation.json"
    )

    val_predictions_path = (
        results_dir
        / "source_validation_predictions.csv"
    )

    target_predictions_path = (
        results_dir
        / "target_test_predictions.csv"
    )

    device = get_device()

    print("=" * 80)
    print("RESNET18 BASELINE - FINAL UNSEEN TARGET EVALUATION")
    print("=" * 80)
    print("Target:", args.target)
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Threshold rule: pooled SOURCE-validation EER")
    print("Target used for threshold selection: NO")
    print("=" * 80)

    model, checkpoint = load_checkpoint_model(
        checkpoint_path,
        device,
    )

    checkpoint_target = str(
        checkpoint.get("target", "")
    )

    if checkpoint_target and checkpoint_target != args.target:
        raise RuntimeError(
            "O checkpoint foi treinado para outro target: "
            f"{checkpoint_target} != {args.target}"
        )

    # ------------------------------------------------------------------
    # STEP 1: SOURCE VALIDATION ONLY
    # ------------------------------------------------------------------
    print("\n[1/3] SOURCE validation inference...")

    val_dataset = FASDataset(
        experiment_csv,
        role="val",
        tese_root=tese_root,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_data = collect_predictions(
        model,
        val_loader,
        device,
    )

    val_auc = roc_auc_score(
        val_data["labels"],
        val_data["scores"],
    )

    threshold, source_val_eer = (
        select_eer_threshold(
            val_data["labels"],
            val_data["scores"],
        )
    )

    val_metrics = threshold_metrics(
        val_data["labels"],
        val_data["scores"],
        threshold,
    )

    val_domain_metrics = metrics_by_domain(
        val_data["labels"],
        val_data["scores"],
        val_data["domain_ids"],
        threshold,
    )

    print(
        f"Source validation pooled AUC: {val_auc:.4f}"
    )
    print(
        f"Selected source-validation EER threshold: {threshold:.6f}"
    )
    print(
        f"Approx. source-validation EER: {fmt_pct(source_val_eer)}"
    )

    write_predictions_csv(
        val_predictions_path,
        val_data,
        threshold,
    )

    # Threshold is now frozen.
    frozen_threshold = float(threshold)

    print(
        "\n[2/3] Threshold FROZEN. "
        "Only now is the unseen target test set loaded."
    )

    # ------------------------------------------------------------------
    # STEP 2: UNSEEN TARGET TEST
    # ------------------------------------------------------------------
    target_dataset = FASDataset(
        experiment_csv,
        role="test",
        tese_root=tese_root,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    target_data = collect_predictions(
        model,
        target_loader,
        device,
    )

    target_metrics = threshold_metrics(
        target_data["labels"],
        target_data["scores"],
        frozen_threshold,
    )

    write_predictions_csv(
        target_predictions_path,
        target_data,
        frozen_threshold,
    )

    # ------------------------------------------------------------------
    # STEP 3: SAVE + REPORT
    # ------------------------------------------------------------------
    summary = {
        "model": "ResNet18 baseline",
        "labels": {
            "0": "bona_fide",
            "1": "spoof",
        },
        "target": args.target,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(
            checkpoint.get("epoch", -1)
        ),
        "checkpoint_best_macro_source_val_auc": float(
            checkpoint.get(
                "best_macro_source_val_auc",
                math.nan,
            )
        ),
        "threshold_protocol": {
            "selection_data": "source validation only",
            "method": "pooled EER",
            "threshold": frozen_threshold,
            "source_validation_eer": source_val_eer,
            "target_used_for_threshold_selection": False,
        },
        "source_validation": {
            "pooled": val_metrics,
            "by_domain_id": val_domain_metrics,
        },
        "unseen_target_test": target_metrics,
        "files": {
            "source_validation_predictions": str(
                val_predictions_path
            ),
            "target_test_predictions": str(
                target_predictions_path
            ),
        },
    }

    save_json(
        summary_path,
        summary,
    )

    print("\n[3/3] FINAL UNSEEN TARGET RESULTS")
    print("=" * 80)
    print(f"Target: {args.target}")
    print(
        f"Frozen threshold: {frozen_threshold:.6f} "
        "(selected on source validation only)"
    )
    print(f"AUC:               {target_metrics['auc']:.4f}")
    print(f"APCER:             {fmt_pct(target_metrics['apcer'])}")
    print(f"BPCER:             {fmt_pct(target_metrics['bpcer'])}")
    print(f"ACER:              {fmt_pct(target_metrics['acer'])}")
    print(f"HTER:              {fmt_pct(target_metrics['hter'])}")
    print(
        "Balanced accuracy: "
        f"{fmt_pct(target_metrics['balanced_accuracy'])}"
    )
    print(
        f"Accuracy:           {fmt_pct(target_metrics['accuracy'])}"
    )
    print(
        "Confusion matrix "
        "(positive class = spoof): "
        f"TN={target_metrics['tn']} "
        f"FP={target_metrics['fp']} "
        f"FN={target_metrics['fn']} "
        f"TP={target_metrics['tp']}"
    )
    print("=" * 80)
    print("Summary:", summary_path)
    print(
        "Source validation predictions:",
        val_predictions_path,
    )
    print(
        "Target test predictions:",
        target_predictions_path,
    )
    print("=" * 80)
    print(
        "Protocol check: target test was used only AFTER "
        "checkpoint and threshold were fixed."
    )


if __name__ == "__main__":
    main()
