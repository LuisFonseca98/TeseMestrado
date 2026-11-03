"""
08_evaluate_safas.py

Final evaluation of the adapted SA-FAS model on the unseen target domain.

Protocol
--------
1. Load best_safas_aligned.pt, selected using SOURCE validation only.
2. Run inference on SOURCE validation data.
3. Select ONE threshold using pooled SOURCE-validation EER only.
4. Freeze that threshold.
5. Only then instantiate/load the unseen TARGET test set.
6. Evaluate AUC, APCER, BPCER, ACER, HTER, balanced accuracy,
   accuracy and confusion matrix.

Labels
------
0 = bona_fide
1 = spoof

Inference
---------
The trained SA-FAS model contains one binary hyperplane per source domain.
At validation/test time, inference uses the mean logit across the source-domain
hyperplanes, matching the training script.

IMPORTANT
---------
This is an adapted image-only SA-FAS experiment, not an official reproduction
of the O-C-I-M protocol.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            "Select threshold on SOURCE validation and evaluate the "
            "adapted SA-FAS aligned checkpoint on the unseen target."
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
            "Optional explicit checkpoint path. Default: "
            "checkpoints/safas/target_<target>/best_safas_aligned.pt"
        ),
    )

    return parser.parse_args()


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


class SAFASResNet18(nn.Module):
    """Architecture required to reload the adapted SA-FAS checkpoint."""

    def __init__(self, source_domain_ids):
        super().__init__()

        self.source_domain_ids = sorted(
            int(x) for x in source_domain_ids
        )

        if len(self.source_domain_ids) < 2:
            raise ValueError(
                "SA-FAS requires at least two source domains."
            )

        backbone = resnet18(weights=None)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        self.classifiers = nn.ModuleDict(
            {
                str(domain_id): nn.Linear(
                    self.feature_dim,
                    1,
                    bias=True,
                )
                for domain_id in self.source_domain_ids
            }
        )

    def encode(self, x):
        features = self.encoder(x)
        return F.normalize(
            features,
            p=2,
            dim=1,
        )

    def mean_logits(self, features):
        logits = [
            self.classifiers[str(domain_id)](
                features
            ).squeeze(1)
            for domain_id in self.source_domain_ids
        ]

        return torch.stack(
            logits,
            dim=0,
        ).mean(dim=0)

    @torch.no_grad()
    def classifier_cosine_similarity(self):
        vectors = []

        for domain_id in self.source_domain_ids:
            layer = self.classifiers[str(domain_id)]
            pieces = [layer.weight.detach().flatten()]
            if layer.bias is not None:
                pieces.append(layer.bias.detach().flatten())
            vectors.append(torch.cat(pieces))

        similarities = []

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarities.append(
                    F.cosine_similarity(
                        vectors[i].unsqueeze(0),
                        vectors[j].unsqueeze(0),
                    ).item()
                )

        if not similarities:
            return 1.0

        return float(
            sum(similarities) / len(similarities)
        )


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    source_domain_ids = checkpoint.get(
        "source_domain_ids"
    )

    if not source_domain_ids:
        raise RuntimeError(
            "Checkpoint does not contain source_domain_ids."
        )

    model = SAFASResNet18(
        source_domain_ids
    ).to(device)

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
                features = model.encode(images)
                logits = model.mean_logits(features)
        else:
            features = model.encode(images)
            logits = model.mean_logits(features)

        spoof_scores = torch.sigmoid(
            logits.float()
        )

        all_labels.extend(
            batch["label"]
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

        all_scores.extend(
            spoof_scores
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )

        all_domain_ids.extend(
            batch["domain_id"]
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

        all_sample_ids.extend(
            list(batch["sample_id"])
        )
        all_datasets.extend(
            list(batch["dataset"])
        )
        all_native_splits.extend(
            list(batch["native_split"])
        )
        all_attack_types.extend(
            list(batch["attack_type"])
        )
        all_qualities.extend(
            list(batch["quality"])
        )
        all_subject_ids.extend(
            list(batch["subject_id"])
        )

    return {
        "labels": np.asarray(
            all_labels,
            dtype=np.int64,
        ),
        "scores": np.asarray(
            all_scores,
            dtype=np.float64,
        ),
        "domain_ids": np.asarray(
            all_domain_ids,
            dtype=np.int64,
        ),
        "sample_ids": all_sample_ids,
        "datasets": all_datasets,
        "native_splits": all_native_splits,
        "attack_types": all_attack_types,
        "qualities": all_qualities,
        "subject_ids": all_subject_ids,
    }


def select_eer_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
):
    if len(np.unique(labels)) < 2:
        raise RuntimeError(
            "Both classes are required to calculate EER."
        )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    finite_mask = np.isfinite(
        thresholds
    )

    if not finite_mask.any():
        raise RuntimeError(
            "No finite ROC threshold was found."
        )

    fpr_f = fpr[finite_mask]
    fnr_f = fnr[finite_mask]
    thresholds_f = thresholds[finite_mask]

    idx = int(
        np.argmin(
            np.abs(
                fpr_f - fnr_f
            )
        )
    )

    threshold = float(
        thresholds_f[idx]
    )

    eer = float(
        (
            fpr_f[idx]
            + fnr_f[idx]
        )
        / 2.0
    )

    return threshold, eer


def threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
):
    predictions = (
        scores >= threshold
    ).astype(np.int64)

    bona_mask = labels == 0
    spoof_mask = labels == 1

    if not bona_mask.any() or not spoof_mask.any():
        raise RuntimeError(
            "APCER/BPCER require both classes."
        )

    tn = int(
        np.sum(
            (labels == 0)
            & (predictions == 0)
        )
    )

    fp = int(
        np.sum(
            (labels == 0)
            & (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (labels == 1)
            & (predictions == 0)
        )
    )

    tp = int(
        np.sum(
            (labels == 1)
            & (predictions == 1)
        )
    )

    apcer = fn / (tp + fn)
    bpcer = fp / (tn + fp)
    acer = (
        apcer + bpcer
    ) / 2.0
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
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "accuracy": float(accuracy),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "samples": int(len(labels)),
        "bona_fide_samples": int(
            np.sum(bona_mask)
        ),
        "spoof_samples": int(
            np.sum(spoof_mask)
        ),
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
        mask = (
            domain_ids == domain_id
        )

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
                    "sample_id": prediction_data[
                        "sample_ids"
                    ][i],
                    "dataset": prediction_data[
                        "datasets"
                    ][i],
                    "native_split": prediction_data[
                        "native_splits"
                    ][i],
                    "domain_id": int(
                        prediction_data[
                            "domain_ids"
                        ][i]
                    ),
                    "subject_id": prediction_data[
                        "subject_ids"
                    ][i],
                    "attack_type": prediction_data[
                        "attack_types"
                    ][i],
                    "quality": prediction_data[
                        "qualities"
                    ][i],
                    "label": int(
                        labels[i]
                    ),
                    "spoof_probability": float(
                        scores[i]
                    ),
                    "threshold": float(
                        threshold
                    ),
                    "prediction": int(
                        predictions[i]
                    ),
                    "correct": int(
                        predictions[i]
                        == labels[i]
                    ),
                }
            )


def save_json(
    path: Path,
    data: dict,
):
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
    return (
        f"{100.0 * value:.2f}%"
    )


def main():
    args = parse_args()

    tese_root = find_tese_root()

    experiment_csv = (
        experiment_csv_for_target(
            tese_root,
            args.target,
        )
    )

    output_key = (
        f"target_{args.target}"
    )

    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint is not None
        else (
            tese_root
            / "checkpoints"
            / "safas"
            / output_key
            / "best_safas_aligned.pt"
        )
    )

    results_dir = (
        tese_root
        / "results"
        / "safas"
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
    print(
        "ADAPTED SA-FAS - FINAL UNSEEN TARGET EVALUATION"
    )
    print("=" * 80)
    print("Target:", args.target)
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print(
        "Checkpoint policy: best SOURCE-validation AUC "
        "with PG-IRM active"
    )
    print(
        "Inference: mean source-domain hyperplane"
    )
    print(
        "Threshold rule: pooled SOURCE-validation EER"
    )
    print(
        "Target used for threshold selection: NO"
    )
    print("=" * 80)

    model, checkpoint = (
        load_checkpoint_model(
            checkpoint_path,
            device,
        )
    )

    checkpoint_target = str(
        checkpoint.get(
            "target",
            "",
        )
    )

    if (
        checkpoint_target
        and checkpoint_target
        != args.target
    ):
        raise RuntimeError(
            "Checkpoint was trained for another target: "
            f"{checkpoint_target} != {args.target}"
        )

    checkpoint_epoch = int(
        checkpoint.get(
            "epoch",
            -1,
        )
    )

    align_start_epoch = int(
        checkpoint.get(
            "config",
            {},
        ).get(
            "align_start_epoch",
            -1,
        )
    )

    if (
        align_start_epoch >= 0
        and checkpoint_epoch
        <= align_start_epoch
    ):
        raise RuntimeError(
            "The selected checkpoint was saved before PG-IRM "
            "became active. Use best_safas_aligned.pt."
        )

    print(
        f"Checkpoint epoch: {checkpoint_epoch}"
    )
    print(
        f"PG-IRM warm-up ends after epoch: {align_start_epoch}"
    )
    print(
        "Classifier cosine similarity: "
        f"{model.classifier_cosine_similarity():.4f}"
    )

    # ---------------------------------------------------------------
    # STEP 1: SOURCE VALIDATION ONLY
    # ---------------------------------------------------------------
    print(
        "\n[1/3] SOURCE validation inference..."
    )

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
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_data = collect_predictions(
        model,
        val_loader,
        device,
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

    val_domain_metrics = (
        metrics_by_domain(
            val_data["labels"],
            val_data["scores"],
            val_data["domain_ids"],
            threshold,
        )
    )

    print(
        "Source validation pooled AUC: "
        f"{val_metrics['auc']:.4f}"
    )
    print(
        "Selected source-validation EER threshold: "
        f"{threshold:.6f}"
    )
    print(
        "Approx. source-validation EER: "
        f"{fmt_pct(source_val_eer)}"
    )

    write_predictions_csv(
        val_predictions_path,
        val_data,
        threshold,
    )

    frozen_threshold = float(
        threshold
    )

    print(
        "\n[2/3] Threshold FROZEN. "
        "Only now is the unseen target test set loaded."
    )

    # ---------------------------------------------------------------
    # STEP 2: UNSEEN TARGET TEST
    # ---------------------------------------------------------------
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
        pin_memory=(
            device.type == "cuda"
        ),
    )

    target_data = collect_predictions(
        model,
        target_loader,
        device,
    )

    target_metrics = (
        threshold_metrics(
            target_data["labels"],
            target_data["scores"],
            frozen_threshold,
        )
    )

    write_predictions_csv(
        target_predictions_path,
        target_data,
        frozen_threshold,
    )

    # ---------------------------------------------------------------
    # STEP 3: SAVE + REPORT
    # ---------------------------------------------------------------
    summary = {
        "model": (
            "Adapted SA-FAS "
            "(ResNet18 + domain-aware SupCon + PG-IRM)"
        ),
        "scope": "image-only",
        "official_reproduction": False,
        "labels": {
            "0": "bona_fide",
            "1": "spoof",
        },
        "target": args.target,
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_policy": (
            "best macro source-validation AUC "
            "among epochs with PG-IRM active"
        ),
        "checkpoint_epoch": (
            checkpoint_epoch
        ),
        "checkpoint_best_macro_source_val_auc": float(
            checkpoint.get(
                "best_macro_source_val_auc",
                math.nan,
            )
        ),
        "source_domain_ids": list(
            checkpoint.get(
                "source_domain_ids",
                [],
            )
        ),
        "align_start_epoch": (
            align_start_epoch
        ),
        "classifier_cosine_similarity": (
            model.classifier_cosine_similarity()
        ),
        "inference": (
            "sigmoid(mean source-domain hyperplane logit)"
        ),
        "threshold_protocol": {
            "selection_data": (
                "source validation only"
            ),
            "method": "pooled EER",
            "threshold": (
                frozen_threshold
            ),
            "source_validation_eer": (
                source_val_eer
            ),
            "target_used_for_threshold_selection": False,
        },
        "source_validation": {
            "pooled": (
                val_metrics
            ),
            "by_domain_id": (
                val_domain_metrics
            ),
        },
        "unseen_target_test": (
            target_metrics
        ),
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

    print(
        "\n[3/3] FINAL UNSEEN TARGET RESULTS"
    )
    print("=" * 80)
    print(
        f"Target: {args.target}"
    )
    print(
        f"Checkpoint epoch: {checkpoint_epoch}"
    )
    print(
        f"Frozen threshold: {frozen_threshold:.6f} "
        "(SOURCE validation only)"
    )
    print(
        f"AUC:               "
        f"{target_metrics['auc']:.4f}"
    )
    print(
        f"APCER:             "
        f"{fmt_pct(target_metrics['apcer'])}"
    )
    print(
        f"BPCER:             "
        f"{fmt_pct(target_metrics['bpcer'])}"
    )
    print(
        f"ACER:              "
        f"{fmt_pct(target_metrics['acer'])}"
    )
    print(
        f"HTER:              "
        f"{fmt_pct(target_metrics['hter'])}"
    )
    print(
        "Balanced accuracy: "
        f"{fmt_pct(target_metrics['balanced_accuracy'])}"
    )
    print(
        f"Accuracy:           "
        f"{fmt_pct(target_metrics['accuracy'])}"
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
    print(
        "Summary:",
        summary_path,
    )
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
        "Protocol check: unseen target was loaded only "
        "after checkpoint and threshold were fixed."
    )


if __name__ == "__main__":
    main()
