"""
07_train_safas.py

Adapted SA-FAS training for the image-only Face Anti-Spoofing project.

IMPORTANT
---------
This is an ADAPTED implementation for the current 3-dataset study. It is not
claimed as an exact reproduction of the official O-C-I-M protocol.

What is kept identical to the ResNet18 baseline
-----------------------------------------------
- Same experiment CSV files / source-target splits
- Same image-only scope
- Same 256x256 RGB preprocessing
- Same domain+class balanced sampler
- Same ImageNet-pretrained ResNet18 backbone family
- Same batch size (default 16)
- Same SGD hyperparameters
- Same 30-epoch preliminary schedule
- Same source-only checkpoint selection by macro validation AUC
- Target test is NOT loaded in this script

What SA-FAS adds
----------------
1. Separability:
   Domain-aware supervised contrastive loss. Positives have BOTH the same
   class label and the same source domain.

2. Alignment:
   One binary classifier (hyperplane) per source domain. After the warm-up,
   PG-IRM-style interpolation moves the domain-wise hyperplanes closer.

3. Inference rule used for validation:
   Mean logit across the source-domain classifiers, following the SA-FAS idea
   of using the mean hyperplane at inference.

Labels
------
0 = bona_fide
1 = spoof
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score

from fas_dataset import (
    FASDataset,
    BalancedDomainClassBatchSampler,
    build_transforms,
    find_tese_root,
    experiment_csv_for_target,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 5e-3
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_EPOCHS = 30
SMOKE_TEST_EPOCHS = 3
DEFAULT_MILESTONES = [15, 25]
DEFAULT_GAMMA = 0.5
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 0

# SA-FAS-specific values from the paper / official implementation.
DEFAULT_SEP_WEIGHT = 0.1
DEFAULT_TEMPERATURE = 0.1
DEFAULT_ALPHA = 0.995
DEFAULT_ALIGN_START_EPOCH = 6


class TwoViewTransform:
    """Generate two independent augmented views of the same image.

    We intentionally reuse the SAME conservative training transform as the
    baseline instead of adding stronger SA-FAS augmentations at this stage.
    This keeps the comparison focused on the learning objective.
    """

    def __init__(self):
        self.base_transform = build_transforms(train=True)

    def __call__(self, image):
        view1 = self.base_transform(image)
        view2 = self.base_transform(image)
        return torch.stack([view1, view2], dim=0)


class SAFASResNet18(nn.Module):
    """ResNet18 encoder + one binary hyperplane per source domain."""

    def __init__(
        self,
        source_domain_ids,
        pretrained: bool = True,
    ):
        super().__init__()

        source_domain_ids = sorted(int(x) for x in source_domain_ids)
        if len(source_domain_ids) < 2:
            raise ValueError("SA-FAS requer pelo menos dois source domains.")

        self.source_domain_ids = source_domain_ids

        if pretrained:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            backbone = resnet18(weights=None)

        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        # Paper formulation uses domain-wise live-vs-spoof hyperplanes.
        # We implement one scalar binary classifier per source domain.
        self.classifiers = nn.ModuleDict(
            {
                str(domain_id): nn.Linear(self.feature_dim, 1, bias=True)
                for domain_id in self.source_domain_ids
            }
        )

    def encode(self, x):
        features = self.encoder(x)
        return F.normalize(features, p=2, dim=1)

    def domain_logits(self, features, domain_ids):
        """Apply each sample's own source-domain classifier.

        All source-domain classifiers are evaluated and the logit associated
        with each sample's own domain is selected with gather(). This avoids
        dtype mismatches under CUDA AMP, where Linear layers may return
        float16 logits while normalized features can remain float32.
        """
        all_logits = torch.stack(
            [
                self.classifiers[str(domain_id)](features).squeeze(1)
                for domain_id in self.source_domain_ids
            ],
            dim=1,
        )  # [B, num_source_domains]

        classifier_columns = torch.empty_like(domain_ids, dtype=torch.long)

        assigned = torch.zeros_like(domain_ids, dtype=torch.bool)

        for column, domain_id in enumerate(self.source_domain_ids):
            mask = domain_ids == domain_id
            classifier_columns[mask] = column
            assigned |= mask

        if not bool(assigned.all()):
            unknown = torch.unique(domain_ids[~assigned]).detach().cpu().tolist()
            raise RuntimeError(
                f"Domain ids sem classificador SA-FAS: {unknown}"
            )

        return all_logits.gather(
            1,
            classifier_columns.unsqueeze(1),
        ).squeeze(1)

    def mean_logits(self, features):
        """Mean hyperplane inference used by SA-FAS."""
        logits = [
            self.classifiers[str(domain_id)](features).squeeze(1)
            for domain_id in self.source_domain_ids
        ]
        return torch.stack(logits, dim=0).mean(dim=0)

    def snapshot_hyperplanes(self):
        """Snapshot pre-SGD domain hyperplanes for PG-IRM interpolation."""
        snapshot = {}

        for domain_id in self.source_domain_ids:
            layer = self.classifiers[str(domain_id)]
            pieces = [layer.weight.detach().clone().flatten()]
            if layer.bias is not None:
                pieces.append(layer.bias.detach().clone().flatten())
            snapshot[domain_id] = torch.cat(pieces)

        return snapshot

    @torch.no_grad()
    def _get_hyperplane_vector(self, domain_id):
        layer = self.classifiers[str(domain_id)]
        pieces = [layer.weight.detach().flatten()]
        if layer.bias is not None:
            pieces.append(layer.bias.detach().flatten())
        return torch.cat(pieces)

    @torch.no_grad()
    def _set_hyperplane_vector(self, domain_id, vector):
        layer = self.classifiers[str(domain_id)]
        weight_numel = layer.weight.numel()

        layer.weight.copy_(
            vector[:weight_numel].view_as(layer.weight)
        )

        if layer.bias is not None:
            layer.bias.copy_(
                vector[weight_numel:].view_as(layer.bias)
            )

    @torch.no_grad()
    def pgirm_interpolate(self, snapshot, alpha: float):
        """Adapted PG-IRM projected-gradient interpolation.

        After SGD, each updated domain hyperplane is interpolated toward the
        farthest OLD hyperplane from another source domain:

            beta <- alpha * beta_updated + (1-alpha) * beta_other_old

        With the current two-source protocol, the "farthest other" is simply
        the other source-domain hyperplane.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha tem de estar entre 0 e 1.")

        updated_vectors = {
            domain_id: self._get_hyperplane_vector(domain_id).clone()
            for domain_id in self.source_domain_ids
        }

        for domain_id in self.source_domain_ids:
            candidates = [
                other
                for other in self.source_domain_ids
                if other != domain_id
            ]

            updated = updated_vectors[domain_id]

            farthest_domain = max(
                candidates,
                key=lambda other: torch.norm(
                    updated - snapshot[other], p=2
                ).item(),
            )

            projected = (
                alpha * updated
                + (1.0 - alpha) * snapshot[farthest_domain]
            )

            self._set_hyperplane_vector(domain_id, projected)

        return self.classifier_cosine_similarity()

    @torch.no_grad()
    def classifier_cosine_similarity(self):
        """Mean pairwise cosine similarity of classifier weight vectors."""
        vectors = []
        for domain_id in self.source_domain_ids:
            weight = self.classifiers[str(domain_id)].weight.detach().flatten()
            vectors.append(F.normalize(weight, dim=0))

        similarities = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarities.append(torch.dot(vectors[i], vectors[j]).item())

        if not similarities:
            return float("nan")

        return sum(similarities) / len(similarities)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train adapted SA-FAS using source domains only."
    )

    parser.add_argument(
        "--target",
        required=True,
        choices=["nuaa", "lcc_fasd", "casia_fasd"],
        help="Unseen target domain. Never used by this training script.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument(
        "--sep-weight",
        type=float,
        default=DEFAULT_SEP_WEIGHT,
        help="Weight lambda for the separability / SupCon loss.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="SupCon temperature.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="PG-IRM interpolation alpha.",
    )
    parser.add_argument(
        "--align-start-epoch",
        type=int,
        default=DEFAULT_ALIGN_START_EPOCH,
        help="Full runs activate PG-IRM after this many warm-up epochs.",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_amp_objects(device):
    use_amp = device.type == "cuda"
    if not use_amp:
        return False, None

    try:
        return True, torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return True, torch.cuda.amp.GradScaler()


def autocast_context(device, enabled):
    if not enabled:
        return torch.autocast(device_type="cpu", enabled=False)

    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def source_domain_ids_from_dataset(dataset):
    return sorted(
        {
            int(row["domain_id"])
            for row in dataset.rows
        }
    )


def make_loaders(
    experiment_csv,
    tese_root,
    batch_size,
    num_workers,
    seed,
):
    # Train gets TWO augmented views per original sample.
    train_dataset = FASDataset(
        experiment_csv,
        role="train",
        tese_root=tese_root,
        transform=TwoViewTransform(),
    )

    # Validation stays one-view and uses the same baseline val preprocessing.
    val_dataset = FASDataset(
        experiment_csv,
        role="val",
        tese_root=tese_root,
        transform=build_transforms(train=False),
    )

    sampler = BalancedDomainClassBatchSampler(
        train_dataset,
        batch_size=batch_size,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_dataset, val_dataset, sampler, train_loader, val_loader


def domain_aware_supcon_loss(features, group_labels, temperature=0.1):
    """Domain-aware supervised contrastive loss.

    Parameters
    ----------
    features:
        Tensor [B, V, D], normally V=2 augmented views.
    group_labels:
        Tensor [B]. Samples are positives iff they have the same group label.
        We encode group = domain_id * 2 + class_id.
    """
    if features.ndim != 3:
        raise ValueError("features deve ter shape [B, V, D].")

    batch_size, n_views, feat_dim = features.shape

    # [B*V, D], ordered sample-view/sample-view...
    contrast = features.reshape(batch_size * n_views, feat_dim)
    contrast = F.normalize(contrast, p=2, dim=1)

    repeated_labels = group_labels.repeat_interleave(n_views)

    logits = torch.matmul(contrast, contrast.T) / temperature

    # Numerical stability.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    identity = torch.eye(
        batch_size * n_views,
        device=features.device,
        dtype=torch.bool,
    )

    positive_mask = (
        repeated_labels.unsqueeze(0) == repeated_labels.unsqueeze(1)
    ) & (~identity)

    denominator_mask = ~identity

    # log sum exp over all non-self samples.
    exp_logits = torch.exp(logits) * denominator_mask.float()
    log_prob = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0

    if not valid.any():
        raise RuntimeError(
            "SupCon batch sem pares positivos. Verifica o balanced sampler."
        )

    mean_log_prob_pos = (
        (positive_mask.float() * log_prob).sum(dim=1)
        / positive_count.clamp_min(1)
    )

    return -mean_log_prob_pos[valid].mean()


def train_one_epoch(
    model,
    loader,
    sampler,
    optimizer,
    device,
    epoch,
    amp_enabled,
    scaler,
    sep_weight,
    temperature,
    alpha,
    align_start_epoch,
):
    model.train()
    sampler.set_epoch(epoch)

    bce = nn.BCEWithLogitsLoss()

    total_samples = 0
    correct = 0
    sum_total_loss = 0.0
    sum_cls_loss = 0.0
    sum_sep_loss = 0.0
    last_cosine = model.classifier_cosine_similarity()

    for batch in loader:
        # [B, 2, 3, 256, 256]
        views = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        domain_ids = batch["domain_id"].to(device, non_blocking=True)

        if views.ndim != 5 or views.size(1) != 2:
            raise RuntimeError(
                f"Esperava [B,2,C,H,W], recebi {tuple(views.shape)}"
            )

        batch_size = labels.size(0)
        flat_images = views.reshape(
            batch_size * 2,
            *views.shape[2:],
        )

        labels_2 = labels.repeat_interleave(2)
        domains_2 = domain_ids.repeat_interleave(2)

        # Positive identity = same domain AND same class.
        group_labels = domain_ids * 2 + labels

        optimizer.zero_grad(set_to_none=True)

        # Snapshot before SGD, matching the PG-IRM update order.
        snapshot = model.snapshot_hyperplanes()

        with autocast_context(device, amp_enabled):
            features_flat = model.encode(flat_images)
            features_two_view = features_flat.reshape(
                batch_size,
                2,
                model.feature_dim,
            )

            domain_logits = model.domain_logits(
                features_flat,
                domains_2,
            )

            cls_loss = bce(
                domain_logits,
                labels_2.float(),
            )

            sep_loss = domain_aware_supcon_loss(
                features_two_view.float(),
                group_labels,
                temperature=temperature,
            )

            total_loss = cls_loss + sep_weight * sep_loss

        if amp_enabled:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # Paper Ta=20. In a 1-based log, activation begins AFTER 20 warm-up
        # epochs. Smoke test lowers this value so the alignment code path is
        # actually tested.
        if epoch > align_start_epoch:
            last_cosine = model.pgirm_interpolate(
                snapshot,
                alpha=alpha,
            )
        else:
            last_cosine = model.classifier_cosine_similarity()

        predictions = (torch.sigmoid(domain_logits.detach()) >= 0.5).long()
        correct += (predictions == labels_2).sum().item()
        total_samples += labels_2.numel()

        sum_total_loss += total_loss.item() * labels_2.numel()
        sum_cls_loss += cls_loss.item() * labels_2.numel()
        sum_sep_loss += sep_loss.item() * labels_2.numel()

    return {
        "loss": sum_total_loss / total_samples,
        "classification_loss": sum_cls_loss / total_samples,
        "separability_loss": sum_sep_loss / total_samples,
        "accuracy": correct / total_samples,
        "classifier_cosine": last_cosine,
    }


@torch.no_grad()
def validate(model, loader, device, amp_enabled):
    model.eval()

    bce = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    total = 0
    all_labels = []
    all_scores = []
    all_domains = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast_context(device, amp_enabled):
            features = model.encode(images)
            mean_logits = model.mean_logits(features)
            loss = bce(mean_logits, labels.float())

        scores = torch.sigmoid(mean_logits.float())

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_scores.extend(scores.cpu().tolist())
        all_domains.extend(batch["domain_id"].cpu().tolist())

    by_domain = defaultdict(lambda: {"labels": [], "scores": []})
    for label, score, domain_id in zip(all_labels, all_scores, all_domains):
        by_domain[int(domain_id)]["labels"].append(int(label))
        by_domain[int(domain_id)]["scores"].append(float(score))

    domain_aucs = {}
    for domain_id, values in sorted(by_domain.items()):
        if len(set(values["labels"])) < 2:
            raise RuntimeError(
                f"Validation domain {domain_id} não contém ambas as classes."
            )
        domain_aucs[domain_id] = roc_auc_score(
            values["labels"],
            values["scores"],
        )

    macro_auc = sum(domain_aucs.values()) / len(domain_aucs)
    pooled_auc = roc_auc_score(all_labels, all_scores)

    return {
        "loss": running_loss / total,
        "macro_auc": macro_auc,
        "pooled_auc": pooled_auc,
        "domain_aucs": domain_aucs,
        "classifier_cosine": model.classifier_cosine_similarity(),
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_macro_auc,
    args,
    experiment_csv,
    source_domain_ids,
    effective_align_start_epoch,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "best_macro_source_val_auc": best_macro_auc,
            "target": args.target,
            "experiment_csv": str(experiment_csv),
            "source_domain_ids": list(source_domain_ids),
            "config": {
                "method": "adapted_sa_fas",
                "architecture": "resnet18",
                "pretrained": not args.no_pretrained,
                "binary_classifier": "one_logit_per_source_domain",
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "sep_weight": args.sep_weight,
                "temperature": args.temperature,
                "alpha": args.alpha,
                "align_start_epoch": effective_align_start_epoch,
                "seed": args.seed,
                "labels": {"0": "bona_fide", "1": "spoof"},
            },
        },
        path,
    )


def write_history(path, history):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_classification_loss",
        "train_separability_loss",
        "train_accuracy",
        "train_classifier_cosine",
        "val_loss",
        "val_macro_auc",
        "val_pooled_auc",
        "val_domain_aucs",
        "val_classifier_cosine",
        "alignment_active",
        "epoch_seconds",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    epochs = (
        args.epochs
        if args.epochs is not None
        else (SMOKE_TEST_EPOCHS if args.smoke_test else DEFAULT_EPOCHS)
    )

    if epochs <= 0:
        raise ValueError("epochs tem de ser > 0.")

    set_seed(args.seed)
    tese_root = find_tese_root()
    experiment_csv = experiment_csv_for_target(tese_root, args.target)
    device = get_device()

    output_key = f"target_{args.target}"
    run_name = "smoke" if args.smoke_test else "safas"

    checkpoint_dir = tese_root / "checkpoints" / "safas" / output_key
    results_dir = tese_root / "results" / "safas" / output_key

    checkpoint_path = checkpoint_dir / (
        "best_safas_smoke.pt" if args.smoke_test else "best_safas.pt"
    )
    aligned_checkpoint_path = checkpoint_dir / (
        "best_safas_aligned_smoke.pt"
        if args.smoke_test
        else "best_safas_aligned.pt"
    )
    history_path = results_dir / (
        "training_history_smoke.csv" if args.smoke_test else "training_history.csv"
    )
    config_path = results_dir / (
        "run_config_smoke.json" if args.smoke_test else "run_config.json"
    )

    (
        train_dataset,
        val_dataset,
        train_sampler,
        train_loader,
        val_loader,
    ) = make_loaders(
        experiment_csv,
        tese_root,
        args.batch_size,
        args.num_workers,
        args.seed,
    )

    source_domain_ids = source_domain_ids_from_dataset(train_dataset)

    if len(source_domain_ids) != 2:
        raise RuntimeError(
            "O protocolo atual espera exatamente 2 source domains; "
            f"encontrei {source_domain_ids}."
        )

    model = SAFASResNet18(
        source_domain_ids,
        pretrained=not args.no_pretrained,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if not args.smoke_test:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=DEFAULT_MILESTONES,
            gamma=DEFAULT_GAMMA,
        )

    amp_enabled, scaler = make_amp_objects(device)

    # 3-epoch smoke test must exercise PG-IRM too, so activate it after the
    # first warm-up epoch. Full runs retain Ta=20.
    effective_align_start_epoch = (
        1 if args.smoke_test else args.align_start_epoch
    )

    run_config = {
        "run_type": run_name,
        "method": "adapted_sa_fas",
        "target": args.target,
        "target_usage": "TEST ONLY - not loaded by this script",
        "source_domain_ids": source_domain_ids,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "optimizer": "SGD",
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "scheduler": None if scheduler is None else "MultiStepLR",
        "scheduler_milestones": None if scheduler is None else DEFAULT_MILESTONES,
        "scheduler_gamma": None if scheduler is None else DEFAULT_GAMMA,
        "sep_weight": args.sep_weight,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "align_start_epoch": effective_align_start_epoch,
        "full_run_align_start_epoch": args.align_start_epoch,
        "two_views": True,
        "image_size": 256,
        "rgb_only": True,
        "pretrained_imagenet": not args.no_pretrained,
        "checkpoint_metric": "macro source-validation AUC",
        "aligned_checkpoint_metric": (
            "best macro source-validation AUC with PG-IRM active"
        ),
        "inference": "mean of source-domain hyperplanes",
        "classification_threshold": None,
        "seed": args.seed,
        "amp": amp_enabled,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "experiment_csv": str(experiment_csv),
    }
    save_json(config_path, run_config)

    print("=" * 80)
    print("ADAPTED SA-FAS")
    print("=" * 80)
    print("Run type:", run_name)
    print("Target:", args.target)
    print("Target usage: TEST ONLY (não utilizado neste script)")
    print("Source domain IDs:", source_domain_ids)
    print("Device:", device)
    print("AMP:", amp_enabled)
    print("ImageNet pretrained:", not args.no_pretrained)
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Batch size:", args.batch_size)
    print("Two augmented views per train sample: True")
    print("Epochs:", epochs)
    print("SGD lr:", args.lr)
    print("Momentum:", args.momentum)
    print("Weight decay:", args.weight_decay)
    print("Separability weight (lambda):", args.sep_weight)
    print("SupCon temperature:", args.temperature)
    print("PG-IRM alpha:", args.alpha)
    print("PG-IRM starts AFTER epoch:", effective_align_start_epoch)
    if scheduler is None:
        print("Scheduler: none (smoke test)")
    else:
        print(
            "Scheduler: MultiStepLR "
            f"milestones={DEFAULT_MILESTONES}, gamma={DEFAULT_GAMMA}"
        )
    print("Checkpoint metric: macro source-validation AUC")
    print(
        "Aligned checkpoint: best macro source-validation AUC "
        "with PG-IRM active"
    )
    print("Inference: mean of source-domain hyperplanes")
    print("Classification threshold: NOT selected")
    print("=" * 80)

    best_macro_auc = float("-inf")
    best_aligned_macro_auc = float("-inf")
    best_aligned_epoch = None
    history = []

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            sampler=train_sampler,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            amp_enabled=amp_enabled,
            scaler=scaler,
            sep_weight=args.sep_weight,
            temperature=args.temperature,
            alpha=args.alpha,
            align_start_epoch=effective_align_start_epoch,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
        )

        alignment_active = epoch > effective_align_start_epoch

        improved = val_metrics["macro_auc"] > best_macro_auc
        if improved:
            best_macro_auc = val_metrics["macro_auc"]
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_macro_auc,
                args,
                experiment_csv,
                source_domain_ids,
                effective_align_start_epoch,
            )

        improved_aligned = (
            alignment_active
            and val_metrics["macro_auc"] > best_aligned_macro_auc
        )
        if improved_aligned:
            best_aligned_macro_auc = val_metrics["macro_auc"]
            best_aligned_epoch = epoch
            save_checkpoint(
                aligned_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_aligned_macro_auc,
                args,
                experiment_csv,
                source_domain_ids,
                effective_align_start_epoch,
            )

        elapsed = time.perf_counter() - start

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "train_classification_loss": train_metrics["classification_loss"],
            "train_separability_loss": train_metrics["separability_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_classifier_cosine": train_metrics["classifier_cosine"],
            "val_loss": val_metrics["loss"],
            "val_macro_auc": val_metrics["macro_auc"],
            "val_pooled_auc": val_metrics["pooled_auc"],
            "val_domain_aucs": json.dumps(val_metrics["domain_aucs"], sort_keys=True),
            "val_classifier_cosine": val_metrics["classifier_cosine"],
            "alignment_active": alignment_active,
            "epoch_seconds": elapsed,
        }
        history.append(row)
        write_history(history_path, history)

        domain_text = ", ".join(
            f"domain_{domain_id}={auc:.4f}"
            for domain_id, auc in sorted(val_metrics["domain_aucs"].items())
        )

        markers = []
        if improved:
            markers.append("BEST")
        if improved_aligned:
            markers.append("BEST-ALIGNED")
        marker_text = (
            "  <-- " + ", ".join(markers)
            if markers
            else ""
        )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"cls={train_metrics['classification_loss']:.4f} | "
            f"sep={train_metrics['separability_loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_auc={val_metrics['macro_auc']:.4f} | "
            f"val_pooled_auc={val_metrics['pooled_auc']:.4f} | "
            f"{domain_text} | "
            f"cos={val_metrics['classifier_cosine']:.4f} | "
            f"align={'ON' if alignment_active else 'OFF'} | "
            f"{elapsed:.1f}s"
            + marker_text
        )

        if scheduler is not None:
            scheduler.step()

    print("\n" + "=" * 80)
    print("SA-FAS TRAINING FINISHED")
    print("=" * 80)
    print(f"Best overall macro source-validation AUC: {best_macro_auc:.4f}")
    print("Best overall checkpoint:", checkpoint_path)

    if best_aligned_epoch is not None:
        print(
            "Best aligned macro source-validation AUC: "
            f"{best_aligned_macro_auc:.4f}"
        )
        print("Best aligned epoch:", best_aligned_epoch)
        print("Best aligned checkpoint:", aligned_checkpoint_path)
    else:
        print("Best aligned checkpoint: NONE (alignment never became active)")

    print("Training history:", history_path)
    print("Run config:", config_path)
    print()
    print(
        "IMPORTANT: o unseen target ainda NÃO foi avaliado. "
        "Use the aligned checkpoint for the final SA-FAS target evaluation."
    )


if __name__ == "__main__":
    main()
