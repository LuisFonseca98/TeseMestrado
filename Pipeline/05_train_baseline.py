"""
05_train_baseline.py

Train the ResNet18 baseline for the image-only Face Anti-Spoofing project.

Current experimental design
---------------------------
Model:
    ResNet18
    ImageNet pretrained weights
    2 output classes

Labels:
    0 = bona_fide
    1 = spoof

Input:
    RGB
    256 x 256

Training:
    Balanced by source domain + class
    CrossEntropyLoss
    SGD
    learning rate = 0.005
    momentum = 0.9
    weight decay = 0.0005
    batch size = 16

Smoke test:
    3 epochs
    No learning-rate decay is necessary for such a short diagnostic run.

Baseline:
    30 epochs
    MultiStepLR milestones = [15, 25]
    gamma = 0.5

Checkpoint:
    Best macro source-validation AUC.

IMPORTANT METHODOLOGICAL RULE
-----------------------------
The unseen target test set is NEVER used during training, model selection,
hyperparameter tuning, or threshold selection.

This script trains and validates on source domains only.
Final unseen-target evaluation will be performed by a separate evaluation
script after training.

No classification threshold is selected in this script.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score

from fas_dataset import (
    FASDataset,
    BalancedDomainClassBatchSampler,
    find_tese_root,
    experiment_csv_for_target,
)


# ---------------------------------------------------------------------------
# Baseline configuration
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

# num_workers=0 is deliberately conservative for the first Windows runs.
# Once the pipeline is stable, this can be increased experimentally.
DEFAULT_NUM_WORKERS = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train ResNet18 baseline using source domains only."
        )
    )

    parser.add_argument(
        "--target",
        required=True,
        choices=[
            "nuaa",
            "lcc_fasd",
            "casia_fasd",
        ],
        help=(
            "Unseen target domain. It is NOT used during training "
            "or validation."
        ),
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run only 3 epochs to verify that training, validation "
            "and checkpoint saving work correctly."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional manual epoch override. "
            "If omitted: 3 with --smoke-test, otherwise 30."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Training batch size. Must be divisible by the number "
            "of domain+class groups (currently 4)."
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Initial SGD learning rate.",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=DEFAULT_MOMENTUM,
        help="SGD momentum.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="SGD weight decay.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "PyTorch DataLoader workers. Start with 0 on Windows."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help=(
            "Disable ImageNet pretrained weights. "
            "Default is to use ImageNet pretrained ResNet18."
        ),
    )

    return parser.parse_args()


def set_seed(seed: int):
    """
    Fix the main random generators.

    This improves reproducibility, although exact bit-for-bit GPU
    reproducibility is not guaranteed across all hardware/software versions.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_model(pretrained: bool = True):
    """
    ResNet18 baseline.

    Why ResNet18?
    -------------
    - It is the same backbone family used by SA-FAS.
    - It provides a fairer later comparison with SA-FAS.
    - It is lightweight enough for the available hardware.

    Final layer:
    1000 ImageNet classes -> 2 FAS classes.
    """
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        2,
    )

    return model


def make_loaders(
    experiment_csv: Path,
    tese_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    """
    Build train and validation loaders.

    TRAIN:
      Uses BalancedDomainClassBatchSampler.

    VALIDATION:
      Keeps the natural source-domain distribution.
      No artificial balancing is applied.

    TARGET TEST:
      Not created here on purpose.
      This prevents accidental use during model development.
    """
    train_dataset = FASDataset(
        experiment_csv,
        role="train",
        tese_root=tese_root,
    )

    val_dataset = FASDataset(
        experiment_csv,
        role="val",
        tese_root=tese_root,
    )

    train_sampler = BalancedDomainClassBatchSampler(
        train_dataset,
        batch_size=batch_size,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
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

    return (
        train_dataset,
        val_dataset,
        train_sampler,
        train_loader,
        val_loader,
    )


def make_amp_objects(device):
    """
    Configure Automatic Mixed Precision (AMP).

    AMP is used only on CUDA.
    On CPU, training stays in regular FP32.

    The helper is written to support both newer and older PyTorch APIs.
    """
    use_amp = device.type == "cuda"

    if not use_amp:
        return False, None

    # Newer PyTorch:
    #   torch.amp.GradScaler("cuda")
    try:
        scaler = torch.amp.GradScaler(
            "cuda"
        )
        return True, scaler
    except (AttributeError, TypeError):
        pass

    # Older compatible API:
    scaler = torch.cuda.amp.GradScaler()
    return True, scaler


def autocast_context(device, enabled):
    """
    Return the correct autocast context for the installed PyTorch version.
    """
    if not enabled:
        return torch.autocast(
            device_type="cpu",
            enabled=False,
        )

    try:
        return torch.amp.autocast(
            device_type="cuda",
            enabled=True,
        )
    except AttributeError:
        return torch.cuda.amp.autocast(
            enabled=True
        )


def train_one_epoch(
    model,
    loader,
    sampler,
    criterion,
    optimizer,
    device,
    epoch,
    amp_enabled,
    scaler,
):
    """
    One training epoch.

    Core sequence:
      forward -> loss -> backward -> optimizer step
    """
    model.train()

    # Makes the balanced sampler reproducible while still changing its
    # sample order between epochs.
    sampler.set_epoch(epoch)

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast_context(
            device,
            amp_enabled,
        ):
            logits = model(images)

            # CrossEntropyLoss expects RAW logits.
            # Do NOT apply softmax before this loss.
            loss = criterion(
                logits,
                labels,
            )

        if amp_enabled:
            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss
        / total
    )

    epoch_accuracy = (
        correct
        / total
    )

    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
    }


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    amp_enabled,
):
    """
    Validate only on SOURCE validation data.

    The main checkpoint metric is macro source-validation AUC:
      1. compute spoof probability for every validation sample;
      2. compute AUC separately for each source domain;
      3. average the domain AUC values equally.

    Why macro by domain?
    --------------------
    Source datasets have different sizes and class distributions.
    A single pooled metric could be dominated by the larger source dataset.
    """
    model.eval()

    running_loss = 0.0
    total = 0

    all_labels = []
    all_spoof_probabilities = []
    all_domain_ids = []

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        with autocast_context(
            device,
            amp_enabled,
        ):
            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

        probabilities = torch.softmax(
            logits.float(),
            dim=1,
        )

        # Class 1 is spoof.
        spoof_probability = probabilities[
            :, 1
        ]

        running_loss += (
            loss.item()
            * images.size(0)
        )

        total += labels.size(0)

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_spoof_probabilities.extend(
            spoof_probability.cpu().tolist()
        )

        all_domain_ids.extend(
            batch["domain_id"].cpu().tolist()
        )

    val_loss = (
        running_loss
        / total
    )

    by_domain = defaultdict(
        lambda: {
            "labels": [],
            "scores": [],
        }
    )

    for label, score, domain_id in zip(
        all_labels,
        all_spoof_probabilities,
        all_domain_ids,
    ):
        by_domain[int(domain_id)]["labels"].append(
            int(label)
        )
        by_domain[int(domain_id)]["scores"].append(
            float(score)
        )

    domain_aucs = {}

    for domain_id, values in sorted(
        by_domain.items()
    ):
        labels = values["labels"]
        scores = values["scores"]

        # AUC requires both classes.
        if len(set(labels)) < 2:
            raise RuntimeError(
                f"Validation domain {domain_id} "
                "não contém ambas as classes; "
                "não é possível calcular AUC."
            )

        domain_aucs[domain_id] = (
            roc_auc_score(
                labels,
                scores,
            )
        )

    macro_auc = (
        sum(domain_aucs.values())
        / len(domain_aucs)
    )

    pooled_auc = roc_auc_score(
        all_labels,
        all_spoof_probabilities,
    )

    return {
        "loss": val_loss,
        "macro_auc": macro_auc,
        "pooled_auc": pooled_auc,
        "domain_aucs": domain_aucs,
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_macro_auc,
    args,
    experiment_csv,
):
    """
    Save enough state to reproduce or resume the selected model later.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "best_macro_source_val_auc": (
                best_macro_auc
            ),
            "target": args.target,
            "experiment_csv": str(
                experiment_csv
            ),
            "config": {
                "architecture": "resnet18",
                "pretrained": (
                    not args.no_pretrained
                ),
                "num_classes": 2,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "momentum": args.momentum,
                "weight_decay": (
                    args.weight_decay
                ),
                "seed": args.seed,
                "labels": {
                    "0": "bona_fide",
                    "1": "spoof",
                },
            },
        },
        path,
    )


def write_history(
    path: Path,
    history,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_macro_auc",
        "val_pooled_auc",
        "val_domain_aucs",
        "epoch_seconds",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(history)


def save_run_config(
    path: Path,
    config: dict,
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
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )


def main():
    args = parse_args()

    epochs = (
        args.epochs
        if args.epochs is not None
        else (
            SMOKE_TEST_EPOCHS
            if args.smoke_test
            else DEFAULT_EPOCHS
        )
    )

    if epochs <= 0:
        raise ValueError(
            "epochs tem de ser > 0."
        )

    set_seed(
        args.seed
    )

    tese_root = find_tese_root()

    experiment_csv = (
        experiment_csv_for_target(
            tese_root,
            args.target,
        )
    )

    device = get_device()

    run_name = (
        "smoke"
        if args.smoke_test
        else "baseline"
    )

    output_key = (
        f"target_{args.target}"
    )

    checkpoint_dir = (
        tese_root
        / "checkpoints"
        / "baseline"
        / output_key
    )

    results_dir = (
        tese_root
        / "results"
        / "baseline"
        / output_key
    )

    checkpoint_path = (
        checkpoint_dir
        / (
            "best_resnet18_smoke.pt"
            if args.smoke_test
            else "best_resnet18.pt"
        )
    )

    history_path = (
        results_dir
        / (
            "training_history_smoke.csv"
            if args.smoke_test
            else "training_history.csv"
        )
    )

    config_path = (
        results_dir
        / (
            "run_config_smoke.json"
            if args.smoke_test
            else "run_config.json"
        )
    )

    (
        train_dataset,
        val_dataset,
        train_sampler,
        train_loader,
        val_loader,
    ) = make_loaders(
        experiment_csv=experiment_csv,
        tese_root=tese_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(
        pretrained=(
            not args.no_pretrained
        )
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # For the 3-epoch smoke test we intentionally do not decay LR.
    # The purpose is diagnostic, not final convergence.
    scheduler = None

    if not args.smoke_test:
        scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=DEFAULT_MILESTONES,
                gamma=DEFAULT_GAMMA,
            )
        )

    amp_enabled, scaler = (
        make_amp_objects(device)
    )

    run_config = {
        "run_type": run_name,
        "target": args.target,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "optimizer": "SGD",
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "scheduler": (
            None
            if scheduler is None
            else "MultiStepLR"
        ),
        "scheduler_milestones": (
            None
            if scheduler is None
            else DEFAULT_MILESTONES
        ),
        "scheduler_gamma": (
            None
            if scheduler is None
            else DEFAULT_GAMMA
        ),
        "seed": args.seed,
        "image_size": 256,
        "rgb_only": True,
        "pretrained_imagenet": (
            not args.no_pretrained
        ),
        "checkpoint_metric": (
            "macro source-validation AUC"
        ),
        "classification_threshold": None,
        "amp": amp_enabled,
        "device": str(device),
        "train_samples": len(
            train_dataset
        ),
        "val_samples": len(
            val_dataset
        ),
        "experiment_csv": str(
            experiment_csv
        ),
    }

    save_run_config(
        config_path,
        run_config,
    )

    print("=" * 80)
    print("RESNET18 BASELINE")
    print("=" * 80)
    print("Run type:", run_name)
    print("Target:", args.target)
    print(
        "Target usage: TEST ONLY "
        "(não utilizado neste script)"
    )
    print("Device:", device)
    print("AMP:", amp_enabled)
    print(
        "ImageNet pretrained:",
        not args.no_pretrained,
    )
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Batch size:", args.batch_size)
    print("Epochs:", epochs)
    print("SGD lr:", args.lr)
    print("Momentum:", args.momentum)
    print(
        "Weight decay:",
        args.weight_decay,
    )

    if scheduler is None:
        print("Scheduler: none (smoke test)")
    else:
        print(
            "Scheduler: MultiStepLR "
            f"milestones={DEFAULT_MILESTONES}, "
            f"gamma={DEFAULT_GAMMA}"
        )

    print(
        "Checkpoint metric: "
        "macro source-validation AUC"
    )
    print(
        "Classification threshold: "
        "NOT selected"
    )
    print("=" * 80)

    best_macro_auc = float("-inf")
    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        epoch_start = time.perf_counter()

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            sampler=train_sampler,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        improved = (
            val_metrics["macro_auc"]
            > best_macro_auc
        )

        if improved:
            best_macro_auc = (
                val_metrics["macro_auc"]
            )

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_macro_auc=best_macro_auc,
                args=args,
                experiment_csv=experiment_csv,
            )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        domain_auc_text = ", ".join(
            (
                f"domain_{domain_id}="
                f"{auc:.4f}"
            )
            for domain_id, auc
            in sorted(
                val_metrics[
                    "domain_aucs"
                ].items()
            )
        )

        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": (
                    train_metrics["loss"]
                ),
                "train_accuracy": (
                    train_metrics["accuracy"]
                ),
                "val_loss": (
                    val_metrics["loss"]
                ),
                "val_macro_auc": (
                    val_metrics["macro_auc"]
                ),
                "val_pooled_auc": (
                    val_metrics["pooled_auc"]
                ),
                "val_domain_aucs": (
                    domain_auc_text
                ),
                "epoch_seconds": (
                    epoch_seconds
                ),
            }
        )

        # Write after every epoch so partial progress is not lost if the run is
        # manually stopped later.
        write_history(
            history_path,
            history,
        )

        marker = (
            "  <-- BEST"
            if improved
            else ""
        )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_auc={val_metrics['macro_auc']:.4f} | "
            f"val_pooled_auc={val_metrics['pooled_auc']:.4f} | "
            f"{domain_auc_text} | "
            f"{epoch_seconds:.1f}s"
            f"{marker}"
        )

        # Scheduler step occurs AFTER the epoch.
        # Therefore the displayed current_lr is the LR actually used for that
        # epoch.
        if scheduler is not None:
            scheduler.step()

    print()
    print("=" * 80)
    print("TRAINING FINISHED")
    print("=" * 80)
    print(
        "Best macro source-validation AUC:",
        f"{best_macro_auc:.4f}",
    )
    print(
        "Best checkpoint:",
        checkpoint_path,
    )
    print(
        "Training history:",
        history_path,
    )
    print(
        "Run config:",
        config_path,
    )
    print()
    print(
        "IMPORTANT: o unseen target ainda NÃO foi avaliado. "
        "O próximo passo será carregar este checkpoint e executar "
        "uma avaliação final separada no target."
    )


if __name__ == "__main__":
    main()
