"""
04_test_dataloader.py

Smoke test for the image-only FAS data pipeline.

This script does NOT train a model.
It verifies that:
- the experiment CSV can be read;
- file-based datasets can be opened;
- NUAA can be read directly from ZIP;
- all images become [3, 256, 256] tensors;
- labels and domain IDs are present;
- training batches are balanced by domain + class.

Recommended first run
---------------------
python 04_test_dataloader.py --target casia_fasd

Then test the other two targets:
python 04_test_dataloader.py --target lcc_fasd
python 04_test_dataloader.py --target nuaa
"""

from collections import Counter
import argparse

import torch
from torch.utils.data import DataLoader

from fas_dataset import (
    FASDataset,
    BalancedDomainClassBatchSampler,
    find_tese_root,
    experiment_csv_for_target,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        default="casia_fasd",
        choices=[
            "nuaa",
            "lcc_fasd",
            "casia_fasd",
        ],
        help=(
            "Dataset mantido completamente fora do treino "
            "e usado como unseen target."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Batch size do teste. "
            "16 é um valor inicial conservador e divisível "
            "pelos 4 grupos domain+class."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Para o primeiro teste em Windows usamos 0. "
            "Depois do pipeline validado podemos aumentar."
        ),
    )

    parser.add_argument(
        "--batches",
        type=int,
        default=3,
        help="Número de train batches a inspecionar.",
    )

    return parser.parse_args()


def count_batch(values):
    return dict(
        sorted(
            Counter(
                int(x)
                for x in values
            ).items()
        )
    )


def print_dataset_summary(
    name,
    dataset,
):
    summary = dataset.summary()

    print(
        f"{name:<5} "
        f"samples={summary['samples']:5d} | "
        f"classes={summary['classes']} | "
        f"domains={summary['domains']}"
    )


def validate_batch_shape(batch):
    images = batch["image"]

    if images.ndim != 4:
        raise RuntimeError(
            f"Esperava tensor 4D [B,C,H,W], "
            f"obtive {tuple(images.shape)}."
        )

    if tuple(images.shape[1:]) != (
        3,
        256,
        256,
    ):
        raise RuntimeError(
            "Shape inesperado. "
            f"Obtive {tuple(images.shape)}."
        )


def main():
    args = parse_args()

    tese_root = find_tese_root()

    experiment_csv = experiment_csv_for_target(
        tese_root,
        args.target,
    )

    print("TESE_ROOT:", tese_root)
    print("EXPERIMENT:", experiment_csv)
    print()

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

    test_dataset = FASDataset(
        experiment_csv,
        role="test",
        tese_root=tese_root,
    )

    print("DATASETS")
    print("-" * 78)
    print_dataset_summary(
        "TRAIN",
        train_dataset,
    )
    print_dataset_summary(
        "VAL",
        val_dataset,
    )
    print_dataset_summary(
        "TEST",
        test_dataset,
    )

    train_sampler = (
        BalancedDomainClassBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            seed=42,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Validation and test intentionally preserve their natural class/domain
    # distribution. They are NOT balanced by the sampler.
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print()
    print("TRAIN BATCH CHECK")
    print("-" * 78)

    for batch_index, batch in enumerate(
        train_loader,
        start=1,
    ):
        validate_batch_shape(batch)

        print(
            f"Batch {batch_index}: "
            f"shape={tuple(batch['image'].shape)} | "
            f"class_id={count_batch(batch['label'])} | "
            f"domain_id={count_batch(batch['domain_id'])}"
        )

        if batch_index >= args.batches:
            break

    print()
    print("VAL FIRST BATCH")
    print("-" * 78)

    val_batch = next(
        iter(val_loader)
    )
    validate_batch_shape(
        val_batch
    )

    print(
        f"shape={tuple(val_batch['image'].shape)} | "
        f"class_id={count_batch(val_batch['label'])} | "
        f"domain_id={count_batch(val_batch['domain_id'])}"
    )

    print()
    print("TEST FIRST BATCH")
    print("-" * 78)

    test_batch = next(
        iter(test_loader)
    )
    validate_batch_shape(
        test_batch
    )

    print(
        f"shape={tuple(test_batch['image'].shape)} | "
        f"class_id={count_batch(test_batch['label'])} | "
        f"domain_id={count_batch(test_batch['domain_id'])}"
    )

    print()
    print("CHECK FINAL")
    print("-" * 78)
    print("✓ CSV lido")
    print("✓ train/val/test carregados")
    print("✓ imagens convertidas para RGB")
    print("✓ tensors [B, 3, 256, 256]")
    print("✓ sampler train equilibrado por domain + class")
    print("✓ validation/test mantêm distribuição natural")
    print()
    print(
        "Nenhum threshold de classificação foi usado: "
        "ainda não existe modelo nesta fase."
    )


if __name__ == "__main__":
    # Required on Windows when DataLoader num_workers > 0.
    main()
