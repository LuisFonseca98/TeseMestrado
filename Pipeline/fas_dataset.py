"""
03_fas_dataset.py

Unified PyTorch data layer for the image-only Face Anti-Spoofing experiments.

What this file does
-------------------
- Reads one of the experiment CSV files created by
  02_build_cross_domain_splits.py.
- Filters samples by role: train / val / test.
- Loads normal image files from disk.
- Loads NUAA images directly from its ZIP archive.
- Applies the same input size and normalization to all datasets.
- Provides a balanced training batch sampler by (domain, class).

What this file does NOT do
--------------------------
- It does not train a model.
- It does not choose a classification threshold.
- It does not use target-test data for any training decision.
- It does not use CASIA depth images.
- It does not perform video/temporal processing.

Labels used in the whole project
--------------------------------
0 = bona_fide
1 = spoof
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict, Counter
from io import BytesIO
import csv
import math
import random
import zipfile

from PIL import Image

import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


# ---------------------------------------------------------------------------
# Preprocessing constants
# ---------------------------------------------------------------------------

# SA-FAS reports 256x256 input crops. Keeping a common fixed size also allows
# images from NUAA, LCC-FASD and CASIA-FASD to be batched together.
IMAGE_SIZE = 256

# ImageNet normalization is used because the baseline will use a torchvision
# ResNet18 and may use ImageNet-pretrained weights.
#
# IMPORTANT:
# These are normalization constants, NOT classification thresholds.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Horizontal flip is applied only during training.
#
# p=0.5 means that each training image has a 50% probability of being flipped.
# This value is fixed before looking at target-test results and is a data
# augmentation probability, NOT a decision threshold.
TRAIN_HORIZONTAL_FLIP_P = 0.5


def find_tese_root(start: Path | None = None) -> Path:
    """
    Find the TESE project root by locating a 'Datasets' directory.

    This avoids hardcoded paths and allows the same scripts to run on the
    laptop and on the tower.
    """
    here = (start or Path.cwd()).resolve()

    for candidate in [here, *here.parents]:
        if (candidate / "Datasets").exists():
            return candidate

    raise FileNotFoundError(
        "Não encontrei a pasta 'Datasets'. "
        "Executa o script dentro da pasta TESE ou numa subpasta."
    )


def build_transforms(train: bool):
    """
    Build the common image preprocessing pipeline.

    Training:
      Resize -> random horizontal flip -> tensor -> normalization

    Validation/Test:
      Resize -> tensor -> normalization

    Why no aggressive augmentation yet?
    -----------------------------------
    The first goal is to build a clean baseline. Strong colour/geometry
    augmentations could themselves change cross-domain generalization, making
    it harder to understand whether later gains come from SA-FAS or from data
    augmentation. We start conservatively and can add stronger augmentation as
    a documented experiment later.
    """
    steps = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]

    if train:
        steps.append(
            transforms.RandomHorizontalFlip(
                p=TRAIN_HORIZONTAL_FLIP_P
            )
        )

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    return transforms.Compose(steps)


def read_csv_rows(csv_path: Path):
    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


class FASDataset(Dataset):
    """
    Unified dataset over NUAA, LCC-FASD and CASIA-FASD.

    Parameters
    ----------
    experiment_csv:
        CSV produced by 02_build_cross_domain_splits.py.

    role:
        One of: train, val, test.

    tese_root:
        Optional explicit project root. Normally auto-detected.

    transform:
        Optional torchvision transform. If omitted, the appropriate default
        transform is selected from the role.
    """

    def __init__(
        self,
        experiment_csv: str | Path,
        role: str,
        tese_root: str | Path | None = None,
        transform=None,
    ):
        self.experiment_csv = Path(experiment_csv).resolve()

        if role not in {"train", "val", "test"}:
            raise ValueError(
                "role tem de ser 'train', 'val' ou 'test'."
            )

        self.role = role
        self.tese_root = (
            Path(tese_root).resolve()
            if tese_root is not None
            else find_tese_root(
                self.experiment_csv.parent
            )
        )

        all_rows = read_csv_rows(
            self.experiment_csv
        )

        self.rows = [
            row
            for row in all_rows
            if row["role"] == role
        ]

        if not self.rows:
            raise ValueError(
                f"Não existem amostras com role='{role}' "
                f"em {self.experiment_csv}"
            )

        self.transform = (
            transform
            if transform is not None
            else build_transforms(
                train=(role == "train")
            )
        )

        # ZipFile handles cannot safely be shared across Windows DataLoader
        # worker processes. They are therefore opened lazily inside each
        # process and cached only there.
        self._zip_handles = {}

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        """
        Ensure DataLoader workers do not inherit/pickle open ZIP handles.
        """
        state = self.__dict__.copy()
        state["_zip_handles"] = {}
        return state

    def close(self):
        for handle in self._zip_handles.values():
            try:
                handle.close()
            except Exception:
                pass

        self._zip_handles.clear()

    def __del__(self):
        self.close()

    def _resolve_project_path(self, value: str) -> Path:
        path = Path(value)

        if path.is_absolute():
            return path

        return self.tese_root / path

    def _get_zip_handle(self, archive_path: Path):
        key = str(archive_path.resolve())

        if key not in self._zip_handles:
            self._zip_handles[key] = zipfile.ZipFile(
                archive_path,
                "r",
            )

        return self._zip_handles[key]

    def _load_image(self, row):
        storage = row["storage"]

        if storage == "file":
            path = self._resolve_project_path(
                row["filepath"]
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Imagem não encontrada: {path}"
                )

            # copy() detaches the PIL image from the file handle before it is
            # closed by the context manager.
            with Image.open(path) as image:
                return image.convert("RGB").copy()

        if storage == "zip":
            archive_path = (
                self._resolve_project_path(
                    row["archive_path"]
                )
            )

            if not archive_path.exists():
                raise FileNotFoundError(
                    f"ZIP não encontrado: {archive_path}"
                )

            zf = self._get_zip_handle(
                archive_path
            )

            try:
                raw = zf.read(
                    row["archive_member"]
                )
            except KeyError as exc:
                raise FileNotFoundError(
                    "Imagem não encontrada dentro do ZIP: "
                    f"{row['archive_member']}"
                ) from exc

            with Image.open(BytesIO(raw)) as image:
                return image.convert("RGB").copy()

        raise ValueError(
            f"Tipo de storage desconhecido: {storage}"
        )

    def __getitem__(self, index):
        row = self.rows[index]
        image = self._load_image(row)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(
                int(row["class_id"]),
                dtype=torch.long,
            ),
            "domain_id": torch.tensor(
                int(row["domain_id"]),
                dtype=torch.long,
            ),
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "native_split": row["native_split"],
            "attack_type": row["attack_type"],
            "quality": row["quality"],
            "subject_id": row["subject_id"],
        }

    def summary(self):
        labels = Counter(
            int(row["class_id"])
            for row in self.rows
        )
        domains = Counter(
            row["domain"]
            for row in self.rows
        )

        return {
            "role": self.role,
            "samples": len(self.rows),
            "classes": dict(labels),
            "domains": dict(domains),
        }


class BalancedDomainClassBatchSampler(Sampler[list[int]]):
    """
    Produce training batches balanced by BOTH source domain and class.

    Example for a two-source experiment
    -----------------------------------
    Suppose the sources are NUAA and LCC-FASD. There are four groups:

      NUAA + bona_fide
      NUAA + spoof
      LCC  + bona_fide
      LCC  + spoof

    With batch_size=16, this sampler returns approximately:

      4 NUAA bona_fide
      4 NUAA spoof
      4 LCC bona_fide
      4 LCC spoof

    Why do this?
    ------------
    LCC-FASD is strongly imbalanced. A normal shuffled DataLoader would allow
    LCC spoof images to dominate many batches.

    This is especially undesirable for SA-FAS because its supervised
    contrastive component relies on useful domain+class groups inside a batch.

    Epoch length
    ------------
    We keep approximately the same number of samples per epoch as the original
    training dataset:

        steps_per_epoch = ceil(dataset_size / batch_size)

    Smaller groups are cycled/reshuffled when necessary. Therefore some
    samples from minority groups can repeat within an epoch.

    This is an explicit training choice and must be documented in the report.
    The original manifest itself remains untouched and retains the natural
    dataset distribution.
    """

    def __init__(
        self,
        dataset: FASDataset,
        batch_size: int = 16,
        seed: int = 42,
    ):
        if dataset.role != "train":
            raise ValueError(
                "BalancedDomainClassBatchSampler só deve "
                "ser usado no conjunto train."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size tem de ser > 0."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        grouped = defaultdict(list)

        for index, row in enumerate(
            dataset.rows
        ):
            key = (
                int(row["domain_id"]),
                int(row["class_id"]),
            )
            grouped[key].append(index)

        self.groups = dict(
            sorted(grouped.items())
        )

        if not self.groups:
            raise ValueError(
                "Não foram encontrados grupos domain+class."
            )

        # In the current experiments there should be:
        # 2 source domains x 2 classes = 4 groups.
        self.num_groups = len(self.groups)

        if self.batch_size % self.num_groups != 0:
            raise ValueError(
                f"batch_size={self.batch_size} não é divisível "
                f"pelos {self.num_groups} grupos domain+class. "
                "Usa, por exemplo, 8, 16, 32, ..."
            )

        self.samples_per_group = (
            self.batch_size
            // self.num_groups
        )

        self.steps_per_epoch = math.ceil(
            len(dataset)
            / self.batch_size
        )

        for key, indices in self.groups.items():
            if len(indices) == 0:
                raise ValueError(
                    f"Grupo vazio: {key}"
                )

    def set_epoch(self, epoch: int):
        """
        Optional hook for reproducible but different sampling each epoch.
        """
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(
            self.seed + self.epoch
        )

        # Create independently shuffled pools for every domain+class group.
        pools = {}
        positions = {}

        for key, indices in self.groups.items():
            pool = list(indices)
            rng.shuffle(pool)
            pools[key] = pool
            positions[key] = 0

        def take_from_group(key, amount):
            selected = []

            while len(selected) < amount:
                pool = pools[key]
                pos = positions[key]

                # Once a group is exhausted, reshuffle it and continue.
                # This is how minority groups can be sampled more than once
                # within an epoch without permanently duplicating the manifest.
                if pos >= len(pool):
                    pool = list(
                        self.groups[key]
                    )
                    rng.shuffle(pool)
                    pools[key] = pool
                    positions[key] = 0
                    pos = 0

                remaining = (
                    amount - len(selected)
                )
                available = (
                    len(pool) - pos
                )
                take = min(
                    remaining,
                    available,
                )

                selected.extend(
                    pool[pos : pos + take]
                )
                positions[key] += take

            return selected

        group_keys = list(
            self.groups.keys()
        )

        for _ in range(
            self.steps_per_epoch
        ):
            batch = []

            for key in group_keys:
                batch.extend(
                    take_from_group(
                        key,
                        self.samples_per_group,
                    )
                )

            # Shuffle positions inside the batch so examples are not ordered
            # by domain/class when passed to the model.
            rng.shuffle(batch)

            yield batch


def experiment_csv_for_target(
    tese_root: Path,
    target: str,
) -> Path:
    """
    Resolve a convenient target name to the generated experiment CSV.
    """
    normalized = target.strip().lower()

    aliases = {
        "nuaa": "target_nuaa.csv",
        "lcc": "target_lcc_fasd.csv",
        "lcc_fasd": "target_lcc_fasd.csv",
        "casia": "target_casia_fasd.csv",
        "casia_fasd": "target_casia_fasd.csv",
    }

    if normalized not in aliases:
        raise ValueError(
            "Target inválido. Usa: "
            "nuaa, lcc_fasd ou casia_fasd."
        )

    path = (
        tese_root
        / "reports"
        / "experiments"
        / aliases[normalized]
    )

    if not path.exists():
        raise FileNotFoundError(
            f"CSV experimental não encontrado: {path}"
        )

    return path
