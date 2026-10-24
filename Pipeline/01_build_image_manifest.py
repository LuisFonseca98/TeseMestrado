"""
Build the canonical image-only manifest for the current FAS study.

Design decisions
----------------
- Current scope is IMAGE ONLY.
- NUAA is read from Detectedface.zip.
- LCC-FASD uses its image folders.
- CASIA-FASD uses only RGB JPG frames; depth is excluded.
- AxonData is excluded from the first implementation because its useful spoof
  material is video-based in the public sample.
- Paths are stored relative to the TESE project so the same manifest logic can
  work on the tower and the laptop.
- No class balancing is performed here. The manifest records the real data.
- No prediction threshold is defined here. Threshold selection belongs to the
  later validation/evaluation stage and must never use the unseen target test.
"""

from pathlib import Path
from collections import Counter
import csv
import zipfile

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Counts validated during the previous dataset analysis. If one differs, stop
# instead of silently training on an incomplete/duplicated local copy.
EXPECTED_COUNTS = {
    "NUAA": 12614,
    "LCC_FASD": 18827,
    "CASIA_FASD": 4063,
}

# Stable numeric identifiers for the later PyTorch pipeline.
# The values have no ranking/semantic meaning.
DOMAIN_IDS = {
    "NUAA": 0,
    "LCC_FASD": 1,
    "CASIA_FASD": 2,
}


def find_tese_root() -> Path:
    """Find the nearest parent containing a Datasets directory."""
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "Datasets").exists():
            return candidate
    raise FileNotFoundError(
        "Não encontrei uma pasta 'Datasets'. Executa o script dentro do projeto TESE."
    )


def relative_to_root(path: Path, root: Path) -> str:
    """Store portable paths relative to TESE whenever possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def find_first_dir(root: Path, folder_name: str) -> Path:
    """Find the shallowest matching directory, tolerating wrapper folders."""
    matches = sorted(
        (p for p in root.rglob(folder_name) if p.is_dir()),
        key=lambda p: (len(p.parts), str(p).lower()),
    )
    if not matches:
        raise FileNotFoundError(
            f"Não encontrei a pasta '{folder_name}' dentro de {root}."
        )
    return matches[0]


def find_first_file(root: Path, filename: str) -> Path:
    matches = sorted(
        (p for p in root.rglob(filename) if p.is_file()),
        key=lambda p: (len(p.parts), str(p).lower()),
    )
    if not matches:
        raise FileNotFoundError(
            f"Não encontrei o ficheiro '{filename}' dentro de {root}."
        )
    return matches[0]


# ---------------------------------------------------------------------------
# NUAA
# ---------------------------------------------------------------------------

def parse_nuaa_filename(filename: str):
    """
    NUAA detected-face names follow:
      subject_glasses_position_session_picture.jpg

    Example:
      0001_00_00_01_0.jpg
    """
    parts = Path(filename).stem.split("_")
    if len(parts) != 5:
        raise ValueError(f"Nome NUAA inesperado: {filename}")

    subject, _glasses, _position, session, _picture = parts
    return {
        "subject_id": subject,
        "session": session,
    }


def build_nuaa_records(tese_root: Path, datasets_root: Path):
    """
    Read NUAA directly from Detectedface.zip.

    Why keep the ZIP?
    -----------------
    It avoids duplicating the dataset. Later, the PyTorch Dataset can read the
    archive using archive_path + archive_member.
    """
    # The NUAA archive may have been downloaded/saved with a different name
    # (for example NUAA.zip instead of Detectedface.zip). Rather than relying
    # on the filename, inspect ZIP files and choose the one that actually
    # contains the official NUAA split index files.
    required_index_files = {
        "client_train_face.txt",
        "client_test_face.txt",
        "imposter_train_face.txt",
        "imposter_test_face.txt",
    }

    zip_candidates = sorted(
        [p for p in datasets_root.rglob("*.zip") if p.is_file()],
        key=lambda p: (len(p.parts), str(p).lower()),
    )

    zip_path = None
    for candidate in zip_candidates:
        try:
            with zipfile.ZipFile(candidate) as candidate_zip:
                basenames = {
                    Path(name.replace("\\", "/")).name
                    for name in candidate_zip.namelist()
                }
            if required_index_files.issubset(basenames):
                zip_path = candidate
                break
        except zipfile.BadZipFile:
            continue

    if zip_path is None:
        raise FileNotFoundError(
            "Não encontrei um ZIP do NUAA com os quatro ficheiros oficiais "
            "de split (client/imposter train/test) dentro de Datasets."
        )

    print("NUAA ZIP encontrado:", zip_path)

    # Use the official index files rather than inferring the split ourselves.
    split_files = [
        ("client_train_face.txt", "bona_fide", 0, "train", "ClientFace", "none"),
        ("client_test_face.txt", "bona_fide", 0, "test", "ClientFace", "none"),
        ("imposter_train_face.txt", "spoof", 1, "train", "ImposterFace", "print"),
        ("imposter_test_face.txt", "spoof", 1, "test", "ImposterFace", "print"),
    ]

    records = []

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()

        def find_member_ending(ending: str) -> str:
            normalized = ending.replace("\\", "/")
            matches = [
                member
                for member in members
                if member.replace("\\", "/").endswith(normalized)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"NUAA: esperava 1 membro terminado em '{normalized}', "
                    f"mas encontrei {len(matches)}."
                )
            return matches[0]

        for (
            index_name,
            label,
            class_id,
            native_split,
            class_folder,
            attack_type,
        ) in split_files:
            index_member = find_member_ending(index_name)
            content = zf.read(index_member).decode("latin1", errors="ignore")

            for line in content.splitlines():
                if not line.strip():
                    continue

                relative = line.split()[0].replace("\\", "/")
                filename = Path(relative).name
                meta = parse_nuaa_filename(filename)

                # Resolve now so a broken/missing file is caught before training.
                member = find_member_ending(f"{class_folder}/{relative}")

                records.append(
                    {
                        "dataset": "NUAA",
                        "domain": "NUAA",
                        "domain_id": DOMAIN_IDS["NUAA"],
                        "native_split": native_split,
                        "label": label,
                        "class_id": class_id,
                        "subject_id": meta["subject_id"],
                        "attack_type": attack_type,
                        "quality": "",
                        "session": meta["session"],
                        "storage": "zip",
                        "filepath": "",
                        "archive_path": relative_to_root(zip_path, tese_root),
                        "archive_member": member,
                    }
                )

    return records


# ---------------------------------------------------------------------------
# LCC-FASD
# ---------------------------------------------------------------------------

def build_lcc_records(tese_root: Path, datasets_root: Path):
    """
    Preserve the native LCC partitions: training/development/evaluation.

    We deliberately do NOT invent subject IDs or detailed attack types because
    the currently available metadata does not support doing so reliably.
    """
    split_names = {
        "training": "LCC_FASD_training",
        "development": "LCC_FASD_development",
        "evaluation": "LCC_FASD_evaluation",
    }

    split_paths = {
        split: find_first_dir(datasets_root, folder_name)
        for split, folder_name in split_names.items()
    }

    records = []

    for native_split, split_path in split_paths.items():
        for folder_name, label, class_id in [
            ("real", "bona_fide", 0),
            ("spoof", "spoof", 1),
        ]:
            class_path = split_path / folder_name
            if not class_path.exists():
                raise FileNotFoundError(f"LCC-FASD: pasta em falta: {class_path}")

            for file in sorted(class_path.rglob("*")):
                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append(
                        {
                            "dataset": "LCC_FASD",
                            "domain": "LCC_FASD",
                            "domain_id": DOMAIN_IDS["LCC_FASD"],
                            "native_split": native_split,
                            "label": label,
                            "class_id": class_id,
                            "subject_id": "",
                            "attack_type": "none" if class_id == 0 else "spoof_unknown",
                            "quality": "",
                            "session": "",
                            "storage": "file",
                            "filepath": relative_to_root(file, tese_root),
                            "archive_path": "",
                            "archive_member": "",
                        }
                    )

    return records


# ---------------------------------------------------------------------------
# CASIA-FASD
# ---------------------------------------------------------------------------

# Mapping already validated during the CASIA dataset analysis.
CASIA_VIDEO_INFO = {
    "1": ("bona_fide", "none", "normal"),
    "2": ("bona_fide", "none", "low"),
    "HR_1": ("bona_fide", "none", "high"),
    "3": ("spoof", "warped_photo", "normal"),
    "4": ("spoof", "warped_photo", "low"),
    "HR_2": ("spoof", "warped_photo", "high"),
    "5": ("spoof", "cut_photo", "normal"),
    "6": ("spoof", "cut_photo", "low"),
    "HR_3": ("spoof", "cut_photo", "high"),
    "7": ("spoof", "video_replay", "normal"),
    "8": ("spoof", "video_replay", "low"),
    "HR_4": ("spoof", "video_replay", "high"),
}


def parse_casia_filename(filename: str):
    """
    Parse the pre-extracted RGB JPG filename.

    Although some images originated from videos, this phase treats the provided
    JPG frames as image samples only. No temporal/video processing is done.
    """
    stem = filename.removesuffix(".jpg")

    try:
        before_avi, _after_avi = stem.split(".avi_")
    except ValueError as exc:
        raise ValueError(f"Nome CASIA inesperado: {filename}") from exc

    left = before_avi.split("_")
    subject = left[0]

    if len(left) >= 3 and left[1] == "HR":
        video_code = f"HR_{left[2]}"
    else:
        video_code = left[1]

    if video_code not in CASIA_VIDEO_INFO:
        raise ValueError(f"Código CASIA desconhecido: {video_code}")

    label, attack_type, quality = CASIA_VIDEO_INFO[video_code]

    return {
        "subject": subject,
        "label": label,
        "class_id": 0 if label == "bona_fide" else 1,
        "attack_type": attack_type,
        "quality": quality,
    }


def build_casia_records(tese_root: Path, datasets_root: Path):
    """
    Use ONLY the CASIA RGB 'color' folders.

    Why exclude depth?
    ------------------
    NUAA and LCC-FASD do not provide the same depth modality. Keeping all three
    domains RGB-only avoids giving CASIA information unavailable elsewhere.
    """
    dataset_path = find_first_dir(datasets_root, "CASIA_FASD")

    train_matches = sorted(
        [p for p in dataset_path.rglob("train_img") if (p / "color").exists()],
        key=lambda p: (len(p.parts), str(p).lower()),
    )
    test_matches = sorted(
        [p for p in dataset_path.rglob("test_img") if (p / "color").exists()],
        key=lambda p: (len(p.parts), str(p).lower()),
    )

    if not train_matches or not test_matches:
        raise FileNotFoundError(
            "CASIA-FASD: não encontrei train_img/color e test_img/color."
        )

    folders = {
        "train": train_matches[0] / "color",
        "test": test_matches[0] / "color",
    }

    records = []

    for native_split, folder in folders.items():
        for file in sorted(folder.glob("*.jpg")):
            meta = parse_casia_filename(file.name)

            records.append(
                {
                    "dataset": "CASIA_FASD",
                    "domain": "CASIA_FASD",
                    "domain_id": DOMAIN_IDS["CASIA_FASD"],
                    "native_split": native_split,
                    "label": meta["label"],
                    "class_id": meta["class_id"],
                    # Numeric IDs are local to train/test in this copy.
                    "subject_id": f"{native_split}_{meta['subject']}",
                    "attack_type": meta["attack_type"],
                    "quality": meta["quality"],
                    "session": "",
                    "storage": "file",
                    "filepath": relative_to_root(file, tese_root),
                    "archive_path": "",
                    "archive_member": "",
                }
            )

    return records


# ---------------------------------------------------------------------------
# Validation + export
# ---------------------------------------------------------------------------

def add_sample_ids(records):
    """
    Stable deterministic IDs help guarantee that baseline and SA-FAS reference
    exactly the same samples.
    """
    records.sort(
        key=lambda r: (
            r["dataset"],
            r["native_split"],
            r["label"],
            r["filepath"],
            r["archive_member"],
        )
    )

    for index, record in enumerate(records, start=1):
        record["sample_id"] = f"img_{index:06d}"


def validate_counts(records):
    counts = Counter(r["dataset"] for r in records)
    for dataset, expected in EXPECTED_COUNTS.items():
        found = counts[dataset]
        if found != expected:
            raise RuntimeError(
                f"{dataset}: esperava {expected} imagens, mas encontrei {found}. "
                "Não continues para o treino antes de perceber esta diferença."
            )


def validate_classes(records):
    for domain in DOMAIN_IDS:
        classes = {
            int(r["class_id"])
            for r in records
            if r["domain"] == domain
        }
        if classes != {0, 1}:
            raise RuntimeError(
                f"{domain}: classes encontradas = {classes}; esperadas = {{0, 1}}."
            )


def write_manifest(records, output_path: Path):
    fieldnames = [
        "sample_id",
        "dataset",
        "domain",
        "domain_id",
        "native_split",
        "label",
        "class_id",
        "subject_id",
        "attack_type",
        "quality",
        "session",
        "storage",
        "filepath",
        "archive_path",
        "archive_member",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_summary(records):
    print("\nResumo do manifest")
    print("-" * 72)

    for dataset in DOMAIN_IDS:
        subset = [r for r in records if r["dataset"] == dataset]
        labels = Counter(r["label"] for r in subset)
        print(
            f"{dataset:<12} total={len(subset):5d} | "
            f"bona_fide={labels['bona_fide']:5d} | spoof={labels['spoof']:5d}"
        )

    print("-" * 72)
    print("TOTAL:", len(records))


def main():
    tese_root = find_tese_root()
    datasets_root = tese_root / "Datasets"
    reports_root = tese_root / "reports"

    print("TESE_ROOT:", tese_root)
    print("DATASETS:", datasets_root)
    print()

    all_records = []

    print("1/3 A ler NUAA...")
    all_records.extend(build_nuaa_records(tese_root, datasets_root))

    print("2/3 A ler LCC-FASD...")
    all_records.extend(build_lcc_records(tese_root, datasets_root))

    print("3/3 A ler CASIA-FASD RGB...")
    all_records.extend(build_casia_records(tese_root, datasets_root))

    add_sample_ids(all_records)
    validate_counts(all_records)
    validate_classes(all_records)

    output_path = reports_root / "fas_image_manifest.csv"
    write_manifest(all_records, output_path)
    print_summary(all_records)

    print("\nManifest criado com sucesso:")
    print(output_path)
    print(
        "\nNOTA: este script NÃO define threshold de classificação. "
        "Esse valor será escolhido mais tarde apenas nos source validation data."
    )


if __name__ == "__main__":
    main()
