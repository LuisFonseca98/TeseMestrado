"""
Create the three preliminary image-only cross-domain experiments.

Experiments
-----------
1. NUAA + LCC-FASD   -> CASIA-FASD
2. NUAA + CASIA-FASD -> LCC-FASD
3. LCC-FASD + CASIA  -> NUAA

Critical rule
-------------
The target dataset is used ONLY as final test data.
It is never used for training, validation, model selection, hyperparameter
selection, or threshold selection.

This is an adapted leave-one-dataset-out protocol inspired by SA-FAS.
It is NOT presented as a reproduction of the original O-C-I-M benchmark.
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import random

# ---------------------------------------------------------------------------
# Fixed protocol values
# ---------------------------------------------------------------------------

# Reproducibility seed. 42 has no special scientific meaning; it is simply
# fixed so baseline and SA-FAS use the exact same subject split.
SEED = 42

# NUAA and CASIA have native train/test but no dedicated development split in
# the copies used here. We reserve 20% of TRAIN SUBJECTS for validation.
#
# Why 20%?
# - gives us a usable source-only validation partition;
# - retains most training identities for learning;
# - is fixed BEFORE seeing target results, so it cannot be tuned to the target.
VAL_SUBJECT_FRACTION = 0.20

DOMAINS = [
    "NUAA",
    "LCC_FASD",
    "CASIA_FASD",
]


def find_tese_root() -> Path:
    """Find project root through its Datasets folder; no absolute path needed."""
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "Datasets").exists():
            return candidate
    raise FileNotFoundError(
        "Não encontrei uma pasta 'Datasets'. Executa o script dentro do projeto TESE."
    )


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_train_subjects(rows):
    """
    Split by SUBJECT instead of splitting individual images randomly.

    Why?
    ----
    If the same person's images are present in both train and validation, the
    model may benefit from identity-specific cues. A subject-level split reduces
    that leakage risk.

    This split is only used inside a SOURCE dataset's official training set.
    """
    subjects = sorted(
        {
            row["subject_id"]
            for row in rows
            if row["subject_id"]
        }
    )

    if len(subjects) < 2:
        raise ValueError(
            "Não existem sujeitos suficientes para criar train/validation."
        )

    rng = random.Random(SEED)
    rng.shuffle(subjects)

    n_val = max(1, round(len(subjects) * VAL_SUBJECT_FRACTION))
    n_val = min(n_val, len(subjects) - 1)

    val_subjects = set(subjects[:n_val])

    roles = {}
    for row in rows:
        roles[row["sample_id"]] = (
            "val" if row["subject_id"] in val_subjects else "train"
        )

    return roles


def build_source_roles(all_rows, domain):
    """
    Define train/validation roles for a dataset when it is a SOURCE domain.

    LCC-FASD:
      training    -> train
      development -> validation
      evaluation  -> ignored while LCC is a source

    NUAA / CASIA-FASD:
      official train -> train/validation split by subject
      official test  -> ignored while the dataset is a source
    """
    domain_rows = [
        row for row in all_rows
        if row["domain"] == domain
    ]

    roles = {
        row["sample_id"]: "ignore"
        for row in domain_rows
    }

    if domain == "LCC_FASD":
        for row in domain_rows:
            if row["native_split"] == "training":
                roles[row["sample_id"]] = "train"
            elif row["native_split"] == "development":
                roles[row["sample_id"]] = "val"
            elif row["native_split"] == "evaluation":
                roles[row["sample_id"]] = "ignore"
        return roles

    if domain in {"NUAA", "CASIA_FASD"}:
        official_train = [
            row for row in domain_rows
            if row["native_split"] == "train"
        ]
        roles.update(split_train_subjects(official_train))
        return roles

    raise ValueError(f"Domínio desconhecido: {domain}")


def target_test_native_split(domain):
    """Return the official held-out partition used when the domain is target."""
    if domain == "LCC_FASD":
        return "evaluation"
    if domain in {"NUAA", "CASIA_FASD"}:
        return "test"
    raise ValueError(f"Domínio desconhecido: {domain}")


def build_experiment(all_rows, target_domain):
    """
    Build one experiment while keeping the target completely unseen until test.
    """
    source_domains = [
        domain for domain in DOMAINS
        if domain != target_domain
    ]

    source_roles = {}
    for domain in source_domains:
        source_roles.update(build_source_roles(all_rows, domain))

    target_native_split = target_test_native_split(target_domain)
    experiment_rows = []

    for original in all_rows:
        row = dict(original)

        if row["domain"] in source_domains:
            role = source_roles.get(row["sample_id"], "ignore")

        elif row["domain"] == target_domain:
            role = (
                "test"
                if row["native_split"] == target_native_split
                else "ignore"
            )

        else:
            role = "ignore"

        if role == "ignore":
            continue

        row["role"] = role
        row["target_domain"] = target_domain
        row["is_source"] = "1" if row["domain"] in source_domains else "0"
        experiment_rows.append(row)

    return experiment_rows


# ---------------------------------------------------------------------------
# Safety checks: fail early instead of discovering leakage after training.
# ---------------------------------------------------------------------------

def validate_no_target_leakage(rows, target_domain):
    for row in rows:
        if row["domain"] == target_domain and row["role"] != "test":
            raise RuntimeError(
                "LEAKAGE: encontrei uma amostra do target "
                f"{target_domain} com role={row['role']}."
            )


def validate_test_domain(rows, target_domain):
    test_domains = {
        row["domain"]
        for row in rows
        if row["role"] == "test"
    }
    if test_domains != {target_domain}:
        raise RuntimeError(
            f"TEST devia conter apenas {target_domain}; encontrei {test_domains}."
        )


def validate_source_domains(rows, target_domain):
    source_domains = {
        row["domain"]
        for row in rows
        if row["role"] in {"train", "val"}
    }
    expected = set(DOMAINS) - {target_domain}
    if source_domains != expected:
        raise RuntimeError(
            f"Sources inesperados. Esperado={expected}; obtido={source_domains}."
        )


def validate_binary_classes(rows):
    """Every train/val/test partition must contain both bona-fide and spoof."""
    for role in ["train", "val", "test"]:
        classes = {
            row["class_id"]
            for row in rows
            if row["role"] == role
        }
        if classes != {"0", "1"}:
            raise RuntimeError(
                f"{role}: classes={classes}; eram esperadas {{'0', '1'}}."
            )


def validate_subject_separation(rows):
    """
    Check source train/validation identity separation where reliable subject IDs
    are available. LCC is intentionally skipped.
    """
    for domain in ["NUAA", "CASIA_FASD"]:
        train_subjects = {
            row["subject_id"]
            for row in rows
            if (
                row["domain"] == domain
                and row["role"] == "train"
                and row["subject_id"]
            )
        }
        val_subjects = {
            row["subject_id"]
            for row in rows
            if (
                row["domain"] == domain
                and row["role"] == "val"
                and row["subject_id"]
            )
        }

        overlap = train_subjects & val_subjects
        if overlap:
            raise RuntimeError(
                f"{domain}: sujeitos repetidos entre train/val: {sorted(overlap)}"
            )


def validate_experiment(rows, target_domain):
    validate_no_target_leakage(rows, target_domain)
    validate_test_domain(rows, target_domain)
    validate_source_domains(rows, target_domain)
    validate_binary_classes(rows)
    validate_subject_separation(rows)


def write_csv(rows, path: Path):
    if not rows:
        raise ValueError("Não existem linhas para exportar.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, target_domain):
    print("\n" + "=" * 78)
    print(f"TARGET NÃO VISTO: {target_domain}")
    print("=" * 78)

    counts = defaultdict(Counter)
    for row in rows:
        counts[(row["role"], row["domain"])][row["label"]] += 1

    for role in ["train", "val", "test"]:
        print(f"\n{role.upper()}")
        for domain in DOMAINS:
            key = (role, domain)
            if key not in counts:
                continue
            c = counts[key]
            print(
                f"  {domain:<12} total={sum(c.values()):5d} | "
                f"bona_fide={c['bona_fide']:5d} | spoof={c['spoof']:5d}"
            )


def main():
    tese_root = find_tese_root()
    manifest_path = tese_root / "reports" / "fas_image_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Falta reports/fas_image_manifest.csv. "
            "Executa primeiro 01_build_image_manifest.py."
        )

    all_rows = read_csv(manifest_path)
    output_dir = tese_root / "reports" / "experiments"

    for target_domain in DOMAINS:
        rows = build_experiment(all_rows, target_domain)
        validate_experiment(rows, target_domain)

        filename = "target_" + target_domain.lower() + ".csv"
        output_path = output_dir / filename
        write_csv(rows, output_path)
        print_summary(rows, target_domain)
        print("\nCSV criado:", output_path)

    print("\n" + "-" * 78)
    print("PROTOCOLO FIXO")
    print("-" * 78)
    print(f"Seed: {SEED}")
    print(f"Validation subjects: {VAL_SUBJECT_FRACTION:.0%}")
    print("Threshold: NÃO definido nesta fase.")
    print(
        "Regra futura: qualquer threshold otimizado será escolhido apenas "
        "nos source validation data; nunca no unseen target test."
    )


if __name__ == "__main__":
    main()
