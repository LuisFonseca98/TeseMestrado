# Image-only FAS — Manifest and Cross-Domain Protocol

## Scope

This phase is intentionally restricted to **images**.

Datasets used for the preliminary implementation:

- NUAA
- LCC-FASD
- CASIA-FASD (RGB only)

Not used in the first implementation:

- CASIA depth modality
- AxonData video attacks
- video-level temporal analysis
- ECLIPS implementation
- O-C-I-M datasets that still require access

The immediate goal is to obtain a controlled comparison between a ResNet18
baseline and the adapted SA-FAS implementation.

## 1. Canonical manifest

Run:

```powershell
python .\01_build_image_manifest.py
```

Output:

```text
reports/fas_image_manifest.csv
```

The manifest is the single source of truth for image metadata.

Important columns:

```text
sample_id
dataset
domain
domain_id
native_split
label
class_id
subject_id
attack_type
quality
session
storage
filepath
archive_path
archive_member
```

Binary labels:

```text
0 = bona_fide
1 = spoof
```

Domain IDs:

```text
0 = NUAA
1 = LCC_FASD
2 = CASIA_FASD
```

The numerical domain IDs are identifiers only; they have no ordering meaning.

## 2. Why the manifest does not balance classes

The manifest records the real local data distribution.

LCC-FASD is strongly imbalanced, but under/oversampling directly in the CSV
would hide the original distribution and make later analysis less transparent.

Balancing belongs in the training DataLoader/Sampler. This will also be useful
for SA-FAS because supervised contrastive learning needs useful combinations of
class and domain examples inside each mini-batch.

## 3. Why paths are relative

No hardcoded Windows path is used.

The scripts locate the project through its `Datasets` directory and store paths
relative to the project root. This is intentional so the same code can work on
both the tower and the laptop, even if the absolute TESE path differs.

## 4. Dataset-specific rules

### NUAA

The official split files inside `Detectedface.zip` are used.

When NUAA is a **source**:

```text
official train
    -> subject-level train/validation split
```

When NUAA is the **unseen target**:

```text
official test -> final test
```

The ZIP is kept compressed to avoid duplicating the dataset.

### LCC-FASD

Its native partitions are preserved.

When LCC-FASD is a **source**:

```text
training    -> train
development -> validation
evaluation  -> unused
```

When LCC-FASD is the **unseen target**:

```text
evaluation -> final test
```

No subject IDs or detailed attack types are invented because the currently
available metadata does not support doing so reliably.

### CASIA-FASD

Only RGB images are used.

Depth is deliberately excluded because NUAA and LCC-FASD do not provide the
same modality. Keeping all domains RGB-only makes the comparison cleaner.

When CASIA-FASD is a **source**:

```text
official train
    -> subject-level train/validation split
```

When CASIA-FASD is the **unseen target**:

```text
official test -> final test
```

Numeric subject identifiers are prefixed with `train_` or `test_` because the
same number in both native splits does not represent the same identity in the
analysed copy.

## 5. Cross-domain experiments

Run:

```powershell
python .\02_build_cross_domain_splits.py
```

Outputs:

```text
reports/experiments/
├── target_casia_fasd.csv
├── target_lcc_fasd.csv
└── target_nuaa.csv
```

The three experiments are:

```text
1. NUAA + LCC-FASD   -> CASIA-FASD
2. NUAA + CASIA-FASD -> LCC-FASD
3. LCC-FASD + CASIA  -> NUAA
```

This is an **adapted leave-one-dataset-out cross-domain protocol inspired by
SA-FAS**. It is not presented as a reproduction of the original O-C-I-M
benchmark.

## 6. Leakage rule

The most important rule is:

```text
UNSEEN TARGET
    -> test only
```

The target dataset must not be used for:

- training;
- validation;
- hyperparameter tuning;
- model/epoch selection;
- classification-threshold selection.

Otherwise the target is no longer genuinely unseen, so the experiment stops
measuring Domain Generalization correctly.

## 7. Validation split

For NUAA and CASIA-FASD, the native training set is split by **subject**:

```text
80% training subjects -> train
20% training subjects -> validation
```

Configured as:

```python
VAL_SUBJECT_FRACTION = 0.20
```

Why subject-level rather than image-level?

If images from the same person appear in both train and validation, the model
may exploit identity-specific information. A subject-level split reduces this
leakage risk.

Why 20%?

- it gives a usable source-only validation set;
- it preserves most training identities for learning;
- it is fixed before observing target results.

This value is a preliminary-study protocol choice and should be reported.

## 8. Random seed

The subject split uses:

```python
SEED = 42
```

The value 42 has no special statistical meaning. It is only fixed for
reproducibility so that baseline and SA-FAS use exactly the same data split.

For the future thesis, repeated seeds can be introduced to quantify variation.

## 9. Classification threshold

**No prediction threshold is defined by these manifest scripts.**

Threshold selection belongs to the later evaluation stage.

The rule will be:

```text
training data
    -> learn weights

source validation data
    -> choose checkpoint and, if required, operating threshold

unseen target test
    -> final evaluation only
```

If we later use an optimized threshold, the exact selection rule will be
written in the code and report. The unseen target will not be used to tune it.

A fixed `0.5` threshold can also be reported as a predefined operating point,
but it must be clearly distinguished from a threshold optimized on validation.

## 10. Expected counts

The manifest builder checks:

```text
NUAA        12,614
LCC-FASD    18,827
CASIA-FASD   4,063 RGB
-------------------
TOTAL       35,504 images
```

If a count differs, the script intentionally stops. We should not begin model
training before explaining whether the local copy is incomplete, duplicated,
or structurally different from the dataset version already analysed.

## 11. Suggested project placement

```text
TESE/
├── Datasets/
├── reports/
├── 01_build_image_manifest.py
└── 02_build_cross_domain_splits.py
```

From the TESE root:

```powershell
python .\01_build_image_manifest.py
python .\02_build_cross_domain_splits.py
```

## 12. Next implementation stage

After checking the generated CSVs:

1. implement a PyTorch `Dataset` that supports both normal files and NUAA ZIP;
2. define image preprocessing;
3. build a class/domain-aware training sampler;
4. implement the ResNet18 baseline;
5. run the three baseline experiments;
6. implement SupCon;
7. implement PG-IRM;
8. combine them into SA-FAS;
9. run the same three experiments unchanged;
10. compare baseline vs. SA-FAS using the same splits.

The manifests and splits must remain unchanged between baseline and SA-FAS.
