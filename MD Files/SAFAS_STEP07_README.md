# Step 07 — Adapted SA-FAS training

## Purpose

Train SA-FAS using exactly the same three-dataset experimental protocol used by the ResNet18 baseline.

This is an **adapted preliminary implementation**, not a reproduction of the official O-C-I-M benchmark.

## What stays the same as the baseline

- NUAA / LCC-FASD / CASIA-FASD experiment CSVs
- two source domains + one unseen target
- target never used during training or checkpoint selection
- RGB 256×256 input
- same conservative preprocessing
- same balanced domain+class sampler
- ImageNet-pretrained ResNet18 backbone
- batch size 16
- SGD, lr 0.005, momentum 0.9, weight decay 0.0005
- 30 epochs
- MultiStepLR milestones 15 and 25, gamma 0.5
- checkpoint selected by macro source-validation AUC

## What SA-FAS adds

### Separability

Two independently augmented views are produced for each training sample.
The supervised contrastive positives are samples with:

- the same class; and
- the same source domain.

Defaults:

- SupCon temperature = 0.1
- separability weight λ = 0.1

### Alignment

A separate binary live/spoof hyperplane is maintained for each source domain.
After the warm-up period, PG-IRM-style interpolation moves the domain-wise hyperplanes toward one another.

Defaults:

- alpha = 0.995
- alignment starts after epoch 20 in the full run

For the 3-epoch smoke test only, alignment starts after epoch 1 so the PG-IRM code path is actually exercised.

### Validation inference

The output logit is the mean of the source-domain classifier logits.

## Files

Place the script at:

```text
TESE/
└── Pipeline/
    └── 07_train_safas.py
```

## First command — smoke test

From the TESE root:

```powershell
python .\Pipeline\07_train_safas.py --target casia_fasd --smoke-test
```

Expected header includes:

```text
ADAPTED SA-FAS
Target: casia_fasd
Device: cuda
AMP: True
Epochs: 3
Separability weight (lambda): 0.1
SupCon temperature: 0.1
PG-IRM alpha: 0.995
PG-IRM starts AFTER epoch: 1
```

The three epoch lines should show:

- total train loss
- classification loss (`cls`)
- separability loss (`sep`)
- train accuracy
- validation macro AUC
- validation pooled AUC
- per-domain AUC
- classifier cosine similarity (`cos`)
- alignment OFF/ON

## Full CASIA-target run

Only after the smoke test is healthy:

```powershell
python .\Pipeline\07_train_safas.py --target casia_fasd
```

Outputs:

```text
checkpoints/safas/target_casia_fasd/best_safas.pt
results/safas/target_casia_fasd/training_history.csv
results/safas/target_casia_fasd/run_config.json
```

## Important

This script does **not** evaluate the unseen target. A separate SA-FAS evaluation script will load the best checkpoint, derive the threshold from source validation only, freeze it, and then evaluate the target exactly as we did for the ResNet18 baseline.
