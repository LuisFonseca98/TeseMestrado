# Step 06 — Final baseline evaluation

## Objective

Evaluate the selected ResNet18 baseline checkpoint on the unseen target
without using the target to make any model-development decision.

## Protocol

1. Load `best_resnet18.pt`.
2. Run inference on the source validation split.
3. Select a classification threshold using only the pooled source-validation
   data at the approximate Equal Error Rate (EER).
4. Freeze the threshold.
5. Only then load the target test split.
6. Report target AUC, APCER, BPCER, ACER, HTER, balanced accuracy,
   accuracy and confusion-matrix counts.

Labels:

- `0 = bona_fide`
- `1 = spoof`

With this convention:

- `BPCER` = bona fide samples classified as spoof.
- `APCER` = spoof samples classified as bona fide.

## CASIA target

From the TESE project root:

```powershell
python .\Pipeline\06_evaluate_baseline.py --target casia_fasd
```

Expected input checkpoint:

```text
checkpoints/
└── baseline/
    └── target_casia_fasd/
        └── best_resnet18.pt
```

Outputs:

```text
results/
└── baseline/
    └── target_casia_fasd/
        ├── final_evaluation.json
        ├── source_validation_predictions.csv
        └── target_test_predictions.csv
```

## Methodological rule

The target is not used for:

- training;
- checkpoint selection;
- hyperparameter tuning;
- threshold selection.

The classification threshold is selected on source validation and frozen before
the target test set is instantiated.
