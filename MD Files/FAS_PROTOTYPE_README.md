# Face Anti-Spoofing Prototype

## 1. Copy the application

Place:

```text
09_fas_prototype.py
```

inside:

```text
TESE\Pipeline\
```

The application expects the same project structure used by the training scripts:

```text
TESE/
├── Datasets/
├── Pipeline/
│   └── 09_fas_prototype.py
├── checkpoints/
│   ├── baseline/
│   └── safas/
└── results/
    ├── baseline/
    └── safas/
```

## 2. Install Streamlit

With the project virtual environment activated:

```powershell
python -m pip install streamlit
```

This only needs to be done once.

## 3. Run the prototype

From the TESE project root:

```powershell
python -m streamlit run .\Pipeline\09_fas_prototype.py
```

Streamlit should open the application in the browser.

## 4. Prototype workflow

1. Select one experiment profile:
   - NUAA + LCC → CASIA
   - NUAA + CASIA → LCC
   - LCC + CASIA → NUAA
2. Choose:
   - Compare both
   - ResNet18 baseline
   - SA-FAS
3. Upload an image.
4. Click **Analyse image**.
5. The application shows:
   - BONA FIDE / SPOOF
   - spoof score
   - source-validation EER threshold
   - distance from threshold
   - checkpoint information

## Important

The selected profile determines which trained checkpoint is used.

Example:

```text
NUAA + LCC → CASIA
```

uses a model trained without CASIA. CASIA was the unseen target in the
corresponding experiment.

The prototype uses the source-validation threshold already saved in:

```text
results\<model>\target_<target>\final_evaluation.json
```

It does not calculate a new threshold from the uploaded image.

This is an experimental academic proof of concept, not a production
biometric-security system.
