# Step 03-04 — PyTorch data pipeline

Place these files in the same folder where you keep the previous scripts:

```text
TESE/
├── Datasets/
├── reports/
├── fas_dataset.py
├── 03_fas_dataset.py
└── 04_test_dataloader.py
```

`fas_dataset.py` contains the real implementation.

`03_fas_dataset.py` exists only to preserve the numbered project order. Python
cannot normally import a module whose name starts with a digit, so
`04_test_dataloader.py` imports `fas_dataset.py`.

## Dependencies

Inside the TESE virtual environment:

```powershell
python -m pip install torch torchvision pillow
```

If PyTorch with CUDA has already been installed, do not reinstall it just for
this step.

## First validation

Run:

```powershell
python .\04_test_dataloader.py --target casia_fasd
```

Expected train batches with the default batch size 16:

```text
class_id={0: 8, 1: 8}
```

and, because there are two source domains:

```text
domain_id={...: 8, ...: 8}
```

Internally this corresponds to four equal domain+class groups:

```text
4 samples source A + bona fide
4 samples source A + spoof
4 samples source B + bona fide
4 samples source B + spoof
```

Then validate the other experiments:

```powershell
python .\04_test_dataloader.py --target lcc_fasd
python .\04_test_dataloader.py --target nuaa
```

## Values currently fixed

```text
IMAGE_SIZE = 256
TRAIN_HORIZONTAL_FLIP_P = 0.5
SAMPLER_SEED = 42
DEFAULT TEST BATCH SIZE = 16
```

The batch size 16 is initially conservative for pipeline validation. It is not
yet a final model hyperparameter.

No classification threshold exists yet. Threshold/model-checkpoint decisions
will be made later using source validation data only.
