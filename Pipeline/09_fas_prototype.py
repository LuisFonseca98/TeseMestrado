"""
09_fas_prototype.py

Streamlit proof-of-concept for the image-based Face Anti-Spoofing project.

The application:
- accepts one RGB image;
- lets the user select one of the three leave-one-dataset-out experiment profiles;
- runs the ResNet18 baseline, adapted SA-FAS, or both;
- uses the SOURCE-validation EER threshold saved by the corresponding
  final evaluation;
- displays the spoof score and BONA FIDE / SPOOF decision.

This is an experimental academic prototype, not a production biometric system.
"""

from __future__ import annotations

from pathlib import Path
import json

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


IMAGE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PROFILES = {
    "NUAA + LCC → CASIA": {
        "target": "casia_fasd",
        "sources": "NUAA + LCC-FASD",
        "target_name": "CASIA-FASD",
    },
    "NUAA + CASIA → LCC": {
        "target": "lcc_fasd",
        "sources": "NUAA + CASIA-FASD",
        "target_name": "LCC-FASD",
    },
    "LCC + CASIA → NUAA": {
        "target": "nuaa",
        "sources": "LCC-FASD + CASIA-FASD",
        "target_name": "NUAA",
    },
}


def find_project_root() -> Path:
    """
    Find the project root using the Datasets directory.

    Works when Streamlit is started from the project root or a subdirectory.
    """
    candidates = [
        Path.cwd().resolve(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
    ]

    checked = set()

    for start in candidates:
        for candidate in [start, *start.parents]:
            candidate = candidate.resolve()
            if candidate in checked:
                continue
            checked.add(candidate)

            if (candidate / "Datasets").exists():
                return candidate

    raise FileNotFoundError(
        "Não encontrei a raiz do projeto (pasta Datasets). "
        "Executa a aplicação dentro da pasta TESE ou de uma subpasta."
    )


def build_preprocess():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


class SAFASResNet18(nn.Module):
    """ResNet18 encoder + one binary hyperplane per source domain."""

    def __init__(self, source_domain_ids):
        super().__init__()

        self.source_domain_ids = sorted(
            int(x) for x in source_domain_ids
        )

        backbone = resnet18(weights=None)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        self.classifiers = nn.ModuleDict(
            {
                str(domain_id): nn.Linear(
                    self.feature_dim,
                    1,
                    bias=True,
                )
                for domain_id in self.source_domain_ids
            }
        )

    def encode(self, x):
        features = self.encoder(x)
        return F.normalize(
            features,
            p=2,
            dim=1,
        )

    def mean_logits(self, features):
        logits = [
            self.classifiers[str(domain_id)](
                features
            ).squeeze(1)
            for domain_id in self.source_domain_ids
        ]

        return torch.stack(
            logits,
            dim=0,
        ).mean(dim=0)


def build_baseline():
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(
        in_features,
        2,
    )
    return model


def checkpoint_candidates(
    root: Path,
    model_type: str,
    target: str,
):
    if model_type == "baseline":
        relative = (
            Path("checkpoints")
            / "baseline"
            / f"target_{target}"
            / "best_resnet18.pt"
        )
    elif model_type == "safas":
        relative = (
            Path("checkpoints")
            / "safas"
            / f"target_{target}"
            / "best_safas_aligned.pt"
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    # Normal project location first.
    # Pipeline/checkpoints is kept as a fallback for older runs.
    return [
        root / relative,
        root / "Pipeline" / relative,
    ]


def resolve_checkpoint(
    root: Path,
    model_type: str,
    target: str,
) -> Path:
    candidates = checkpoint_candidates(
        root,
        model_type,
        target,
    )

    for path in candidates:
        if path.exists():
            return path

    attempted = "\n".join(
        str(path) for path in candidates
    )

    raise FileNotFoundError(
        f"Checkpoint {model_type}/{target} não encontrado.\n"
        f"Locais verificados:\n{attempted}"
    )


def resolve_evaluation_json(
    root: Path,
    model_type: str,
    target: str,
) -> Path:
    path = (
        root
        / "results"
        / model_type
        / f"target_{target}"
        / "final_evaluation.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Resultado final não encontrado: "
            f"{path}"
        )

    return path


def read_evaluation_json(
    root: Path,
    model_type: str,
    target: str,
):
    path = resolve_evaluation_json(
        root,
        model_type,
        target,
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    threshold = float(
        data["threshold_protocol"]["threshold"]
    )

    return data, threshold, path


@st.cache_resource(show_spinner=False)
def load_baseline_model(
    checkpoint_path_str: str,
    device_name: str,
):
    checkpoint_path = Path(
        checkpoint_path_str
    )

    device = torch.device(
        device_name
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = build_baseline().to(
        device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


@st.cache_resource(show_spinner=False)
def load_safas_model(
    checkpoint_path_str: str,
    device_name: str,
):
    checkpoint_path = Path(
        checkpoint_path_str
    )

    device = torch.device(
        device_name
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    source_domain_ids = checkpoint.get(
        "source_domain_ids"
    )

    if not source_domain_ids:
        raise RuntimeError(
            "O checkpoint SA-FAS não contém source_domain_ids."
        )

    model = SAFASResNet18(
        source_domain_ids
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


@torch.inference_mode()
def predict_baseline(
    model,
    image_tensor,
    device,
):
    x = image_tensor.unsqueeze(0).to(
        device
    )

    use_amp = device.type == "cuda"

    if use_amp:
        with torch.amp.autocast(
            device_type="cuda",
            enabled=True,
        ):
            logits = model(x)
    else:
        logits = model(x)

    probability = torch.softmax(
        logits.float(),
        dim=1,
    )[0, 1]

    return float(
        probability.cpu().item()
    )


@torch.inference_mode()
def predict_safas(
    model,
    image_tensor,
    device,
):
    x = image_tensor.unsqueeze(0).to(
        device
    )

    use_amp = device.type == "cuda"

    if use_amp:
        with torch.amp.autocast(
            device_type="cuda",
            enabled=True,
        ):
            features = model.encode(x)
            logits = model.mean_logits(
                features
            )
    else:
        features = model.encode(x)
        logits = model.mean_logits(
            features
        )

    probability = torch.sigmoid(
        logits.float()
    )[0]

    return float(
        probability.cpu().item()
    )


def result_label(
    spoof_score: float,
    threshold: float,
):
    if spoof_score >= threshold:
        return "SPOOF", True
    return "BONA FIDE", False


def render_prediction(
    title: str,
    spoof_score: float,
    threshold: float,
    checkpoint_epoch,
    source_val_auc,
):
    label, is_spoof = result_label(
        spoof_score,
        threshold,
    )

    st.subheader(title)

    if is_spoof:
        st.error(
            f"Prediction: {label}"
        )
    else:
        st.success(
            f"Prediction: {label}"
        )

    st.metric(
        "Spoof score",
        f"{spoof_score * 100:.2f}%",
    )

    st.write(
        f"**Decision threshold:** "
        f"{threshold:.6f}"
    )

    margin = spoof_score - threshold

    st.write(
        f"**Distance from threshold:** "
        f"{margin:+.6f}"
    )

    if checkpoint_epoch is not None:
        st.caption(
            f"Checkpoint epoch: {checkpoint_epoch}"
        )

    if source_val_auc is not None:
        st.caption(
            "Checkpoint/source validation AUC: "
            f"{source_val_auc:.4f}"
        )


def metadata_from_result(
    result_json: dict,
):
    epoch = result_json.get(
        "checkpoint_epoch"
    )

    source_auc = result_json.get(
        "checkpoint_best_macro_source_val_auc"
    )

    try:
        source_auc = float(
            source_auc
        )
    except (
        TypeError,
        ValueError,
    ):
        source_auc = None

    return epoch, source_auc


def main():
    st.set_page_config(
        page_title="Face Anti-Spoofing Prototype",
        page_icon="🔐",
        layout="wide",
    )

    st.title(
        "Face Anti-Spoofing Prototype"
    )

    st.write(
        "Image-based proof of concept for distinguishing "
        "**bona fide** facial presentations from **spoof attacks**."
    )

    st.caption(
        "Experimental academic prototype. "
        "The output is not a production biometric-security decision."
    )

    try:
        root = find_project_root()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    with st.sidebar:
        st.header(
            "Experiment"
        )

        profile_label = st.selectbox(
            "Leave-one-dataset-out profile",
            options=list(
                PROFILES.keys()
            ),
            index=0,
        )

        mode = st.radio(
            "Model",
            options=[
                "Compare both",
                "ResNet18 baseline",
                "SA-FAS",
            ],
            index=0,
        )

        profile = PROFILES[
            profile_label
        ]

        st.write(
            f"**Training sources:** "
            f"{profile['sources']}"
        )
        st.write(
            f"**Unseen target:** "
            f"{profile['target_name']}"
        )
        st.write(
            f"**Device:** {device}"
        )

        st.divider()

        st.caption(
            "The threshold used by each model is loaded "
            "from its source-validation EER evaluation."
        )

    uploaded_file = st.file_uploader(
        "Select a facial image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp",
        ],
    )

    if uploaded_file is None:
        st.info(
            "Upload an image to start the analysis."
        )
        st.stop()

    try:
        image = Image.open(
            uploaded_file
        ).convert(
            "RGB"
        )
    except Exception as exc:
        st.error(
            f"Não foi possível abrir a imagem: {exc}"
        )
        st.stop()

    image_col, info_col = st.columns(
        [1, 1]
    )

    with image_col:
        st.image(
            image,
            caption="Selected image",
            use_container_width=True,
            width=500
        )

    with info_col:
        st.subheader(
            "Selected experimental profile"
        )
        st.write(
            f"**Sources:** {profile['sources']}"
        )
        st.write(
            f"**Target held out during training:** "
            f"{profile['target_name']}"
        )
        st.write(
            "**Input preprocessing:** "
            "RGB → 256×256 → ImageNet normalization"
        )
        st.write(
            "**Positive class:** spoof"
        )

    analyze = st.button(
        "Analyse image",
        type="primary",
        use_container_width=True,
    )

    if not analyze:
        st.stop()

    preprocess = build_preprocess()
    image_tensor = preprocess(
        image
    )

    target = profile[
        "target"
    ]

    run_baseline = mode in (
        "Compare both",
        "ResNet18 baseline",
    )

    run_safas = mode in (
        "Compare both",
        "SA-FAS",
    )

    baseline_output = None
    safas_output = None

    with st.spinner(
        "Running model inference..."
    ):
        if run_baseline:
            try:
                baseline_checkpoint = (
                    resolve_checkpoint(
                        root,
                        "baseline",
                        target,
                    )
                )

                (
                    baseline_result,
                    baseline_threshold,
                    _,
                ) = read_evaluation_json(
                    root,
                    "baseline",
                    target,
                )

                baseline_model, _ = (
                    load_baseline_model(
                        str(
                            baseline_checkpoint
                        ),
                        str(device),
                    )
                )

                baseline_score = (
                    predict_baseline(
                        baseline_model,
                        image_tensor,
                        device,
                    )
                )

                (
                    baseline_epoch,
                    baseline_source_auc,
                ) = metadata_from_result(
                    baseline_result
                )

                baseline_output = {
                    "score": baseline_score,
                    "threshold": baseline_threshold,
                    "epoch": baseline_epoch,
                    "source_auc": baseline_source_auc,
                }

            except Exception as exc:
                baseline_output = {
                    "error": str(exc)
                }

        if run_safas:
            try:
                safas_checkpoint = (
                    resolve_checkpoint(
                        root,
                        "safas",
                        target,
                    )
                )

                (
                    safas_result,
                    safas_threshold,
                    _,
                ) = read_evaluation_json(
                    root,
                    "safas",
                    target,
                )

                safas_model, _ = (
                    load_safas_model(
                        str(
                            safas_checkpoint
                        ),
                        str(device),
                    )
                )

                safas_score = (
                    predict_safas(
                        safas_model,
                        image_tensor,
                        device,
                    )
                )

                (
                    safas_epoch,
                    safas_source_auc,
                ) = metadata_from_result(
                    safas_result
                )

                safas_output = {
                    "score": safas_score,
                    "threshold": safas_threshold,
                    "epoch": safas_epoch,
                    "source_auc": safas_source_auc,
                }

            except Exception as exc:
                safas_output = {
                    "error": str(exc)
                }

    st.divider()
    st.header(
        "Prediction"
    )

    outputs = []

    if run_baseline:
        outputs.append(
            (
                "ResNet18 baseline",
                baseline_output,
            )
        )

    if run_safas:
        outputs.append(
            (
                "SA-FAS",
                safas_output,
            )
        )

    columns = st.columns(
        len(outputs)
    )

    for column, (
        title,
        output,
    ) in zip(
        columns,
        outputs,
    ):
        with column:
            if output is None:
                st.error(
                    "No output."
                )
                continue

            if "error" in output:
                st.error(
                    f"{title}: {output['error']}"
                )
                continue

            render_prediction(
                title=title,
                spoof_score=output[
                    "score"
                ],
                threshold=output[
                    "threshold"
                ],
                checkpoint_epoch=output[
                    "epoch"
                ],
                source_val_auc=output[
                    "source_auc"
                ],
            )

    if (
        run_baseline
        and run_safas
        and baseline_output
        and safas_output
        and "error" not in baseline_output
        and "error" not in safas_output
    ):
        baseline_label, _ = result_label(
            baseline_output["score"],
            baseline_output["threshold"],
        )

        safas_label, _ = result_label(
            safas_output["score"],
            safas_output["threshold"],
        )

        st.divider()

        if baseline_label == safas_label:
            st.info(
                "Both models produced the same binary decision: "
                f"**{baseline_label}**."
            )
        else:
            st.warning(
                "The two models disagree on this image. "
                f"ResNet18: **{baseline_label}** | "
                f"SA-FAS: **{safas_label}**."
            )


if __name__ == "__main__":
    main()
