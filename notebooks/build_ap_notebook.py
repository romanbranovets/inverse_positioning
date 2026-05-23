"""One-off helper to generate ap.ipynb."""
import json
from pathlib import Path

NB_PATH = Path(__file__).with_name("ap.ipynb")

AUDIT_HELPERS = r'''
def bounded_logits(logits, clamp_abs=CONFIDENCE_LOGIT_ABS_CLAMP):
    return logits.clamp(-clamp_abs, clamp_abs)


def nwj_terms(positive_logits, negative_logits, clamp_abs=CONFIDENCE_LOGIT_ABS_CLAMP):
    positive_t = bounded_logits(positive_logits, clamp_abs)
    negative_t = bounded_logits(negative_logits, clamp_abs)
    negative_exp = torch.exp(negative_t - 1.0)
    return positive_t, negative_exp


@torch.no_grad()
def normal_quantile(probability):
    p = torch.tensor(float(probability), dtype=torch.float64)
    return float(torch.distributions.Normal(0.0, 1.0).icdf(p).item())


@torch.no_grad()
def nwj_summary_from_logits(positive_logits, negative_logits, clamp_abs):
    positive_t, negative_exp = nwj_terms(positive_logits, negative_logits, clamp_abs)
    positive_mean = positive_t.mean().double()
    negative_mean = negative_exp.mean().double()
    nwj_nats = positive_mean - negative_mean
    return dict(
        nwj_nats=float(nwj_nats.cpu()),
        positive_mean_nats=float(positive_mean.cpu()),
        negative_penalty_nats=float(negative_mean.cpu()),
        positive_samples=int(positive_t.numel()),
        negative_samples=int(negative_exp.numel()),
    )


@torch.no_grad()
def batch_mean_lcb(values, delta):
    values = torch.as_tensor(values, dtype=torch.float64)
    n = int(values.numel())
    mean = float(values.mean().item())
    if n < 2:
        return mean
    standard_error = float(values.std(unbiased=True).item() / math.sqrt(n))
    z = normal_quantile(1.0 - delta)
    inflation = math.sqrt(n / max(n - 2, 1)) if n > 2 else 2.0
    return mean - z * inflation * standard_error


@torch.no_grad()
def estimate_nwj_bound(
    model,
    step_count,
    batch_size,
    batches,
    negative_count,
    delta,
    chunk_size=EVAL_LOGIT_CHUNK_SIZE,
    clamp_abs=CONFIDENCE_LOGIT_ABS_CLAMP,
):
    was_training = model.training
    model.eval()
    if str(DEVICE).startswith("cuda"):
        torch.cuda.empty_cache()

    batch_nwj = []
    positive_weighted_sum = 0.0
    negative_weighted_sum = 0.0
    positive_samples = 0
    negative_samples = 0
    for _ in range(batches):
        x, pad_mask, endpoint, _ = make_trajectories(batch_size, step_counts=step_count)
        positive_logit, negative_logit = contrastive_log_probs(
            model,
            x,
            pad_mask,
            endpoint,
            negative_count,
            negative_chunk_size=chunk_size,
        )
        summary = nwj_summary_from_logits(positive_logit, negative_logit, clamp_abs)
        batch_nwj.append(summary["nwj_nats"])
        positive_weighted_sum += (
            summary["positive_mean_nats"] * summary["positive_samples"]
        )
        negative_weighted_sum += (
            summary["negative_penalty_nats"] * summary["negative_samples"]
        )
        positive_samples += summary["positive_samples"]
        negative_samples += summary["negative_samples"]

    if was_training:
        model.train()

    nwj_mean_nats = float(np.mean(batch_nwj))
    nwj_lcb_nats = batch_mean_lcb(batch_nwj, delta)
    positive_mean_nats = positive_weighted_sum / max(positive_samples, 1)
    negative_penalty_nats = negative_weighted_sum / max(negative_samples, 1)
    return dict(
        nwj_bits=nwj_mean_nats / math.log(2.0),
        nwj_lcb_bits=nwj_lcb_nats / math.log(2.0),
        positive_mean_bits=positive_mean_nats / math.log(2.0),
        negative_penalty_bits=negative_penalty_nats / math.log(2.0),
        positive_samples=int(positive_samples),
        negative_samples=int(negative_samples),
        audit_batches=int(batches),
        delta=float(delta),
        clamp_abs=float(clamp_abs),
    )


def create_ap_bundle():
    model = AmortizedPosterior(
        feature_dim=FEATURE_DIM,
        feature_mean=FEATURE_MEAN,
        feature_std=FEATURE_STD,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_components=N_COMPONENTS,
        dropout=DROPOUT,
        kappa_floor=KAPPA_FLOOR,
        kappa_init=KAPPA_INIT,
    ).to(device=DEVICE, dtype=DTYPE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAIN_EPOCHS * TRAIN_BATCHES_PER_EPOCH,
        eta_min=0.1 * LEARNING_RATE,
    )
    return model, optimizer, scheduler


_parameter_count_model = AmortizedPosterior(
    feature_dim=FEATURE_DIM,
    feature_mean=FEATURE_MEAN,
    feature_std=FEATURE_STD,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_encoder_layers=N_ENCODER_LAYERS,
    n_components=N_COMPONENTS,
    dropout=DROPOUT,
    kappa_floor=KAPPA_FLOOR,
    kappa_init=KAPPA_INIT,
).to(device=DEVICE, dtype=DTYPE)
sum(p.numel() for p in _parameter_count_model.parameters())
'''

TRAIN_LOOP = r'''
def train_ap_for_steps(step_count):
    model, optimizer, scheduler = create_ap_bundle()
    history = []
    best_val_nwj_bits = -float("inf")
    best_state_dict = None
    best_epoch = None
    stale_epochs = 0
    model.train()

    for epoch in range(1, TRAIN_EPOCHS + 1):
        epoch_objective = []
        epoch_nll = []
        epoch_nwj = []
        epoch_physics = []
        skipped_batches = 0
        pos_logits = []
        neg_logits = []

        for _ in range(TRAIN_BATCHES_PER_EPOCH):
            x, pad_mask, endpoint, _ = make_trajectories(
                TRAIN_BATCH_SIZE, step_counts=step_count
            )
            loss, metrics = ap_training_loss(
                model,
                x,
                pad_mask,
                endpoint,
                negative_count=TRAIN_NEGATIVES,
                clamp_abs=LOGIT_ABS_CLAMP,
                exp_clamp=TRAIN_NWJ_EXP_CLAMP,
                nll_weight=NLL_WEIGHT,
                nwj_weight=NWJ_WEIGHT,
                physics_weight=PHYSICS_WEIGHT,
                kappa_reg_weight=KAPPA_REG_WEIGHT,
                kappa_target=KAPPA_TARGET,
            )
            if not torch.isfinite(loss):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                pos_logits.append(metrics["positive_logit"])
                neg_logits.append(metrics["negative_logit"])

            epoch_objective.append(float(loss.detach().cpu()))
            epoch_nll.append(float(metrics["nll"].cpu()))
            epoch_nwj.append(float(metrics["nwj"].cpu()))
            epoch_physics.append(float(metrics["physics"].cpu()))

        if not epoch_nll:
            print(f"all batches skipped at epoch {epoch}; stopping this model")
            break

        pos_logits = torch.cat(pos_logits)
        neg_logits = torch.cat(neg_logits)
        train_nwj_bits = float(
            (
                nwj_training_nats(
                    pos_logits,
                    neg_logits,
                    LOGIT_ABS_CLAMP,
                    TRAIN_NWJ_EXP_CLAMP,
                )
                / math.log(2.0)
            ).cpu()
        )

        should_validate = (
            epoch == 1
            or epoch % VALIDATION_EVERY == 0
            or epoch == TRAIN_EPOCHS
            or epoch >= NWJ_EARLY_STOP_MIN_EPOCHS
        )
        validation = None
        if should_validate:
            validation = estimate_nwj_bound(
                model,
                step_count,
                VALIDATION_BATCH_SIZE,
                VALIDATION_BATCHES,
                VALIDATION_NEGATIVES_PER_POSITIVE,
                0.5,
                clamp_abs=LOGIT_ABS_CLAMP,
            )
            model.train()

        row = dict(
            epoch=epoch,
            objective=float(np.mean(epoch_objective)),
            nll_bits=float(np.mean(epoch_nll) / math.log(2.0)),
            train_nwj_bits=train_nwj_bits,
            val_nwj_bits=np.nan if validation is None else validation["nwj_bits"],
            physics=float(np.mean(epoch_physics)),
            skipped_batches=skipped_batches,
        )
        history.append(row)

        if validation is not None and math.isfinite(row["val_nwj_bits"]):
            if row["val_nwj_bits"] > best_val_nwj_bits + NWJ_EARLY_STOP_MIN_DELTA_BITS:
                best_val_nwj_bits = row["val_nwj_bits"]
                best_epoch = epoch
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            elif epoch >= NWJ_EARLY_STOP_MIN_EPOCHS:
                stale_epochs += 1

        should_stop = (
            epoch >= NWJ_EARLY_STOP_MIN_EPOCHS
            and stale_epochs >= NWJ_EARLY_STOP_PATIENCE
            and best_state_dict is not None
        )

        if (
            epoch == 1
            or epoch % PLOT_EVERY == 0
            or epoch == TRAIN_EPOCHS
            or should_stop
        ):
            clear_output(wait=True)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[item["epoch"] for item in history],
                    y=[item["nll_bits"] for item in history],
                    mode="lines",
                    name=f"steps={step_count} train NLL",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[item["epoch"] for item in history],
                    y=[item["train_nwj_bits"] for item in history],
                    mode="lines",
                    name="train NWJ",
                    yaxis="y2",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[
                        item["epoch"]
                        for item in history
                        if math.isfinite(item["val_nwj_bits"])
                    ],
                    y=[
                        item["val_nwj_bits"]
                        for item in history
                        if math.isfinite(item["val_nwj_bits"])
                    ],
                    mode="lines+markers",
                    name="validation NWJ",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                title=(
                    f"Training amortized posterior, steps={step_count}, "
                    f"epoch {epoch}/{TRAIN_EPOCHS}"
                ),
                xaxis_title="epoch",
                yaxis_title="NLL bits",
                yaxis2=dict(title="NWJ bits", overlaying="y", side="right"),
                height=360,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            display(fig)

        if should_stop:
            print(
                f"early stop at epoch {epoch}: train_NLL={row['nll_bits']:.4f} bits, "
                f"val_NWJ={row['val_nwj_bits']:.4f} bits, "
                f"best_val_NWJ={best_val_nwj_bits:.4f} bits at epoch {best_epoch}, "
                f"skipped={row['skipped_batches']}"
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return {
        "model": model,
        "history": history,
        "step_count": step_count,
        "best_epoch": best_epoch,
        "best_val_nwj_bits": best_val_nwj_bits,
    }


trained_models = {}
for steps in EVAL_STEP_COUNTS:
    trained_models[int(steps)] = train_ap_for_steps(int(steps))

for steps, result in trained_models.items():
    last = result["history"][-1]
    print(
        f"steps={steps:2d}: train_NLL={last['nll_bits']:.4f} bits, "
        f"train_NWJ={last['train_nwj_bits']:.4f} bits, "
        f"best_val_NWJ={result['best_val_nwj_bits']:.4f} bits at epoch {result['best_epoch']}, "
        f"skipped={last['skipped_batches']}"
    )
'''

AUDIT_CELL = r'''
rows = []
row_delta = EVAL_TOTAL_CONFIDENCE_DELTA / (
    len(trained_models) * len(EVAL_STEP_COUNTS)
)
print(
    f"Per-row failure probability <= {row_delta:.3g}; "
    f"P(any plotted audit NWJ LCB is above its batch-mean target) <= {EVAL_TOTAL_CONFIDENCE_DELTA:.3g}"
)
print(
    f"Audit critic clamp = +/-{CONFIDENCE_LOGIT_ABS_CLAMP:.1f} nats; target={TARGET_AUDIT_BITS:.1f} bits"
)

for trained_steps, result in trained_models.items():
    model = result["model"]
    for eval_steps in EVAL_STEP_COUNTS:
        metrics = estimate_nwj_bound(
            model,
            int(eval_steps),
            EVAL_BATCH_SIZE,
            EVAL_BATCHES_PER_POINT,
            EVAL_NEGATIVES_PER_POSITIVE,
            row_delta,
            clamp_abs=CONFIDENCE_LOGIT_ABS_CLAMP,
        )
        rows.append(
            dict(
                trained_steps=int(trained_steps),
                eval_steps=int(eval_steps),
                distance_m=int(eval_steps) * STEP_METERS,
                matched=int(trained_steps) == int(eval_steps),
                **metrics,
            )
        )

for row in rows:
    print(
        f"trained={row['trained_steps']:2d}, eval={row['eval_steps']:2d}, "
        f"distance={row['distance_m']:6.1f} m, "
        f"audit_NWJ_LCB={row['nwj_lcb_bits']:.4f} bits, "
        f"NWJ={row['nwj_bits']:.4f} bits, "
        f"pos={row['positive_mean_bits']:.4f} bits, "
        f"neg_penalty={row['negative_penalty_bits']:.4f} bits"
    )
'''

PLOT_CELL = r'''
fig = go.Figure()
for trained_steps in sorted({row["trained_steps"] for row in rows}):
    curve = sorted(
        [row for row in rows if row["trained_steps"] == trained_steps],
        key=lambda row: row["eval_steps"],
    )
    fig.add_trace(
        go.Scatter(
            x=[row["eval_steps"] for row in curve],
            y=[row["nwj_lcb_bits"] for row in curve],
            mode="lines+markers",
            name=f"trained steps={trained_steps}",
            customdata=[
                [row["distance_m"], row["matched"], row["nwj_bits"], row["delta"]]
                for row in curve
            ],
            hovertemplate=(
                "trained steps="
                + str(trained_steps)
                + "<br>eval steps=%{x}<br>distance=%{customdata[0]:.0f} m"
                + "<br>matched=%{customdata[1]}"
                + "<br>audit NWJ LCB=%{y:.4f} bits"
                + "<br>NWJ=%{customdata[2]:.4f} bits"
                + "<br>row failure prob<=%{customdata[3]:.3g}<extra></extra>"
            ),
        )
    )

matched_rows = [row for row in rows if row["matched"]]
if matched_rows:
    fig.add_trace(
        go.Scatter(
            x=[row["eval_steps"] for row in matched_rows],
            y=[row["nwj_lcb_bits"] for row in matched_rows],
            mode="markers",
            name="matched diagonal",
            marker=dict(size=10, symbol="diamond-open"),
        )
    )

fig.update_layout(
    title="30-bit audit NWJ lower confidence bound by trained model and evaluation step count",
    xaxis=dict(title="evaluation step count", dtick=1),
    yaxis=dict(title="audit NWJ lower confidence bound, bits"),
    height=460,
    margin=dict(l=40, r=20, t=60, b=40),
)
fig.show()
'''

BENCHMARK_CELL = r'''
@torch.no_grad()
def benchmark_audit_scoring(model, step_count, batch_size=128, negative_count=128, repeats=3):
    model.eval()
    x, pad_mask, endpoint, _ = make_trajectories(batch_size, step_counts=step_count)
    if str(DEVICE).startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        contrastive_log_probs(
            model,
            x,
            pad_mask,
            endpoint,
            negative_count,
            negative_chunk_size=EVAL_LOGIT_CHUNK_SIZE,
        )
        if str(DEVICE).startswith("cuda"):
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    queries = batch_size * (1 + negative_count)
    return dict(
        step_count=step_count,
        batch_size=batch_size,
        negative_count=negative_count,
        seconds=elapsed,
        queries_per_second=queries / elapsed,
    )

print("Amortized posterior audit scoring throughput (encode once + cheap vMF queries):")
for steps in [1, 8, 16]:
    stats = benchmark_audit_scoring(trained_models[steps]["model"], steps)
    print(
        f"steps={steps:2d}: {stats['seconds']:.4f} s / batch, "
        f"{stats['queries_per_second']:.0f} endpoint scores/s"
    )
'''


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md(
        "# Amortized Posterior (AP) MI Estimator\n\n"
        "This notebook trains an **amortized conditional density** on the sphere: one trajectory encode, then cheap vMF-mixture scoring for any endpoint. Training combines NLL, NWJ, and a physics consistency term on the predicted mean direction.\n\n"
        "Evaluation reuses the same NWJ audit protocol and plots as `discriminator.ipynb`, so rows are directly comparable across methods."
    ),
    md(
        "## 0. Kaggle / remote setup\n\n"
        "On Kaggle (or any clean runtime), this cell clones the project from GitHub and adds `notebooks/` to `sys.path`. Locally, it reuses the checkout next to this notebook.\n\n"
        "Optional environment variables:\n"
        "- `INVERSE_POSITIONING_REPO` (default: `https://github.com/romanbranovets/inverse_positioning.git`)\n"
        "- `INVERSE_POSITIONING_REF` (default: `main`)"
    ),
    code(
        """import os
import subprocess
import sys
from pathlib import Path

GITHUB_REPO = os.environ.get(
    "INVERSE_POSITIONING_REPO",
    "https://github.com/romanbranovets/inverse_positioning.git",
)
GITHUB_REF = os.environ.get("INVERSE_POSITIONING_REF", "main")

NOTEBOOK_ROOT = Path.cwd()
LOCAL_CANDIDATES = [
    NOTEBOOK_ROOT,
    NOTEBOOK_ROOT / "notebooks",
    NOTEBOOK_ROOT.parent,
    NOTEBOOK_ROOT.parent / "notebooks",
]


def find_shared_root():
    for root in LOCAL_CANDIDATES:
        if (root / "shared" / "density_common.py").exists():
            return root
        if (root / "shared" / "trajectory_sampler.py").exists():
            return root
    return None


shared_root = find_shared_root()
if shared_root is None:
    clone_dir = NOTEBOOK_ROOT / "_inverse_positioning"
    if not clone_dir.exists():
        print(f"Cloning {GITHUB_REPO} @ {GITHUB_REF} ...")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                GITHUB_REF,
                GITHUB_REPO,
                str(clone_dir),
            ],
            check=True,
        )
    shared_root = clone_dir / "notebooks"
    if not (shared_root / "shared" / "density_common.py").exists():
        raise FileNotFoundError(
            "Cloned repo does not contain notebooks/shared/density_common.py. "
            "Push the latest shared modules and set INVERSE_POSITIONING_REF."
        )

if str(shared_root) not in sys.path:
    sys.path.insert(0, str(shared_root))

try:
    import plotly  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "plotly"], check=True)

print(f"shared_root={shared_root}")"""
    ),
    md(
        "## 1. Setup\n\n"
        "Training samples, validation samples, and audit samples are generated independently on the fly from the shared synthetic field and trajectory generator."
    ),
    code(
        """import math
import random
import time

import numpy as np
import plotly.graph_objects as go
import torch
from IPython.display import clear_output, display

from shared.amortized_posterior import (
    AmortizedPosterior,
    ap_training_loss,
    contrastive_log_probs,
    nwj_training_nats,
)
from shared.sphere_utils import DEVICE
from shared.magentic_field import *
from shared.trajectory_sampler import *

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"device={DEVICE}, step={STEP_METERS:.1f} m, angular step={STEP_RAD:.3e} rad")

FEATURE_MEAN, FEATURE_STD = estimate_feature_normalization(device=DEVICE)

D_MODEL = 384
N_HEADS = 4
N_ENCODER_LAYERS = 4
N_COMPONENTS = 16
DROPOUT = 0.03
KAPPA_FLOOR = 20.0
KAPPA_INIT = 50.0
KAPPA_TARGET = 30.0

NLL_WEIGHT = 1.0
NWJ_WEIGHT = 0.5
PHYSICS_WEIGHT = 0.1
KAPPA_REG_WEIGHT = 0.01
TRAIN_NEGATIVES = 32

TRAIN_EPOCHS = 120
TRAIN_BATCHES_PER_EPOCH = 8
TRAIN_BATCH_SIZE = 512
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
PLOT_EVERY = 4
TRAIN_NWJ_EXP_CLAMP = 8.0
LOGIT_ABS_CLAMP = 22.0
CONFIDENCE_LOGIT_ABS_CLAMP = 22.0
TARGET_AUDIT_BITS = 30.0

VALIDATION_EVERY = 4
VALIDATION_BATCH_SIZE = 512
VALIDATION_BATCHES = 2
VALIDATION_NEGATIVES_PER_POSITIVE = 32
NWJ_EARLY_STOP_MIN_EPOCHS = 32
NWJ_EARLY_STOP_PATIENCE = 12
NWJ_EARLY_STOP_MIN_DELTA_BITS = 0.03

EVAL_STEP_COUNTS = [1, 2, 4, 8, 12, 16]
EVAL_BATCH_SIZE = 512
EVAL_BATCHES_PER_POINT = 32
EVAL_NEGATIVES_PER_POSITIVE = 128
EVAL_LOGIT_CHUNK_SIZE = 4096
EVAL_TOTAL_CONFIDENCE_DELTA = 0.01"""
    ),
    md(
        "## 2. Amortized Posterior Model\n\n"
        "The critic score for audit is `log p_theta(endpoint | trajectory)` from a vMF mixture whose parameters are predicted in a **single** trajectory encode. Negative scoring reuses cached mixture parameters and only evaluates closed-form vMF log densities."
    ),
    code(AUDIT_HELPERS),
    md(
        "## 3. Training\n\n"
        "Each fixed step count gets its own model. The objective is `NLL + NWJ + physics + kappa regularization`. Early stopping uses strict validation NWJ, matching the audit estimator."
    ),
    code(TRAIN_LOOP),
    md(
        "## 4. 30-Bit Audit Evaluation\n\n"
        "Each trained model is evaluated with the **same NWJ audit protocol and output format** as `discriminator.ipynb`. The critic score is `log p_theta(endpoint | trajectory)`.\n\n"
        "Each row reports a one-sided lower confidence bound over independent batch-level NWJ estimates. `EVAL_TOTAL_CONFIDENCE_DELTA` is split across every plotted row."
    ),
    code(AUDIT_CELL),
    code(PLOT_CELL),
    md(
        "## 5. Audit scoring throughput\n\n"
        "Amortized posterior encodes each trajectory once and evaluates negatives with cheap vMF queries only."
    ),
    code(BENCHMARK_CELL),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NB_PATH}")
