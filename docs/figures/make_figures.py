"""Draws the figures of the README from the numbers in the reports of this repository.

The tables behind the figures were produced by scripts that read data which is not in the repository, and their CSV files
are not committed, so the numbers are typed in below, each with the file and section it comes from. Nothing is recomputed.
Fixed palette (Okabe-Ito, colour-blind safe), dpi 150; the sample size n is on every plot.

usage: python docs/figures/make_figures.py          # writes PNG files to docs/media/
Needs matplotlib and numpy.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "media"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, RED, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7F7F7F", "#56B4E9"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.dpi": 150, "axes.titlesize": 11, "axes.titleweight": "bold"})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def bars_with_labels(ax, x, heights, colours, errors=None, fmt="{:.2f}"):
    ax.bar(x, heights, color=colours, width=0.62, yerr=errors, capsize=4, error_kw={"lw": 1.2, "ecolor": "#222222"})
    for xi, h in zip(x, heights):
        ax.text(xi, 0.012, fmt.format(h), ha="center", va="bottom", color="white", fontweight="bold", fontsize=10)


# ---------------------------------------------------------------- detector: what the leak-free split showed
# experiments/detector/README.md, "Result": deployed model AP50 0.74 on images from its old training split and 0.41 on
# images it had not seen; on the 117 leak-free test images AP50 0.72 (new) against 0.42 (deployed), 95 % bootstrap
# interval of the difference +0.23 to +0.37. Full test split (165 images, 292 fish): 0.66 against 0.55.
def detector_figure(path_name="detector-leak-free-ap50.png", with_title=True):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    bars_with_labels(a, [0, 1], [0.74, 0.41], [GREY, GREY])
    a.set_xticks([0, 1], ["its old training\nsplit images", "images it had\nnever seen"])
    a.set_ylim(0, 1)
    a.set_ylabel("AP50")
    a.set_title("Deployed detector: the old split flattered it" if with_title else "")
    a.text(0.5, 0.93, "n of each subset not recorded", ha="center", fontsize=8, color="#444444", transform=a.transAxes)
    x = np.array([0, 1, 3, 4])
    bars_with_labels(b, x, [0.55, 0.66, 0.42, 0.72], [GREY, BLUE, GREY, BLUE])
    b.set_xticks(x, ["deployed", "new", "deployed", "new"])
    b.set_ylim(0, 1)
    b.set_title("Leak-free test split" if with_title else "")
    b.text(0.5, -0.3, "all 165 test images\n(292 fish)", ha="center", transform=b.get_xaxis_transform())
    b.text(3.5, -0.3, "the 117 test images the deployed\nmodel had never seen", ha="center", transform=b.get_xaxis_transform())
    b.text(3.5, 0.84, "difference +0.23 to +0.37\n(95 % bootstrap interval)", ha="center", fontsize=8, color="#222222")
    fig.tight_layout()
    save(fig, path_name)


# ---------------------------------------------------------------- classifier: accuracy by approach
# experiments/results/report.md, part A (matched split, 7 classes, "all eval", n = 45 unique crops, 95 % intervals) and
# part B (stratified group 4-fold CV x 5 seeds, 7 classes, n = 138, sd over seeds); the shipped model's number is from
# experiments/classifier/README.md (4 folds x 8 seeds: 0.572 +- 0.036); SAM from classifier/SAM_EXPERIMENT_RESULT.md.
def classifier_figure(name="classifier-accuracy-by-approach.png"):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    labels_a = ["YOLO classifier\n(deployed;\nserver preproc.)", "YOLO classifier\n(deployed;\nnative preproc.)", "DINOv2-224\n+ LR", "DINOv2-448\n+ LR", "BioCLIP\n+ LR"]
    acc_a = np.array([0.29, 0.38, 0.51, 0.53, 0.40])
    lo = np.array([0.18, 0.25, 0.37, 0.39, 0.27])
    hi = np.array([0.43, 0.52, 0.65, 0.67, 0.55])
    x = np.arange(len(acc_a))
    a.bar(x, acc_a, color=[GREY, GREY, BLUE, BLUE, SKY], width=0.62, yerr=[acc_a - lo, hi - acc_a], capsize=4)
    for xi, v in zip(x, acc_a):
        a.text(xi, 0.012, f"{v:.2f}", ha="center", va="bottom", color="white", fontweight="bold")
    a.set_xticks(x, labels_a, fontsize=8)
    a.set_ylim(0, 0.8)
    a.set_ylabel("accuracy, 7 classes")
    a.set_title("Matched split, n = 45 crops (95 % interval)")
    a.text(0.5, 0.95, "the intervals of the YOLO baseline and DINOv2 overlap", ha="center", fontsize=8, transform=a.transAxes, color="#444444")
    labels_b = ["DINOv2-224\n(shipped)", "DINOv2-448", "SAM-masked\ncrops", "BioCLIP"]
    acc_b = np.array([0.572, 0.58, 0.450, 0.42])
    sd_b = np.array([0.036, 0.03, 0.028, 0.02])
    xb = np.arange(len(acc_b))
    b.bar(xb, acc_b, color=[BLUE, BLUE, ORANGE, SKY], width=0.62, yerr=sd_b, capsize=4)
    for xi, v in zip(xb, acc_b):
        b.text(xi, 0.012, f"{v:.2f}", ha="center", va="bottom", color="white", fontweight="bold")
    b.set_xticks(xb, labels_b, fontsize=8)
    b.set_ylim(0, 0.8)
    b.set_title("Grouped CV, n = 138 crops (± sd over seeds)")
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- the confidence gate
# experiments/classifier/README.md, "Confidence gate": same out-of-fold predictions that the threshold was chosen on.
def gate_figure(name="confidence-gate-tradeoff.png"):
    thr = np.array([0.00, 0.45, 0.65, 0.83])
    kept = np.array([100, 92, 70, 49])
    acc = np.array([0.57, 0.60, 0.67, 0.75])
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.plot(kept, acc, "-o", color=BLUE, lw=2)
    for k, a_, t in zip(kept, acc, thr):
        ax.annotate(f"threshold {t:.2f}", (k, a_), textcoords="offset points", xytext=(6, -14), fontsize=8)
    ax.axhline(0.75, color=RED, lw=1, ls="--")
    ax.text(100, 0.755, "target 0.75", color=RED, ha="right", fontsize=8)
    ax.set_xlabel("share of crops kept (not flagged uncertain), %")
    ax.set_ylabel("accuracy among the kept crops")
    ax.set_xlim(105, 40)
    ax.set_ylim(0.5, 0.82)
    ax.set_title("Confidence gate, out-of-fold, n = 138 crops")
    ax.text(0.02, 0.04, "the threshold was selected on these same predictions,\nso 0.75 is a target, not an independent measurement",
            transform=ax.transAxes, fontsize=8, color="#444444")
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- confusion matrix
# experiments/results/report.md, "Confusion matrix (seed 0, out-of-fold): dinov2_base_448, 7-class". This is the 448 embedder
# of the comparison, not the shipped 224 one; n = 138 crops.
def confusion_figure(name="confusion-7class-oof.png"):
    classes = ["dermatomycosis", "fin_rot", "healthy", "hexamitosis", "mycobacteriosis", "oodiniosis", "plistophorosis"]
    m = np.array([[4, 1, 3, 0, 1, 3, 1], [0, 14, 4, 0, 6, 0, 0], [2, 4, 29, 2, 3, 2, 0], [3, 1, 2, 11, 0, 0, 0],
                  [0, 6, 4, 0, 9, 1, 1], [1, 1, 1, 2, 2, 6, 2], [0, 0, 0, 0, 0, 0, 6]])
    assert m.sum() == 138
    rows = m / m.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", color="white" if rows[i, j] > 0.5 else "#222222", fontsize=9)
    ax.set_xticks(range(7), classes, rotation=40, ha="right")
    ax.set_yticks(range(7), [f"{c} (n={m[i].sum()})" for i, c in enumerate(classes)])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Out-of-fold confusion, DINOv2-448 + LR, seed 0\n(counts; colour = share of the true class)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- silver data
# experiments/results/report.md, "C2. Does adding silver data help?": group CV on gold (n = 138), silver added to the
# training folds only. Differences of 0.01 to 0.02 are within the seed noise of part B (sd 0.01 to 0.03).
def silver_figure(name="silver-data-effect.png"):
    emb = ["DINOv2-224", "DINOv2-448", "BioCLIP-224"]
    cfg = ["gold only", "+ healthy x100", "+ healthy x100\n+ fungal -> dermatomycosis x100"]
    acc = np.array([[0.57, 0.59, 0.57], [0.58, 0.59, 0.57], [0.42, 0.40, 0.41]])
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    w = 0.26
    for k, (c, col) in enumerate(zip(cfg, [GREY, BLUE, ORANGE])):
        ax.bar(np.arange(3) + (k - 1) * w, acc[:, k], w, label=c, color=col)
        for i, v in enumerate(acc[:, k]):
            ax.text(i + (k - 1) * w, v + 0.008, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), emb)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("accuracy, 7 classes")
    ax.set_title("Adding external silver data does not help (n = 138 gold crops)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- external images
# experiments/results/report.md, "C1": the gold-trained 7-class DINOv2-224 model applied to external images (about 250 per class).
def external_figure(name="external-probe-predicted-healthy.png"):
    classes = ["aeromoniasis", "bacterial\ngill disease", "bacterial\nred disease", "saprolegniasis\n(fungal)", "healthy fish", "parasitic\ndiseases", "white tail\ndisease"]
    share = np.array([0.43, 0.29, 0.42, 0.54, 0.54, 0.55, 0.42])
    n = [250, 249, 248, 250, 250, 249, 250]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    x = np.arange(len(classes))
    ax.bar(x, share, color=[RED] * 3 + [RED, GREEN, RED, RED], width=0.62)
    for xi, v, k in zip(x, share, n):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
        ax.text(xi, 0.012, f"n={k}", ha="center", fontsize=7, color="white")
    ax.set_xticks(x, classes, fontsize=8)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("share called `healthy`")
    ax.set_title("External pond / market fish: the model does not transfer (DINOv2-224)")
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- hero: what honest evaluation changed
def hero_figure(name="evaluation-changed-conclusions.png"):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.8, 4.1))
    # detector: the same deployed model on two kinds of images, and the replacement
    x = np.array([0, 1, 2.6, 3.6])
    bars_with_labels(a, x, [0.74, 0.41, 0.42, 0.72], [GREY, GREY, GREY, BLUE])
    a.set_xticks(x, ["old model,\nits own\ntraining images", "old model,\nunseen\nimages", "old model,\n117 unseen\ntest images", "new model,\nthe same 117"], fontsize=8)
    a.set_ylim(0, 1)
    a.set_ylabel("AP50")
    a.set_title("Detector: the old split flattered the old model")
    a.text(0.5, 0.94, "n of the first two subsets not recorded", ha="center", fontsize=8, color="#444444", transform=a.transAxes)
    # classifier: matched split with intervals
    labels = ["YOLO classifier\n(deployed)", "DINOv2-224\n+ linear head"]
    acc = np.array([0.38, 0.51])
    lo = np.array([0.25, 0.37])
    hi = np.array([0.52, 0.65])
    xb = np.arange(2)
    b.bar(xb, acc, color=[GREY, BLUE], width=0.5, yerr=[acc - lo, hi - acc], capsize=5)
    for xi, v in zip(xb, acc):
        b.text(xi, 0.012, f"{v:.2f}", ha="center", va="bottom", color="white", fontweight="bold")
    b.set_xticks(xb, labels)
    b.set_ylim(0, 0.8)
    b.set_ylabel("accuracy, 7 classes")
    b.set_title("Classifier, n = 45 crops, 95 % intervals overlap")
    fig.tight_layout()
    save(fig, name)


if __name__ == "__main__":
    hero_figure()
    detector_figure()
    classifier_figure()
    gate_figure()
    confusion_figure()
    silver_figure()
    external_figure()
