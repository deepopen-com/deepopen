"""Render the benchmark figure used in the GitHub and Hugging Face READMEs.

  USE_TF=0 python3 notebooks/make_plots.py

Reads local_benchmark_results.json (51-language sweep) and laya_benchmark_results.json
(T4 latency + English suites); writes assets/laya_benchmark.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, "assets")
os.makedirs(ASSETS, exist_ok=True)

# Validated categorical slots 1 and 2 (light mode) + ink tokens.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def strip(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def main():
    loc = json.load(open(os.path.join(REPO, "local_benchmark_results.json")))
    col = json.load(open(os.path.join(REPO, "laya_benchmark_results.json")))

    en = loc["part_a"]["by_model"]["english"]["per_language"]
    ml = loc["part_a"]["by_model"].get("multilingual", {}).get("per_language", {})
    langs = sorted(set(en) & set(ml), key=lambda l: (ml[l]["accuracy"] - en[l]["accuracy"]))
    if not langs:
        raise SystemExit("multilingual sweep not finished yet")

    fig = plt.figure(figsize=(16, max(9, 0.23 * len(langs) + 3.2)))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], height_ratios=[1, 1, 1],
                          wspace=0.28, hspace=0.55)

    # ---------------------------------------------------------------- A. dumbbell
    axA = fig.add_subplot(gs[:, 0])
    y = np.arange(len(langs))
    e = np.array([en[l]["accuracy"] for l in langs])
    m = np.array([ml[l]["accuracy"] for l in langs])
    axA.hlines(y, e, m, color=GRID, lw=2.4, zorder=1)
    axA.scatter(e, y, s=62, color=BLUE, zorder=3, label="laya (English checkpoint)",
                edgecolors=SURFACE, linewidths=1.6)
    axA.scatter(m, y, s=62, color=ORANGE, zorder=3, label="laya-multilingual",
                edgecolors=SURFACE, linewidths=1.6)
    axA.axvline(0.05, color=INK3, lw=1.4, ls=(0, (4, 3)), zorder=2)
    axA.text(0.052, len(langs) - 0.4, "random guess (0.050)", color=INK3, fontsize=9, va="top")
    axA.set_yticks(y); axA.set_yticklabels(langs, fontsize=9)
    axA.set_ylim(-0.8, len(langs) - 0.2)
    axA.set_xlim(-0.02, max(0.9, float(max(e.max(), m.max())) + 0.06))
    axA.set_xlabel("accuracy  ·  MASSIVE intent, 20 options", fontsize=10)
    axA.set_title("Every language: which checkpoint can read it",
                  fontsize=14, fontweight="bold", color=INK, loc="left", pad=12)
    axA.xaxis.grid(True, color=GRID, lw=0.9); axA.set_axisbelow(True)
    strip(axA)
    axA.legend(frameon=False, loc="lower right", fontsize=10, labelcolor=INK2)

    # ---------------------------------------------------------------- B. latency
    axB = fig.add_subplot(gs[0, 1])
    lat = col["latency"]
    ns = ["1_questions", "5_questions", "10_questions", "50_questions"]
    lab = ["1", "5", "10", "50"]
    a = [lat["laya"][n]["p50_ms"] for n in ns]
    b = [lat["laya-multilingual"][n]["p50_ms"] for n in ns]
    x = np.arange(len(ns)); w = 0.36
    axB.bar(x - w / 2 - 0.01, a, w, color=BLUE, label="laya", zorder=3)
    axB.bar(x + w / 2 + 0.01, b, w, color=ORANGE, label="laya-multilingual", zorder=3)
    for xi, v in zip(x - w / 2 - 0.01, a):
        axB.text(xi, v + 18, "%.0f" % v, ha="center", fontsize=9, color=INK2)
    for xi, v in zip(x + w / 2 + 0.01, b):
        axB.text(xi, v + 18, "%.0f" % v, ha="center", fontsize=9, color=INK2)
    axB.axhline(256, color=INK3, lw=1.4, ls=(0, (4, 3)), zorder=2)
    axB.text(3.42, 272, "Jev, measured\n236–276 ms", color=INK3, fontsize=8.5, ha="right")
    axB.set_xticks(x); axB.set_xticklabels(lab)
    axB.set_xlabel("questions per call", fontsize=10)
    axB.set_ylabel("p50 latency (ms)", fontsize=10)
    axB.set_ylim(0, max(a + b) * 1.18)
    axB.set_title("Speed on one T4", fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)
    axB.yaxis.grid(True, color=GRID, lw=0.9); axB.set_axisbelow(True)
    strip(axB); axB.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK2)

    # ---------------------------------------------------------------- C. vs Jev
    axC = fig.add_subplot(gs[1, 1])
    names = ["AG News\n(4 labels)", "DAIR Emotion\n(6 labels)", "typed-decisions\n(2,000 dec.)"]
    jev = [0.910, 0.480, 0.727]
    lay = [col["suites"]["en.ag_news"]["laya"]["calibrated"]["accuracy"],
           col["suites"]["en.emotion"]["laya"]["calibrated"]["accuracy"], 0.766]
    x = np.arange(3)
    axC.bar(x - w / 2 - 0.01, jev, w, color=INK3, label="Jev (published)", zorder=3)
    axC.bar(x + w / 2 + 0.01, lay, w, color=BLUE, label="Laya (measured)", zorder=3)
    for xi, v in zip(x - w / 2 - 0.01, jev):
        axC.text(xi, v + .02, "%.3f" % v, ha="center", fontsize=9, color=INK2)
    for xi, v in zip(x + w / 2 + 0.01, lay):
        axC.text(xi, v + .02, "%.3f" % v, ha="center", fontsize=9, color=INK2)
    axC.set_xticks(x); axC.set_xticklabels(names, fontsize=9)
    axC.set_ylim(0, 1.06); axC.set_ylabel("accuracy", fontsize=10)
    axC.set_title("Against Jev, same public datasets", fontsize=13, fontweight="bold",
                  color=INK, loc="left", pad=10)
    axC.yaxis.grid(True, color=GRID, lw=0.9); axC.set_axisbelow(True)
    strip(axC); axC.legend(frameon=False, fontsize=9.5, loc="lower left", labelcolor=INK2)
    axC.text(0, -0.34, "typed-decisions uses the fine-tuned checkpoint. Jev figures are "
             "third-party published, not measured here.",
             transform=axC.transAxes, fontsize=8, color=INK3)

    # ---------------------------------------------------------------- D. calibration
    axD = fig.add_subplot(gs[2, 1])
    rep = col["calibration_repair"]
    models = ["laya", "laya-multilingual"]
    ship = [rep[m]["mean_ece_shipped"] for m in models]
    refit = [rep[m]["mean_ece_refit"] for m in models]
    x = np.arange(2)
    axD.bar(x - w / 2 - 0.01, ship, w, color=ORANGE, label="as shipped", zorder=3)
    axD.bar(x + w / 2 + 0.01, refit, w, color=BLUE, label="temperature refit", zorder=3)
    for xi, v in zip(x - w / 2 - 0.01, ship):
        axD.text(xi, v + .012, "%.3f" % v, ha="center", fontsize=9, color=INK2)
    for xi, v in zip(x + w / 2 + 0.01, refit):
        axD.text(xi, v + .012, "%.3f" % v, ha="center", fontsize=9, color=INK2)
    axD.axhline(0.246, color=INK3, lw=1.4, ls=(0, (4, 3)), zorder=2)
    axD.text(1.45, 0.253, "Jev ECE 0.246", color=INK3, fontsize=8.5, ha="right")
    axD.set_xticks(x); axD.set_xticklabels(models, fontsize=9.5)
    axD.set_ylabel("mean ECE  (lower is better)", fontsize=10)
    axD.set_ylim(0, max(ship) * 1.25)
    axD.set_title("Calibration: fit your temperatures", fontsize=13, fontweight="bold",
                  color=INK, loc="left", pad=10)
    axD.yaxis.grid(True, color=GRID, lw=0.9); axD.set_axisbelow(True)
    strip(axD); axD.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK2)

    fig.suptitle("Laya benchmark  ·  three checkpoints, identical questions, one T4",
                 fontsize=16, fontweight="bold", color=INK, x=0.012, ha="left", y=0.985)
    out = os.path.join(ASSETS, "laya_benchmark.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out, "(%.0f KB)" % (os.path.getsize(out) / 1024))


if __name__ == "__main__":
    main()
