import argparse                                   # command-line options (--model, --runs, ...)
from pathlib import Path                          # paths joined with /, folder creation
import pandas as pd                               # reads the CSV logs into tables (DataFrames)
import matplotlib
matplotlib.use("Agg")                             # no screen on Kaggle: draw straight to a PNG file
import matplotlib.pyplot as plt                   # the figure
from scipy.stats import spearmanr                 # rank correlation and its p-value


SIGNAL = {"wgangp": ("w1_hat", "critic estimate of W1"),
          "dcgan": ("lossD", "discriminator loss")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(SIGNAL), default="wgangp")  # which run to analyse
    p.add_argument("--runs", default="runs")         # parent of runs/<model>/; /kaggle/working/runs on Kaggle
    p.add_argument("--window", type=int, default=5)  # rolling mean over 5 log rows = 500 generator steps
    p.add_argument("--out", default="figs")          # folder for the figure, linked from the README
    p.add_argument("--min_step", type=int, default=1)  # log rows and FIDs before this step are left out: at step 0 the critic is untrained
    args = p.parse_args()

    col, label = SIGNAL[args.model]                  # e.g. ("w1_hat", "critic estimate of W1")
    run = Path(args.runs) / args.model               # e.g. runs/wgangp
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)   # create figs/ if it does not exist

    # ---- 1. Load the two logs written during training ----
    log = pd.read_csv(run / "log.csv")               # one row every 100 generator steps: step, losses, signal
    log = log[log["step"] >= args.min_step]          # default 1: drops the step-0 signal, before the smoothing
    fids = pd.read_csv(run / "log_fid.csv")              # one row every --fid_every generator steps: step, fid
    fids = fids[fids["step"] >= args.min_step]       # default 1: drops the step-0 FID, measured before any training

    # ---- 2. Smooth the signal ----
    log["signal_s"] = log[col].rolling(args.window, min_periods=1).mean()

    # ---- 3. Align the two time series ----
    m = pd.merge_asof(fids.sort_values("step"),
                      log[["step", "signal_s"]].sort_values("step"),
                      on="step", direction="backward")
    m = m.dropna()                                   # an FID before the first log row has no partner

    # ---- 4. Spearman's rho ----
    rho, pval = spearmanr(m["signal_s"], m["fid"])
    # The p-value assumes independent points; checkpoints along one run are autocorrelated, so it is
    # optimistic. Report rho with n and read it qualitatively.
    print(f"{args.model}: Spearman rho = {rho:.2f} (p = {pval:.3g}, n = {len(m)} checkpoints)")

    # ---- 5. Figure: signal on the left axis, FID on the right, shared x = generator step ----
    fig, ax = plt.subplots(figsize=(6, 3.5))        # 6 x 3.5 inches: fits one README column
    # The trailing comma unpacks the one-element list plot() returns, to pass the line to the legend.
    l1, = ax.plot(log["step"], log["signal_s"], color="C0", label=f"{label} (smoothed)")
    ax.set_xlabel("generator step")
    ax.set_ylabel(label, color="C0")                 # axis label in the line's colour: says which is which
    ax2 = ax.twinx()                                 # second y-axis sharing x: the two have different units
    # "o-": a dot per checkpoint joined by lines, so the reader sees where FID was actually measured.
    l2, = ax2.plot(m["step"], m["fid"], "o-", color="C1", label="FID")
    ax2.set_ylabel("FID (lower is better)", color="C1")
    # Each axis has its own legend by default; pass both lines to one legend instead.
    ax.legend(handles=[l1, l2], loc="upper right", frameon=False)
    ax.set_title(f"{args.model}: Spearman rho = {rho:.2f} over {len(m)} checkpoints")
    fig.tight_layout()                               # shrink margins without cutting the axis labels
    fig.savefig(out / f"signal_vs_fid_{args.model}.png", dpi=200)   # 200 dpi: sharp in the README


# Runs main() only when the file is executed (python analysis.py), not when it is imported.
if __name__ == "__main__":
    main()