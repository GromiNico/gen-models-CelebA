#run from the repo root: python -m watermark.plot_frontier (reads results/fid_sweep.csv, writes figs/watermark_frontier.png)
#quality-robustness frontier of the fine-tuned generators: one curve per extractor, one point per lambda
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# extractors in a fixed order and color, the base run (old extractor, not trained enough) is left out
RUNS = [('A', 'extractor trained with α = 0.05 (A)', '#2a78d6', 'o'),
        ('B', 'extractor trained with α = 0.1 (B)', '#eb6834', '^')]
# column of fid_sweep.csv, panel title, y label
PANELS = [('tpr_none', 'detected at FPR 1e-6, no attack', 'fraction of images'),
          ('tpr_crop0.5', 'detected at FPR 1e-6, after a 50% crop', 'fraction of images'),
          ('acc_jpeg80', 'bit accuracy after JPEG 80', 'bit accuracy (32 bits)'),
          ('fid', 'FID against 10k real images (lower is better)', 'FID')]
# hand-placed labels (offset in points, alignment) where the curves and the noise line meet in the FID panel
NUDGE = {('fid', 'B', 10): ((-6, 5), 'right'), ('fid', 'A', 10): ((6, 22), 'left')}


def main():
    #arguments
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default = 'results/fid_sweep.csv')
    p.add_argument('--out', default = 'figs/watermark_frontier.png')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    models = df[df.lam.notna() & df.run.isin([r[0] for r in RUNS])]
    g0 = df.loc[df.run == 'G0', 'fid'].iloc[0]
    noise = df[df.run == 'G0+noise'].sort_values('psnr')

    fig, axes = plt.subplots(2, 2, figsize = (10, 7.5), sharex = True)
    for ax, (col, title, ylabel) in zip(axes.flat, PANELS):
        for run, name, c, mk in RUNS:
            r = models[models.run == run].sort_values('psnr')
            ax.plot(r.psnr, r[col], color = c, lw = 1.8, marker = mk, ms = 6, label = name)
            # lambda of each point, written on the side away from the other curves at the same PSNR:
            # above right, below left, or on the right when the curves almost touch
            others = models[models.run != run].sort_values('psnr')
            tol = 0.015*(models[col].max() - models[col].min())
            for x, y, lam in zip(r.psnr, r[col], r.lam):
                gap = y - np.interp(x, others.psnr, others[col]) if len(others) else 1
                xytext, ha, va = ((6, 5), 'left', 'baseline') if gap > tol else ((-6, -13), 'right', 'baseline') if gap < -tol else ((8, 0), 'left', 'center')
                arrow = None
                if (col, run, lam) in NUDGE:
                    (xytext, ha), va = NUDGE[(col, run, lam)], 'baseline'
                    arrow = dict(arrowstyle = '-', color = '#52514e', lw = 0.6, shrinkA = 1, shrinkB = 4) if abs(xytext[1]) > 10 else None
                ax.annotate(f'λ={lam:g}', (x, y), textcoords = 'offset points', xytext = xytext, ha = ha, va = va, fontsize = 8, color = '#52514e', arrowprops = arrow)
        ax.set_title(title, fontsize = 10)
        ax.set_ylabel(ylabel)
        ax.grid(alpha = 0.25)
        ax.spines[['top', 'right']].set_visible(False)

    # detection panels go from 0 to 1, the JPEG panel starts at chance level
    for ax in axes[0]:
        ax.set_ylim(-0.02, 1.02)
    axes[1, 0].set_ylim(0.45, 1.0)
    axes[1, 0].axhline(0.5, color = '#52514e', lw = 0.8, ls = '--')
    axes[1, 0].text(models.psnr.max(), 0.508, 'chance ', fontsize = 8, color = '#52514e', ha = 'right')

    # FID references: the original generator, and the original generator plus random noise of the same PSNR
    ax = axes[1, 1]
    ax.axhline(g0, color = '#52514e', lw = 0.8, ls = ':')
    ax.text(models.psnr.max(), g0 + 1.5, f'G0, not watermarked ({g0:.1f}) ', fontsize = 8, color = '#52514e', ha = 'right')
    ax.plot(noise.psnr, noise.fid, color = '#0b0b0b', lw = 1.2, ls = '--', marker = 'x', ms = 7)
    ax.text(noise.psnr.iloc[0], noise.fid.iloc[0] + 2.5, 'G0 + random noise', fontsize = 8, color = '#0b0b0b')
    ax.set_ylim(20, max(models.fid.max(), noise.fid.max()) + 8)

    # some room on both sides for the labels of the first and last points
    axes[0, 0].set_xlim(models.psnr.min() - 1.2, models.psnr.max() + 1.2)
    # the panels share the x axis, values and label shown on all of them (matplotlib hides them on the top row)
    for ax in axes.flat:
        ax.tick_params(labelbottom = True)
        ax.set_xlabel('PSNR of G_theta(z) against G0(z) (dB)')
    plt.tight_layout()
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc = 'upper center', ncol = len(RUNS), fontsize = 9, frameon = False, bbox_to_anchor = (0.5, 1.04))
    Path(args.out).parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(args.out, dpi = 200, bbox_inches = 'tight')
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
