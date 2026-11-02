# Training curves of a train_hidden.py run: held-out bit accuracy per attack (real and generated images) and PSNR.
#   python -m watermark.plot_logs runs/hidden                 # writes runs/hidden/curves.png
#   python -m watermark.plot_logs runs/hidden --warmup 1 --show   # run trained with --warmup 1, also opens a window
import argparse
from pathlib import Path
import pandas as pd

ATTACKS = ['none', 'jpeg80', 'jpeg50', 'crop0.5', 'noise0.05', 'combined']
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']  # an attack keeps its colour in every panel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir')
    p.add_argument('--warmup', type = int, default = 2)  # the --warmup of the training run, to mark when attacks started
    p.add_argument('--show', action = 'store_true')
    args = p.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = Path(args.dir)
    df = pd.read_csv(d / 'log_eval.csv')
    switch = df.loc[df.epoch >= args.warmup, 'step'].min()  # first eval after the attacks were switched on
    sets = [t for t in ['real', 'gen'] if f'{t}_none' in df]

    fig, axes = plt.subplots(1, len(sets) + 1, figsize = (5*(len(sets) + 1), 3.8))
    for ax, t in zip(axes, sets):
        for a, c in zip(ATTACKS, COLORS):
            ax.plot(df.step, df[f'{t}_{a}'], color = c, lw = 1.8, label = a)
        ax.plot(df.step, df[f'{t}_clean'], color = '#52514e', lw = 1.2, ls = '--', label = 'clean (control)')
        ax.axhline(30/32, color = '#52514e', lw = 0.8, ls = ':')
        ax.text(df.step.iloc[0], 30/32 + 0.008, '30/32 bits (detection threshold)', fontsize = 8, color = '#52514e')
        ax.set_ylim(0.45, 1.0)
        ax.set_title(f"held-out bit accuracy, {'real' if t == 'real' else 'generated'} images", fontsize = 10)
    for t, ls in zip(sets, ['-', '--']):
        axes[-1].plot(df.step, df[f'{t}_psnr'], color = '#0b0b0b', lw = 1.5, ls = ls, label = t)
    axes[-1].axhline(32, color = '#52514e', lw = 0.8, ls = ':')
    axes[-1].text(df.step.iloc[0], 32.3, 'floor at alpha = 0.05', fontsize = 8, color = '#52514e')
    axes[-1].set_title('PSNR of x_w against x (dB)', fontsize = 10)
    axes[-1].legend(fontsize = 8, frameon = False)
    for ax in axes:
        if pd.notna(switch):
            ax.axvline(switch, color = '#52514e', lw = 0.8, alpha = 0.5)
            ax.text(switch, ax.get_ylim()[0], ' attacks on', fontsize = 8, color = '#52514e', va = 'bottom')
        ax.set_xlabel('step')
        ax.grid(alpha = 0.25)
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc = 'upper center', ncol = 7, fontsize = 9, frameon = False, bbox_to_anchor = (0.5, 1.07))
    fig.savefig(d / 'curves.png', dpi = 150, bbox_inches = 'tight')

    cols = ['step', 'epoch'] + [f'{t}_{c}' for t in sets for c in ['psnr', 'clean'] + ATTACKS]
    print(df[cols].tail(3).round(3).to_string(index = False))
    print(f"saved {d / 'curves.png'}")
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
