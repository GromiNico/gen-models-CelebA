#run from the repo root: python -m watermark.eval_wm --g_ckpt ... --wprime ... --models "runs/finetune/lam*/G_theta.pt" --keep ...
import argparse
import csv
import glob
from pathlib import Path
import torch
import torchvision.utils as vu
import lpips
import numpy as np
from torch.utils.data import DataLoader
from data import CelebA64
from models.dcgan import G
from watermark.attacks import EVAL_2DAY, eval_attack
from watermark.detect import threshold, matches, binom_pmf, load_or_make_key
from watermark.hidden import load_whitened, psnr
from watermark.train_hidden import load_G, Z_DIM
from watermark.whiten import TEST_START

ATTACKS = ['none'] + EVAL_2DAY
COLS = ['psnr', 'lpips'] + [f'acc_{a}' for a in ATTACKS] + [f'tpr_{a}' for a in ATTACKS] + ['fid']


# loads a fine-tuned generator, its lambda and whether it was fine-tuned through the attacks
def load_theta(path, dev):
    ck = torch.load(path, map_location = dev)
    netG = G(Z_DIM).to(dev)
    netG.load_state_dict(ck['netG'])
    attack = ck.get('attack')
    if attack is None:  # older checkpoints don't store it, it is read from the summary.csv of their sweep
        s = Path(path).parent.parent / 'summary.csv'
        rows = [r for r in csv.DictReader(open(s)) if r['lam'] and float(r['lam']) == ck['lam']] if s.exists() else []
        attack = rows[-1]['attack'] == 'True' if rows else None
    return netG.eval().requires_grad_(False), ck['lam'], attack


# M (bits matching the key) under every eval attack, for a stream of image batches
@torch.no_grad()
def match_counts(batches, Wp, key):
    M = {a: [] for a in ATTACKS}
    for x in batches:
        for a in ATTACKS:
            M[a].append(matches(Wp(eval_attack(x, a)), key).cpu())
    return {a: torch.cat(v) for a, v in M.items()}


# bit accuracy and detection rate (M >= tau) under every attack
def robustness(M, k, tau):
    return {**{f'acc_{a}': (M[a]/k).mean().item() for a in ATTACKS}, **{f'tpr_{a}': (M[a] >= tau).float().mean().item() for a in ATTACKS}}


# PSNR and LPIPS of G_theta against G0 on the same latents
@torch.no_grad()
def quality(Gt, z, x0, perc, bs):
    p, l = 0.0, 0.0
    for i in range(0, len(z), bs):
        xt = Gt(z[i:i+bs])
        p += psnr(xt, x0[i:i+bs]).item()*len(xt)
        l += perc(xt, x0[i:i+bs]).sum().item()
    return {'psnr': p/len(z), 'lpips': l/len(z)}


# bit accuracy against PSNR, one point per lambda, without attack and under JPEG 50
def plot_tradeoff(path, rows, keep_label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series = [(True, 'fine-tuned through attacks', '#2a78d6', 'o'), (False, 'fine-tuned without attacks', '#eb6834', 's'), (None, 'attack setting unknown', '#52514e', '^')]
    fig, axes = plt.subplots(1, 2, figsize = (10, 3.8), sharey = True)
    for ax, a, title in zip(axes, ['none', 'jpeg50'], ['no attack', 'JPEG 50']):
        for flag, name, c, mk in series:
            pts = sorted([r for r in rows if r['attack'] == flag], key = lambda r: r['psnr'])
            if not pts:
                continue
            ax.plot([r['psnr'] for r in pts], [r[f'acc_{a}'] for r in pts], color = c, lw = 1.5, marker = mk, ms = 7, label = name)
            others = sorted([r for r in rows if r['attack'] != flag], key = lambda r: r['psnr'])
            for r in pts:
                # label on the side away from the other series at the same PSNR: above right or below left
                other = np.interp(r['psnr'], [o['psnr'] for o in others], [o[f'acc_{a}'] for o in others]) if others else -1
                above = r[f'acc_{a}'] >= other
                ax.annotate(f"λ={r['lam']:g}", (r['psnr'], r[f'acc_{a}']), textcoords = 'offset points', xytext = (6, 5) if above else (-6, -13),
                            ha = 'left' if above else 'right', fontsize = 8, color = '#52514e')
                if r['label'] == keep_label:
                    ax.plot(r['psnr'], r[f'acc_{a}'], 'o', ms = 14, mfc = 'none', mec = '#0b0b0b', mew = 1.2)
        ax.set_xlim(min(r['psnr'] for r in rows) - 2, max(r['psnr'] for r in rows) + 1.5)
        ax.axhline(30/32, color = '#52514e', lw = 0.8, ls = ':')
        ax.text(ax.get_xlim()[1], 30/32 + 0.008, '30/32 bits (detection threshold) ', fontsize = 8, color = '#52514e', ha = 'right')
        ax.axhline(0.5, color = '#52514e', lw = 0.8, ls = '--')
        ax.text(ax.get_xlim()[0], 0.508, ' chance', fontsize = 8, color = '#52514e')
        ax.set_title(f'bit accuracy, {title}', fontsize = 10)
        ax.set_xlabel('PSNR of G_theta(z) against G0(z) (dB)')
        ax.grid(alpha = 0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_ylabel('bit accuracy (32 bits)')
    plt.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc = 'upper center', ncol = 3, fontsize = 9, frameon = False, bbox_to_anchor = (0.5, 1.08))
    fig.savefig(path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)


# histogram of M on clean and watermarked images against Bin(k, 1/2)
def plot_hist(path, counts, k, tau):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = list(range(k + 1))
    fig, ax = plt.subplots(figsize = (7.5, 4.2))
    ax.plot(x, binom_pmf(k), color = '#52514e', lw = 1.2, ls = '--', label = f'Bin({k}, 1/2): bits matching the key by pure chance')
    for (name, M), c in zip(counts.items(), ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']):
        f = torch.bincount(M.long(), minlength = k + 1).double()/len(M)
        f[f == 0] = float('nan')  # empty bins are not drawn on the log scale
        ax.plot(x, f, color = c, lw = 1.8, marker = 'o', ms = 4, label = name)
    ax.axvline(tau, color = '#52514e', lw = 0.8, ls = ':', label = f'detection threshold: M >= {tau} (false-positive rate 1e-6)')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel(f'M = number of the {k} decoded bits that match the secret key')
    ax.set_ylabel('fraction of images (log scale)')
    ax.grid(alpha = 0.25)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize = 8, frameon = False, loc = 'upper center', bbox_to_anchor = (0.5, -0.17), ncol = 2)
    fig.savefig(path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)


def main():
    #arguments
    p = argparse.ArgumentParser()
    p.add_argument('--g_ckpt', required = True)  # the original generator G0
    p.add_argument('--wprime', required = True)  # w_prime.pt from whiten.py, the key is read from key.pt next to it
    p.add_argument('--models', nargs = '+', required = True)  # G_theta.pt files or patterns like "runs/finetune/lam*/G_theta.pt"
    p.add_argument('--keep', default = None)  # the chosen G_theta: attack table, grid, histogram (and FID)
    p.add_argument('--n', type = int, default = 2000)  # number of fixed latents per model (10000 on GPU)
    p.add_argument('--n_psnr', type = int, default = 1000)
    p.add_argument('--fid', action = 'store_true')  # FID of G0 and of --keep (slow on CPU)
    p.add_argument('--fid_ref', default = 'celeba64_ref')  # clean-fid statistics made by train_gan.py --fid_real
    p.add_argument('--fid_n', type = int, default = 10000)
    p.add_argument("--data_root", default = 'data/img_align_celeba')
    p.add_argument('--perc', default = 'vgg', choices = ['vgg', 'alex'])
    p.add_argument('--bs', type = int, default = 250)
    p.add_argument('--workers', type = int, default = 0)
    p.add_argument('--out', default = 'runs/eval_wm')
    args = p.parse_args()

    #2 setup, GPU, out dir and the list of models
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)
    keep = str(Path(args.keep)) if args.keep else None  # normalized so that runs/a and runs\a are the same model
    paths = sorted({str(Path(q)) for pat in args.models for q in glob.glob(pat)} | ({keep} if keep else set()))
    assert paths, f"no G_theta found for {args.models}"

    #3 frozen G0 and W', key, detection threshold and LPIPS
    G0 = load_G(args.g_ckpt, dev)
    Wp = load_whitened(args.wprime, dev)
    k = Wp.P.shape[0]
    key = load_or_make_key(Path(args.wprime).parent / 'key.pt', k).to(dev)
    tau = threshold(k, 1e-6)
    perc = lpips.LPIPS(net = args.perc, verbose = False).to(dev).eval().requires_grad_(False)

    #4 the same latents for every model and every evaluation run
    z_path = out / 'z_eval.pt'
    z = torch.load(z_path) if z_path.exists() else torch.randn(10000, Z_DIM, generator = torch.Generator().manual_seed(2024))
    torch.save(z, z_path)
    z = z[:args.n].to(dev)
    with torch.no_grad():
        x0 = torch.cat([G0(z[i:i+args.bs]) for i in range(0, len(z), args.bs)])
    print(f"{len(paths)} models, {len(z)} latents, key m* = {''.join(str(int(b)) for b in key)}, tau = {tau}/{k} (FPR 1e-6)")

    #5 clean real and G0 images: false alarms of the detector under each attack
    real = CelebA64(args.data_root, n = TEST_START + args.n, start = TEST_START)
    M_real = match_counts((x.to(dev) for x in DataLoader(real, batch_size = args.bs, num_workers = args.workers)), Wp, key)
    M_g0 = match_counts((x0[i:i+args.bs] for i in range(0, len(x0), args.bs)), Wp, key)
    rows = [{'label': 'clean real (test split)', 'lam': None, 'attack': None, **robustness(M_real, k, tau)},
            {'label': 'G0 (not watermarked)', 'lam': None, 'attack': None, **robustness(M_g0, k, tau)}]

    #6 every fine-tuned generator: quality against G0 and robustness
    models, M_keep = {}, None
    for path in paths:
        Gt, lam, attack = load_theta(path, dev)
        label = f"lam={lam:g}" + {True: ' +attacks', False: '', None: ' (attacks?)'}[attack]
        M = match_counts((Gt(z[i:i+args.bs]) for i in range(0, len(z), args.bs)), Wp, key)
        rows.append({'label': label, 'lam': lam, 'attack': attack, 'path': path,
                     **quality(Gt, z[:args.n_psnr], x0[:args.n_psnr], perc, args.bs), **robustness(M, k, tau)})
        r = rows[-1]
        print(f"  {label:20s} PSNR {r['psnr']:.1f} dB, LPIPS {r['lpips']:.4f} | bit acc / TPR: "
              + ", ".join(f"{a} {r['acc_' + a]:.3f} / {r['tpr_' + a]:.2f}" for a in ATTACKS))
        models[path] = Gt
        if path == keep:
            keep_label, M_keep = label, M

    #7 results file and trade-off figure
    def write_results():
        with open(out / 'results.csv', 'w', newline = '') as f:
            w = csv.writer(f)
            w.writerow(['model', 'lam', 'attack_trained'] + COLS + ['path'])
            for r in rows:
                w.writerow([r['label'], r['lam'], r['attack']] + [r.get(c, '') for c in COLS] + [r.get('path', '')])
    write_results()
    plot_tradeoff(out / 'tradeoff.png', [r for r in rows if r['lam'] is not None], keep_label if M_keep is not None else None)

    #8 kept model: attack table, histogram of M, image grid and FID
    if M_keep is not None:
        # attack table, with the false alarm rate of clean images under the same attack
        r = next(r for r in rows if r['label'] == keep_label)
        lines = [f"{'attack':10s} {'bit acc':>8s} {'TPR':>6s} {'FPR real':>9s} {'FPR G0':>8s}   ({keep_label}, {len(z)} latents, tau = {tau})"]
        with open(out / 'attacks_kept.csv', 'w', newline = '') as f:
            w = csv.writer(f)
            w.writerow(['attack', 'bit_acc', 'tpr', 'fpr_real', 'fpr_g0'])
            for a in ATTACKS:
                vals = [r[f'acc_{a}'], r[f'tpr_{a}'], rows[0][f'tpr_{a}'], rows[1][f'tpr_{a}']]
                w.writerow([a] + vals)
                lines.append(f"{a:10s} {vals[0]:8.3f} {vals[1]:6.3f} {vals[2]:9.4f} {vals[3]:8.4f}")
        print('\n'.join(lines))

        how = f"λ = {r['lam']:g}" + (', fine-tuned through attacks' if r['attack'] else '')
        plot_hist(out / 'hist_M.png', {'real photos, no watermark': M_real['none'],
                                       'original generator G0, no watermark': M_g0['none'],
                                       f'watermarked generator ({how})': M_keep['none'],
                                       'same watermarked images after JPEG 50': M_keep['jpeg50']}, k, tau)
        # grid of 8 latents: G0(z), G_theta(z) and the amplified difference 10*|G_theta(z) - G0(z)|
        with torch.no_grad():
            xt = models[keep](z[:8])
        vu.save_image(torch.cat([x0[:8]*0.5 + 0.5, xt*0.5 + 0.5, (10*(xt - x0[:8]).abs()).clamp(0, 1)]), out / 'grid.png', nrow = 8)

        if args.fid:
            from cleanfid import fid
            from eval_fid import fid_eval
            if not fid.test_stats_exists(args.fid_ref, mode = 'clean'):
                print(f"no clean-fid statistics named {args.fid_ref}: make them with eval_fid.make_ref_stats, FID skipped")
            else:
                for r, net in [(rows[1], G0), (r, models[keep])]:
                    r['fid'] = fid_eval(args.fid_ref, net, str(out / 'fid_samples'), n = args.fid_n, z_dim = Z_DIM, device = dev)
                    net.eval()  # sample_to_dir leaves the generator in train mode
                    print(f"  FID {r['label']}: {r['fid']:.2f}")
                write_results()
    print(f"saved results.csv, tradeoff.png" + (", attacks_kept.csv, hist_M.png, grid.png" if M_keep is not None else "") + f" in {out}")


if __name__ == '__main__':
    main()
