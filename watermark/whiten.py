# End of step 3: fit the whitening layer W' = Sigma^{-1/2} (W - mu) on clean images (real CelebA + generator samples),
# then check the null model of the detection test on other clean images: for the secret key, M ~ Bin(k, 1/2).
# Run from the repo root, for example on Kaggle:
#   python -m watermark.whiten --ckpt /kaggle/working/hidden/last.pt --g_ckpt <generator last.pt> --data_root /tmp/celeba
# Writes next to --ckpt: w_prime.pt (W + whitening, load with hidden.load_whitened), key.pt (m*), whiten_hist.png, whiten_report.txt
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from scipy.stats import chi2
from data import CelebA64
from models.utils import seed_all
from watermark.attacks import quantize
from watermark.detect import threshold, fpr, binom_pmf, matches, load_or_make_key
from watermark.hidden import Encoder, Extractor, random_msg, bit_acc
from watermark.train_hidden import load_G, Z_DIM, N_TRAIN

# official CelebA split, 0-based: train < 162770 <= validation < 182637 <= test.
# train_hidden.py evaluates on the first 1000 validation images, so the fit starts after them
FIT_START = N_TRAIN + 1000
TEST_START = 182637


@torch.no_grad()
def logits_real(W, root, start, n, bs, dev, workers):
    ds = CelebA64(root, n = start + n, start = start)
    assert len(ds) == n, f"only {len(ds)} images from index {start} in {root}"
    return torch.cat([W(x.to(dev)).cpu() for x in DataLoader(ds, batch_size = bs, num_workers = workers)])


# generated images rounded to uint8, as they would be saved
@torch.no_grad()
def logits_gen(W, netG, n, bs, dev, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, n, bs):
        z = torch.randn(min(bs, n - i), Z_DIM, generator = g).to(dev)
        out.append(W(quantize(netG(z))).cpu())
    return torch.cat(out)


# mu and P = Sigma^{-1/2} = U diag(lambda^{-1/2}) U^T from the eigendecomposition of the covariance
def fit_whitening(y, eps=1e-6):
    y = y.double()
    mu = y.mean(0)
    lam, U = torch.linalg.eigh(torch.cov((y - mu).T))
    P = U @ torch.diag(lam.clamp_min(eps*lam.max()).rsqrt()) @ U.T
    return mu.float(), P.float()


# chi-square goodness of fit of the observed M counts to Bin(k, 1/2), tails merged until expected counts are >= 5
def chi2_binomial(obs, k, min_exp=5):
    exp = obs.sum()*binom_pmf(k)
    lo, hi = 0, k
    while exp[:lo + 1].sum() < min_exp:
        lo += 1
    while exp[hi:].sum() < min_exp:
        hi -= 1
    o = torch.cat([obs[:lo + 1].sum().view(1), obs[lo + 1:hi], obs[hi:].sum().view(1)])
    e = torch.cat([exp[:lo + 1].sum().view(1), exp[lo + 1:hi], exp[hi:].sum().view(1)])
    return chi2.sf(((o - e)**2/e).sum().item(), len(o) - 1)


# how far the decoded bits of clean images are from k fair independent coins, and what that does to M
def null_check(y, key, k):
    b = (y > 0).float()
    c = torch.nan_to_num(torch.corrcoef(b.T))  # a bit that never flips has no correlation: counted as 0
    M = matches(y, key)
    obs = torch.bincount(M.long(), minlength = k + 1).double()
    r = {'worst bit |P(1) - 0.5|': (b.mean(0) - 0.5).abs().max().item(),
         'max |corr| between bits': (c - torch.diag(torch.diag(c))).abs().max().item(),
         f'mean of M (H0: {k/2:g})': M.float().mean().item(),
         f'variance of M (H0: {k/4:g})': M.float().var().item(),
         'chi-square p-value': chi2_binomial(obs, k)}
    for target in [1e-2, 1e-3]:
        t = threshold(k, target)
        r[f'FPR at tau={t} (H0: {fpr(t, k):.1e})'] = (M >= t).float().mean().item()
    return r, obs


def plot_hist(path, counts, k, tau):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x, pmf = list(range(k + 1)), binom_pmf(k)
    fig, axes = plt.subplots(1, len(counts), figsize = (5*len(counts), 3.6), sharey = True)
    for ax, (title, sets) in zip(axes, counts.items()):
        ax.plot(x, pmf, color = '#52514e', lw = 1.2, ls = '--', label = 'Bin(k, 1/2)')
        for (name, obs), c in zip(sets.items(), ['#2a78d6', '#eb6834']):
            f = obs/obs.sum()
            f[f == 0] = float('nan')  # empty bins left out of the log scale
            ax.plot(x, f, color = c, lw = 1.8, marker = 'o', ms = 4, label = f'clean {name} images')
        ax.axvline(tau, color = '#52514e', lw = 0.8, ls = ':')
        ax.text(tau, 0.3, f' tau = {tau}\n FPR 1e-6', fontsize = 8, color = '#52514e', va = 'top')
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1)
        ax.set_title(title, fontsize = 10)
        ax.set_xlabel('M = decoded bits matching the key')
        ax.grid(alpha = 0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('fraction of images')
    plt.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc = 'upper center', ncol = 3, fontsize = 9, frameon = False, bbox_to_anchor = (0.5, 1.08))
    fig.savefig(path, dpi = 150, bbox_inches = 'tight')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required = True)  # train_hidden.py last.pt
    p.add_argument('--g_ckpt', required = True)  # the generator to watermark (train_gan.py last.pt)
    p.add_argument("--data_root", default = 'data/img_align_celeba')
    p.add_argument('--out', default = None)  # default: the folder of --ckpt
    p.add_argument('--n_fit', type = int, default = 10000)  # half real validation images, half generator samples
    p.add_argument('--n_check', type = int, default = 10000)  # per source: real test images, fresh generator samples
    p.add_argument('--bs', type = int, default = 500)
    p.add_argument('--workers', type = int, default = 2)
    p.add_argument('--key_seed', type = int, default = 0)
    args = p.parse_args()

    seed_all(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out or Path(args.ckpt).parent)
    out.mkdir(parents = True, exist_ok = True)

    ck = torch.load(args.ckpt, map_location = dev)
    k = ck['cfg']['k']
    W = Extractor(k).to(dev).eval()
    W.load_state_dict(ck['W'])
    netG = load_G(args.g_ckpt, dev)
    key = load_or_make_key(out / 'key.pt', k, args.key_seed)
    print(f"checkpoint {args.ckpt} (epoch {ck['epoch'] + 1}, k = {k}), key m* = {''.join(str(int(b)) for b in key)}")

    #1 fit on clean images
    n_real = args.n_fit//2
    y_fit = torch.cat([logits_real(W, args.data_root, FIT_START, n_real, args.bs, dev, args.workers),
                       logits_gen(W, netG, args.n_fit - n_real, args.bs, dev, seed = 1)])
    mu, P = fit_whitening(y_fit)
    print(f"whitening fitted on {n_real} real + {args.n_fit - n_real} generated clean images")

    #2 check the null model on other clean images, before and after whitening
    y_check = {'real': logits_real(W, args.data_root, TEST_START, args.n_check, args.bs, dev, args.workers),
               'generated': logits_gen(W, netG, args.n_check, args.bs, dev, seed = 2)}
    white = lambda y: (y - mu) @ P.T
    results, counts = {}, {'raw W': {}, "whitened W'": {}}
    for name, y in y_check.items():
        results[f'W {name}'], counts['raw W'][name] = null_check(y, key, k)
        results[f"W' {name}"], counts["whitened W'"][name] = null_check(white(y), key, k)

    #3 W' must still read the messages the encoder hides: bit accuracy on watermarked test images, no attack
    E = Encoder(k, alpha = ck['cfg']['alpha']).to(dev).eval()
    E.load_state_dict(ck['E'])
    ds = CelebA64(args.data_root, n = TEST_START + 1000, start = TEST_START)
    x = torch.stack([ds[i] for i in range(len(ds))])
    m = random_msg(len(x), k, generator = torch.Generator().manual_seed(3))
    with torch.no_grad():
        y_w = torch.cat([W(quantize(E(x[i:i+250].to(dev), m[i:i+250].to(dev)))).cpu() for i in range(0, len(x), 250)])

    lines = [f"{'clean images (' + str(args.n_check) + ' per source)':<36}" + ''.join(f"{c:>14}" for c in results)]
    for row in next(iter(results.values())):
        lines.append(f"{row:<36}" + ''.join(f"{results[c][row]:>14.4g}" for c in results))
    lines.append(f"bit accuracy on 1000 watermarked test images: W {bit_acc(y_w, m).item():.3f}, W' {bit_acc(white(y_w), m).item():.3f}")
    print('\n'.join(lines))
    (out / 'whiten_report.txt').write_text('\n'.join(lines) + '\n')

    torch.save({'W': {n: v.cpu() for n, v in W.state_dict().items()}, 'mu': mu, 'P': P, 'cfg': ck['cfg']}, out / 'w_prime.pt')
    plot_hist(out / 'whiten_hist.png', counts, k, threshold(k, 1e-6))
    print(f"saved {out / 'w_prime.pt'}, {out / 'key.pt'}, {out / 'whiten_hist.png'}, {out / 'whiten_report.txt'}")


if __name__ == '__main__':
    main()
