# Step 4: fine-tune a copy G_theta of the generator so that every sample carries the key m*, as read by the frozen
# whitened extractor W':  min_theta E_z[ BCE(W'(G_theta(z)), m*) + lambda * LPIPS(G_theta(z), G0(z)) ]
# Run from the repo root (CPU is fine at 64x64), one fine-tuning per lambda:
#   python -m watermark.finetune_gen --g_ckpt runs/wgangp/last.pt --wprime runs/hidden_ep12/w_prime.pt --lam 0.5 2 8
# Writes <out>/lam<lambda>/: G_theta.pt (same format as train_gan.py, key 'netG'), log.csv, log_eval.csv, samples.png,
# and <out>/summary.csv with the final evaluation of G0 and of every G_theta.
import argparse
import copy
import csv
import math
import random
import time
from pathlib import Path
import torch
import torchvision.utils as vu
import lpips
from models.utils import seed_all
from watermark.attacks import TRAIN_ATTACKS, train_attack, eval_attack
from watermark.detect import threshold, matches, load_or_make_key
from watermark.hidden import load_whitened, msg_loss, bit_acc, psnr
from watermark.train_hidden import load_G, Z_DIM

EVAL_ATTACKS = ['none', 'jpeg80', 'jpeg50', 'crop0.5', 'combined']
EVAL_COLS = ['psnr', 'lpips'] + [f'acc_{a}' for a in EVAL_ATTACKS] + [f'tpr_{a}' for a in EVAL_ATTACKS]


# linear warm-up to lr, then cosine decay to lr_min (Stable Signature: 20 warm-up steps, 1e-4 -> 1e-6)
def lr_at(step, steps, lr, warmup=20, lr_min=1e-6):
    if step < warmup:
        return lr*(step + 1)/warmup
    t = (step - warmup)/max(1, steps - warmup)
    return lr_min + 0.5*(lr - lr_min)*(1 + math.cos(math.pi*t))


# on fixed latents: quality of G_theta against G0 (PSNR, LPIPS), bit accuracy against the key and detection rate
# (TPR: fraction with M >= tau, tau set for FPR 1e-6) under a few eval attacks; images rounded to uint8 first
@torch.no_grad()
def evaluate(G0, Gt, Wp, perc, key, z, tau, bs=128):
    res = dict.fromkeys(EVAL_COLS, 0.0)
    for i in range(0, len(z), bs):
        x0, xt = G0(z[i:i+bs]), Gt(z[i:i+bs])
        n = len(x0)
        res['psnr'] += psnr(xt, x0).item()*n
        res['lpips'] += perc(xt, x0).sum().item()
        for a in EVAL_ATTACKS:
            y = Wp(eval_attack(xt, a))
            res[f'acc_{a}'] += bit_acc(y, key.expand_as(y)).item()*n
            res[f'tpr_{a}'] += (matches(y, key) >= tau).sum().item()
    return {c: v/len(z) for c, v in res.items()}


def show(name, r):
    print(f"  {name}: PSNR {r['psnr']:.1f} dB, LPIPS {r['lpips']:.4f} | bit acc / TPR: "
          + ", ".join(f"{a} {r['acc_' + a]:.3f} / {r['tpr_' + a]:.2f}" for a in EVAL_ATTACKS))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--g_ckpt', required = True)  # the generator to watermark (train_gan.py last.pt)
    p.add_argument('--wprime', required = True)  # whiten.py output; the key is read from key.pt next to it
    p.add_argument('--lam', type = float, nargs = '+', default = [0.5, 2.0, 8.0])
    p.add_argument('--steps', type = int, default = 100)
    p.add_argument('--bs', type = int, default = 4)
    p.add_argument('--lr', type = float, default = 1e-4)
    p.add_argument('--perc', default = 'vgg', choices = ['vgg', 'alex'])  # LPIPS backbone; vgg is advised for optimization
    p.add_argument('--attack', action = 'store_true')  # ablation: pass G_theta(z) through the train-time attack layer
    p.add_argument('--jpeg_grad', default = 'ste', choices = ['ste', 'kornia'])  # gradient of the train-time JPEG
    p.add_argument('--n_eval', type = int, default = 256)
    p.add_argument('--eval_every', type = int, default = 50)
    p.add_argument('--out', default = 'runs/finetune')
    args = p.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)

    G0 = load_G(args.g_ckpt, dev)  # frozen, eval mode
    g_epoch = torch.load(args.g_ckpt, map_location = 'cpu')['epoch']
    Wp = load_whitened(args.wprime, dev)  # frozen W'
    k = Wp.P.shape[0]
    key = load_or_make_key(Path(args.wprime).parent / 'key.pt', k).to(dev)
    tau = threshold(k, 1e-6)
    perc = lpips.LPIPS(net = args.perc, verbose = False).to(dev).eval().requires_grad_(False)
    z_eval = torch.randn(args.n_eval, Z_DIM, generator = torch.Generator().manual_seed(4321)).to(dev)
    print(f"key m* = {''.join(str(int(b)) for b in key)}, tau = {tau}/{k} (FPR 1e-6), LPIPS-{args.perc}, {args.steps} steps of batch {args.bs}")

    summary = open(out / 'summary.csv', 'w', newline = '')
    writer_sum = csv.writer(summary)
    writer_sum.writerow(['model', 'lam', 'steps', 'bs', 'attack'] + EVAL_COLS)
    r0 = evaluate(G0, G0, Wp, perc, key, z_eval, tau)
    show('G0 (not watermarked)', r0)
    writer_sum.writerow(['G0', '', '', '', ''] + [r0[c] for c in EVAL_COLS])
    summary.flush()

    for lam in args.lam:
        seed_all(0)
        d = out / f'lam{lam:g}'
        d.mkdir(parents = True, exist_ok = True)
        # G_theta starts as G0 and stays in eval mode: BatchNorm keeps the running statistics of G0
        Gt = copy.deepcopy(G0).requires_grad_(True)
        opt = torch.optim.AdamW(Gt.parameters(), lr = args.lr)
        log, log_eval = open(d / 'log.csv', 'w', newline = ''), open(d / 'log_eval.csv', 'w', newline = '')
        writer, writer_eval = csv.writer(log), csv.writer(log_eval)
        writer.writerow(['step', 'lr', 'loss', 'loss_msg', 'loss_perc', 'bit_acc', 'psnr'])
        writer_eval.writerow(['step'] + EVAL_COLS)
        print(f"lambda = {lam:g}")
        t0 = time.time()

        for step in range(args.steps):
            for group in opt.param_groups:
                group['lr'] = lr_at(step, args.steps, args.lr)
            z = torch.randn(args.bs, Z_DIM, device = dev)
            with torch.no_grad():
                x0 = G0(z)
            xt = Gt(z)
            y = Wp(train_attack(xt, random.choice(TRAIN_ATTACKS), args.jpeg_grad) if args.attack else xt)
            l_msg = msg_loss(y, key.expand_as(y))
            l_perc = perc(xt, x0).mean()
            loss = l_msg + lam*l_perc
            opt.zero_grad(set_to_none = True)
            loss.backward()
            opt.step()

            if step % 10 == 0:
                acc, p_t = bit_acc(y, key.expand_as(y)).item(), psnr(xt.detach(), x0).item()
                writer.writerow([step, opt.param_groups[0]['lr'], loss.item(), l_msg.item(), l_perc.item(), acc, p_t])
                log.flush()
                print(f"  step {step} loss {loss.item():.3f} (msg {l_msg.item():.3f}, perc {l_perc.item():.4f}) bit_acc {acc:.3f} PSNR {p_t:.1f} dB")
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                r = evaluate(G0, Gt, Wp, perc, key, z_eval, tau)
                writer_eval.writerow([step + 1] + [r[c] for c in EVAL_COLS])
                log_eval.flush()
                show(f'step {step + 1}', r)
        log.close()
        log_eval.close()

        writer_sum.writerow(['G_theta', lam, args.steps, args.bs, args.attack] + [r[c] for c in EVAL_COLS])
        summary.flush()
        torch.save({'netG': Gt.state_dict(), 'lam': lam, 'key': key.cpu(), 'g_ckpt': args.g_ckpt, 'steps': args.steps,
                    'attack': args.attack, 'jpeg_grad': args.jpeg_grad, 'epoch': g_epoch}, d / 'G_theta.pt')
        # G0(z), G_theta(z) and the amplified difference 10*|G_theta(z) - G0(z)| for 8 latents
        with torch.no_grad():
            x0, xt = G0(z_eval[:8]), Gt(z_eval[:8])
        vu.save_image(torch.cat([x0*0.5 + 0.5, xt*0.5 + 0.5, (10*(xt - x0).abs()).clamp(0, 1)]), d / 'samples.png', nrow = 8)
        print(f"  saved {d / 'G_theta.pt'} ({time.time() - t0:.0f} s)")
    summary.close()


if __name__ == '__main__':
    main()
