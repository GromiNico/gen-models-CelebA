# Step 3: train the HiDDeN encoder E and extractor W jointly through the attack layer.
# Run from the repo root, for example on Kaggle:
#   python -m watermark.train_hidden --data_root <celeba jpgs> --out /kaggle/working/hidden --g_ckpt /kaggle/working/dcgan/last.pt
# Resumes from <out>/last.pt unless --fresh. Logs: log.csv (training batches), log_eval.csv (held-out bit accuracy per attack).
import argparse
import csv
import random
import time
from pathlib import Path
import torch
import torchvision.utils as vu
from torch.utils.data import DataLoader
from data import CelebA64
from models.dcgan import G
from models.utils import seed_all
from watermark.attacks import TRAIN_ATTACKS, EVAL_ATTACKS, EVAL_2DAY, train_attack, eval_attack
from watermark.hidden import Encoder, Extractor, random_msg, msg_loss, bit_acc, psnr

Z_DIM = 100
N_TRAIN = 162770  # official CelebA split: images 1-162770 are train, the next ones validation
EVAL_COLS = ['psnr', 'clean'] + list(EVAL_ATTACKS)


def load_G(path, dev):
    netG = G(Z_DIM).to(dev)
    netG.load_state_dict(torch.load(path, map_location = dev)['netG'])
    return netG.eval().requires_grad_(False)


# held-out bit accuracy under every eval attack, PSNR of x_w against x, and the clean-image control
# (W on the same images without watermark, against the same messages: should stay at 0.5)
@torch.no_grad()
def evaluate(E, W, x, m, bs=250):
    E.eval(); W.eval()
    res = dict.fromkeys(EVAL_COLS, 0.0)
    for i in range(0, len(x), bs):
        xb, mb = x[i:i+bs], m[i:i+bs]
        xw = E(xb, mb)
        res['psnr'] += psnr(xw, xb).item()*len(xb)
        res['clean'] += bit_acc(W(xb), mb).item()*len(xb)
        for a in EVAL_ATTACKS:
            res[a] += bit_acc(W(eval_attack(xw, a)), mb).item()*len(xb)
    E.train(); W.train()
    return {c: v/len(x) for c, v in res.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default = 'data/img_align_celeba')
    p.add_argument("--n", type = int, default = N_TRAIN)  # first n images for training
    p.add_argument("--n_val", type = int, default = 1000)  # held-out images taken after the train split
    p.add_argument("--k", type = int, default = 32)
    p.add_argument("--alpha", type = float, default = 0.05)
    p.add_argument("--epochs", type = int, default = 10)
    p.add_argument("--warmup", type = int, default = 2)  # epochs with the identity attack only
    p.add_argument("--attacks", nargs = '+', default = TRAIN_ATTACKS, choices = TRAIN_ATTACKS)
    p.add_argument("--bs", type = int, default = 128)
    p.add_argument("--lr", type = float)  # default 1e-3; when resuming, overrides the saved learning rate
    p.add_argument("--workers", type = int, default = 2)
    p.add_argument("--out", default = "runs/hidden")
    p.add_argument('--fresh', action = 'store_true')
    p.add_argument('--eval_every', type = int, default = 500)
    p.add_argument('--g_ckpt', default = None)  # generator checkpoint (train_gan.py last.pt): adds a generated held-out set
    p.add_argument('--g_frac', type = float, default = 0.0)  # fraction of each training batch replaced by generator samples
    args = p.parse_args()

    #setup, seed, GPU, out dir
    seed_all(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)
    cfg = {'k': args.k, 'alpha': args.alpha}

    #data: training batches, and fixed held-out images with fixed messages
    ds = CelebA64(args.data_root, n = args.n)
    loader = DataLoader(ds, batch_size = args.bs, shuffle = True, num_workers = args.workers, pin_memory = (dev == 'cuda'), drop_last = True)
    val = CelebA64(args.data_root, n = N_TRAIN + args.n_val, start = N_TRAIN)
    g = torch.Generator().manual_seed(1234)
    val_sets = {'real': (torch.stack([val[i] for i in range(len(val))]).to(dev), random_msg(len(val), args.k, generator = g).to(dev))}

    netG = None
    if args.g_ckpt:
        netG = load_G(args.g_ckpt, dev)
        z = torch.randn(args.n_val, Z_DIM, generator = g).to(dev)
        with torch.no_grad():
            x_gen = torch.cat([netG(z[i:i+250]) for i in range(0, args.n_val, 250)])
        val_sets['gen'] = (x_gen, random_msg(args.n_val, args.k, generator = g).to(dev))
    print(f"{len(ds)} training images, held-out sets: {', '.join(f'{t} ({len(v[0])})' for t, v in val_sets.items())}")

    E = Encoder(args.k, alpha = args.alpha).to(dev)
    W = Extractor(args.k).to(dev)
    opt = torch.optim.Adam(list(E.parameters()) + list(W.parameters()), lr = args.lr or 1e-3)

    #resume from last checkpoint if it exists
    start_epoch, step = 0, 0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists() and not args.fresh:
        ck = torch.load(ckpt_path, map_location = dev)
        assert ck['cfg'] == cfg, f"checkpoint was trained with {ck['cfg']}, not {cfg}: use --fresh or another --out"
        E.load_state_dict(ck['E']); W.load_state_dict(ck['W']); opt.load_state_dict(ck['opt'])
        start_epoch, step = ck['epoch'] + 1, ck['step']
        print(f"Resuming from {ckpt_path}, {start_epoch} epochs already done.")
        # the optimizer state restores the saved learning rate; --lr replaces it (e.g. a 10x drop for the last epochs)
        if args.lr is not None:
            for group in opt.param_groups:
                group['lr'] = args.lr
    print(f"learning rate {opt.param_groups[0]['lr']:g}")

    if start_epoch >= args.epochs:
        print(f"Nothing to do, already trained for {start_epoch} epochs, raise epochs or add --fresh to start again")
        return

    log = open(out / 'log.csv', 'a' if step > 0 else 'w', newline = '')
    log_eval = open(out / 'log_eval.csv', 'a' if step > 0 else 'w', newline = '')
    writer, writer_eval = csv.writer(log), csv.writer(log_eval)
    if step == 0:
        writer.writerow(['step', 'epoch', 'attack', 'loss', 'bit_acc', 'psnr'])
        writer_eval.writerow(['step', 'epoch'] + [f"{t}_{c}" for t in val_sets for c in EVAL_COLS])

    def run_eval(epoch):
        row = [step, epoch]
        for t, (xv, mv) in val_sets.items():
            r = evaluate(E, W, xv, mv)
            row += [r[c] for c in EVAL_COLS]
            print(f"  eval [{t}] PSNR {r['psnr']:.1f} dB, clean {r['clean']:.3f}, " + ", ".join(f"{a} {r[a]:.3f}" for a in ['none'] + EVAL_2DAY))
            if t == 'real':
                acc_real = r['none']
        writer_eval.writerow(row)
        log_eval.flush()
        return acc_real

    acc_real = None
    for epoch in range(start_epoch, args.epochs):
        # curriculum: identity only first, otherwise the bit accuracy often stays stuck at 50%
        names = ['identity'] if epoch < args.warmup else args.attacks
        if epoch == args.warmup:
            print(f"epoch {epoch}: attacks on ({', '.join(names)})")
            if acc_real is not None and acc_real < 0.9:
                print(f"  warning: held-out bit accuracy is only {acc_real:.3f} without attacks, consider a longer --warmup")
        E.train(); W.train()
        t0 = time.time()

        for x in loader:
            x = x.to(dev, non_blocking = True)
            if netG is not None and args.g_frac > 0:
                n_g = round(args.g_frac*x.size(0))
                with torch.no_grad():
                    x[:n_g] = netG(torch.randn(n_g, Z_DIM, device = dev))

            m = random_msg(x.size(0), args.k, device = dev)
            xw = E(x, m)
            a = random.choice(names)  # one attack for the whole batch
            logits = W(train_attack(xw, a))
            loss = msg_loss(logits, m)
            opt.zero_grad(set_to_none = True)
            loss.backward()
            opt.step()

            if step % 100 == 0:
                acc, p_w = bit_acc(logits, m).item(), psnr(xw.detach(), x).item()
                writer.writerow([step, epoch, a, loss.item(), acc, p_w])
                log.flush()
                print(f"epoch {epoch} step {step} attack {a} loss {loss.item():.3f} bit_acc {acc:.3f} PSNR {p_w:.1f} dB")
            step += 1
            if step % args.eval_every == 0:
                acc_real = run_eval(epoch)

        if step % args.eval_every != 0:  # otherwise already evaluated at this step
            acc_real = run_eval(epoch)

        # x, x_w and the amplified difference 10*|x_w - x| for 8 held-out images
        xv, mv = val_sets['real'][0][:8], val_sets['real'][1][:8]
        E.eval()
        with torch.no_grad():
            xw = E(xv, mv)
        E.train()
        vu.save_image(torch.cat([xv*0.5 + 0.5, xw*0.5 + 0.5, (10*(xw - xv).abs()).clamp(0, 1)]), out / f"samples_ep{epoch:02d}.png", nrow = 8)
        torch.save({'E': E.state_dict(), 'W': W.state_dict(), 'opt': opt.state_dict(), 'step': step, 'epoch': epoch, 'cfg': cfg}, ckpt_path)

        print(f"epoch {epoch} done in {time.time()-t0:.0f} s")
    log.close()
    log_eval.close()


if __name__ == '__main__':
    main()
