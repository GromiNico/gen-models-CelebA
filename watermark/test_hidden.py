#run from the repo root: python -m watermark.test_hidden (a few seconds), add --learn for a short training (~10 min on CPU)
import argparse
import math
import time
import torch
from watermark.hidden import Encoder, Extractor, random_msg, msg_loss, bit_acc, psnr

K, ALPHA = 32, 0.05


def test_shapes():
    E, W = Encoder(K), Extractor(K)
    x, m = torch.rand(4, 3, 64, 64)*2 - 1, random_msg(4, K)
    xw = E(x, m)
    assert xw.shape == x.shape, xw.shape
    assert W(xw).shape == (4, K)
    # thanks to the global average pooling W reads crops of any size
    for h, w in [(32, 32), (48, 40), (16, 64)]:
        assert W(torch.rand(2, 3, h, w)).shape == (2, K), (h, w)


def test_distortion_bound():
    E = Encoder(K, alpha = ALPHA)
    x = torch.rand(8, 3, 64, 64)*2 - 1
    x[:, :, :8] = 1.0  # saturated pixels must stay in range after the residual
    x[:, :, -8:] = -1.0
    xw = E(x, random_msg(8, K))
    assert (xw - x).abs().max() <= ALPHA + 1e-6
    assert xw.min() >= -1 and xw.max() <= 1
    # each pixel moves by at most alpha so PSNR >= 10 log10(4/alpha^2), 32 dB for alpha = 0.05
    assert psnr(xw, x) >= 10*math.log10(4/ALPHA**2) - 1e-3


def test_message_changes_output():
    E = Encoder(K).eval()
    x = torch.rand(4, 3, 64, 64)*2 - 1
    m = random_msg(4, K)
    with torch.no_grad():
        assert not torch.allclose(E(x, m), E(x, 1 - m))


def test_gradients():
    E, W = Encoder(K), Extractor(K)
    x, m = torch.rand(4, 3, 64, 64)*2 - 1, random_msg(4, K)
    msg_loss(W(E(x, m)), m).backward()
    for name, net in [('E', E), ('W', W)]:
        for n, p in net.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"{name}.{n}"
            assert p.grad.abs().sum() > 0, f"{name}.{n} has zero gradient"


def test_helpers():
    m = random_msg(16, K)
    good = (2*m - 1)*20  # confident logits with the right sign
    assert bit_acc(good, m) == 1 and bit_acc(-good, m) == 0
    assert msg_loss(good, m) < 1e-6
    assert abs(msg_loss(torch.zeros_like(m), m).item() - math.log(2)) < 1e-6
    x = torch.zeros(2, 3, 8, 8)
    assert abs(psnr(x + 0.02, x).item() - 40.0) < 1e-3  # MSE 4e-4 -> 10 log10(1e4)


# trains E and W for a few hundred steps on real images (identity attack only) and checks that W reads
# new random messages hidden in held-out images. On CPU it only shows that the bits start getting through,
# the full training is train_hidden.py on GPU. 'clean' (same images without watermark) must stay at chance
def test_learns(data_root, steps, bs, k, min_acc):
    from torch.utils.data import DataLoader
    from data import CelebA64

    torch.manual_seed(0)
    loader = DataLoader(CelebA64(data_root, n = 2000), batch_size = bs, shuffle = True, drop_last = True)
    held = CelebA64(data_root, n = 2064, start = 2000)
    held = torch.stack([held[i] for i in range(len(held))])
    m_held = random_msg(len(held), k, generator = torch.Generator().manual_seed(1))

    E, W = Encoder(k, alpha = ALPHA), Extractor(k)
    opt = torch.optim.Adam(list(E.parameters()) + list(W.parameters()), lr = 1e-3)

    step, t0 = 0, time.time()
    while step < steps:
        for x in loader:
            E.train(); W.train()
            m = random_msg(x.size(0), k)
            loss = msg_loss(W(E(x, m)), m)
            opt.zero_grad(set_to_none = True)
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0 or step == steps:
                E.eval(); W.eval()
                with torch.no_grad():
                    xw = E(held, m_held)
                    acc, p = bit_acc(W(xw), m_held).item(), psnr(xw, held).item()
                    clean = bit_acc(W(held), m_held).item()
                print(f"  step {step} loss {loss.item():.3f} held-out bit acc {acc:.3f} clean acc {clean:.3f} PSNR {p:.1f} dB ({time.time()-t0:.0f} s)")
            if step == steps:
                break

    assert acc >= min_acc, f"held-out bit accuracy {acc:.3f} < {min_acc}"
    assert abs(clean - 0.5) < 0.1, f"clean-image accuracy {clean:.3f} is far from chance"
    assert p >= 10*math.log10(4/ALPHA**2) - 1e-3


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--learn', action = 'store_true')
    p.add_argument("--data_root", default = 'data/img_align_celeba')
    # CPU defaults (about 10 min): k = 8 learns faster than 32, 400 steps give about 0.65 bit accuracy
    p.add_argument('--steps', type = int, default = 400)
    p.add_argument('--bs', type = int, default = 16)
    p.add_argument('--k', type = int, default = 8)
    p.add_argument('--min_acc', type = float, default = 0.6)
    args = p.parse_args()

    torch.manual_seed(0)
    tests = [(t.__name__, t) for t in [test_shapes, test_distortion_bound, test_message_changes_output, test_gradients, test_helpers]]
    if args.learn:
        tests.append(('test_learns', lambda: test_learns(args.data_root, args.steps, args.bs, args.k, args.min_acc)))

    E, W = Encoder(K), Extractor(K)
    print(f"Encoder {sum(p.numel() for p in E.parameters())/1e6:.2f} M params, Extractor {sum(p.numel() for p in W.parameters())/1e6:.2f} M params")
    failed = 0
    for name, t in tests:
        try:
            t()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    print("="*50)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(failed > 0)


if __name__ == '__main__':
    main()
