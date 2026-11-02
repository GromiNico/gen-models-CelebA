# Tests for the attack layer (step 2). Run from the repo root:
#   python -m watermark.test_attacks
#   python -m watermark.test_attacks --grid runs/watermark/attacks.png   # + every eval attack on 8 CelebA images
import argparse
import random
import torch
from watermark.attacks import (TRAIN_ATTACKS, EVAL_ATTACKS, EVAL_2DAY, train_attack, eval_attack, attack,
                               jpeg, brightness, quantize, to_uint8)
from watermark.hidden import Extractor


def smooth_images(b=4, s=64):
    # smooth random images, closer to photos than white noise, so JPEG behaves as on real data
    x = torch.rand(b, 3, 8, 8)*1.6 - 0.8
    return torch.nn.functional.interpolate(x, size = (s, s), mode = 'bicubic', align_corners = False).clamp(-1, 1)


def test_train_attacks_shape_range_grad():
    for name in TRAIN_ATTACKS:
        x = smooth_images().requires_grad_(True)
        y = train_attack(x, name)
        assert y.shape == x.shape, name
        assert y.min() >= -1 - 1e-6 and y.max() <= 1 + 1e-6, name
        y.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0, f"{name}: no gradient reaches the input"


def test_jpeg_straight_through():
    x = smooth_images().requires_grad_(True)
    random.seed(0)
    y = train_attack(x, 'jpeg')
    random.seed(0)
    q = random.randint(50, 95)
    # forward = real JPEG at the sampled quality, backward = identity
    assert torch.allclose(y, jpeg(x, q), atol = 1e-6)
    assert not torch.allclose(y, x, atol = 1e-3)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_jpeg_kornia_gradient():
    try:
        import kornia  # noqa: F401
    except ImportError:
        print("  (kornia not installed, test skipped)")
        return
    x = smooth_images().requires_grad_(True)
    random.seed(0)
    y = train_attack(x, 'jpeg', jpeg_grad = 'kornia')
    random.seed(0)
    q = random.randint(50, 95)
    # same forward as the straight-through version (the real JPEG), but a gradient that is not the identity
    assert torch.allclose(y, jpeg(x, q), atol = 1e-6)
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert not torch.allclose(x.grad, torch.ones_like(x))


def test_jpeg_quality_order():
    x = quantize(smooth_images(8))
    err = {q: ((jpeg(x, q) - x)**2).mean().item() for q in [90, 70, 50]}
    assert err[90] < err[70] < err[50], err


def test_eval_attacks_uint8_grid():
    x = smooth_images()
    for name in EVAL_ATTACKS:
        y = eval_attack(x, name)
        u = (y + 1)*127.5
        assert torch.allclose(u, u.round(), atol = 1e-4), f"{name}: not on the uint8 grid"
        assert not y.requires_grad


def test_eval_crop_sizes():
    x = smooth_images()
    # side scaled by sqrt(p): 64*sqrt(0.5) = 45.25 -> 45
    for p, side in [(0.75, 55), (0.5, 45), (0.25, 32), (0.1, 20)]:
        assert eval_attack(x, f'crop{p}').shape[-2:] == (side, side), p
    assert eval_attack(x, 'combined').shape[-2:] == (45, 45)
    # center crop: the kept pixels are the original ones
    assert torch.equal(eval_attack(x, 'crop0.25'), quantize(x)[..., 16:48, 16:48])


def test_eval_noise_std():
    x = torch.zeros(16, 3, 64, 64)  # mid-gray, far from the clamp
    for s in [0.02, 0.05, 0.1]:
        std = (eval_attack(x, f'noise{s}') - x).std().item()
        assert abs(std - s) < 0.1*s + 0.005, (s, std)


def test_brightness():
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0]).view(1, 1, 1, 4)
    # [0,1] values 0, 0.5, 0.75, 1 times 1.5 -> 0, 0.75, 1 (clamped), 1
    assert torch.allclose(brightness(x, 1.5).flatten(), torch.tensor([-1.0, 0.5, 1.0, 1.0]))
    assert torch.allclose(brightness(x, 1.0), x)
    x = quantize(smooth_images())
    assert eval_attack(x, 'bright2.0').mean() > eval_attack(x, 'bright1.5').mean() > x.mean()


def test_none_is_quantization():
    x = smooth_images()
    assert torch.equal(eval_attack(x, 'none'), quantize(x))
    assert torch.equal(to_uint8(quantize(x)), to_uint8(x))


def test_attack_random_pick():
    random.seed(0)
    x = smooth_images(2)
    for _ in range(20):
        assert attack(x, train = True).shape == x.shape
        attack(x, train = False)
    assert torch.equal(attack(x, train = True, names = ['identity']), x)
    assert set(EVAL_2DAY) <= set(EVAL_ATTACKS)


def test_extractor_reads_every_eval_attack():
    W = Extractor(32).eval()
    x = smooth_images()
    with torch.no_grad():
        for name in EVAL_ATTACKS:
            assert W(eval_attack(x, name)).shape == (4, 32), name


def save_grid(path, data_root):
    import torchvision.utils as vu
    from pathlib import Path
    from data import CelebA64
    ds = CelebA64(data_root, n = 8)
    x = torch.stack([ds[i] for i in range(8)])
    rows = []
    for name in EVAL_ATTACKS:
        y = eval_attack(x, name)
        # crops are smaller than 64x64: paste them centred on a gray canvas
        canvas = torch.zeros_like(x)
        h, w = y.shape[-2:]
        canvas[..., (64 - h)//2:(64 - h)//2 + h, (64 - w)//2:(64 - w)//2 + w] = y
        rows.append(canvas)
    Path(path).parent.mkdir(parents = True, exist_ok = True)
    vu.save_image(torch.cat(rows)*0.5 + 0.5, path, nrow = 8, padding = 2)
    print(f"rows (top to bottom): {', '.join(EVAL_ATTACKS)}")
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--grid', default = None)
    p.add_argument("--data_root", default = 'data/img_align_celeba')
    args = p.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    tests = [test_train_attacks_shape_range_grad, test_jpeg_straight_through, test_jpeg_kornia_gradient, test_jpeg_quality_order,
             test_eval_attacks_uint8_grid, test_eval_crop_sizes, test_eval_noise_std, test_brightness,
             test_none_is_quantization, test_attack_random_pick, test_extractor_reads_every_eval_attack]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print("="*50)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    if args.grid:
        save_grid(args.grid, args.data_root)
    raise SystemExit(failed > 0)


if __name__ == '__main__':
    main()
