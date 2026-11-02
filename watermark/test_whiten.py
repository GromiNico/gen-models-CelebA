# Tests for the detection test (detect.py) and the whitening layer (whiten.py, hidden.Whitened). Run from the repo root:
#   python -m watermark.test_whiten
import tempfile
from pathlib import Path
import torch
from watermark.detect import fpr, threshold, binom_pmf, matches, load_or_make_key
from watermark.hidden import Extractor, Whitened, load_whitened
from watermark.whiten import fit_whitening, null_check


def test_thresholds():
    # the table of the guide (step 5): smallest tau with FPR <= target, and its exact FPR
    for k, target, tau, exact in [(32, 1e-3, 26, 2.7e-4), (32, 1e-6, 30, 1.2e-7), (48, 1e-3, 36, 3.6e-4), (48, 1e-6, 41, 3.1e-7)]:
        assert threshold(k, target) == tau, (k, target, threshold(k, target))
        assert abs(fpr(tau, k) - exact)/exact < 0.05, (k, tau, fpr(tau, k))
        assert fpr(tau - 1, k) > target
    assert fpr(0, 32) == 1 and abs(binom_pmf(32).sum().item() - 1) < 1e-12


def test_matches_and_key():
    key = torch.tensor([1., 0., 1., 1.])
    logits = torch.tensor([[3., -1., 2., 5.], [-3., 1., -2., -5.], [1., 1., 1., 1.]])
    assert matches(logits, key).tolist() == [4, 0, 3]
    with tempfile.TemporaryDirectory() as d:
        k1 = load_or_make_key(Path(d)/'key.pt', 32, seed = 0)
        k2 = load_or_make_key(Path(d)/'key.pt', 32, seed = 99)  # reloaded from disk, the seed is ignored
        assert torch.equal(k1, k2) and set(k1.unique().tolist()) <= {0., 1.}


def test_fit_whitening():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(8, 8, generator = g)
    y = torch.randn(20000, 8, generator = g) @ A + 3  # correlated and biased
    mu, P = fit_whitening(y)
    yw = (y - mu) @ P.T
    assert torch.allclose(P, P.T, atol = 1e-5)
    assert yw.mean(0).abs().max() < 1e-3
    assert (torch.cov(yw.T) - torch.eye(8)).abs().max() < 1e-3


def test_whitened_module_roundtrip():
    W = Extractor(8).eval()
    mu, P = torch.randn(8), torch.randn(8, 8)
    x = torch.rand(3, 3, 64, 64)*2 - 1
    Wp = Whitened(W, mu, P).eval()
    with torch.no_grad():
        assert torch.allclose(Wp(x), (W(x) - mu) @ P.T, atol = 1e-5)
        with tempfile.TemporaryDirectory() as d:
            torch.save({'W': W.state_dict(), 'mu': mu, 'P': P, 'cfg': {'k': 8, 'alpha': 0.05}}, Path(d)/'w.pt')
            Wl = load_whitened(Path(d)/'w.pt')
            assert torch.allclose(Wl(x), Wp(x), atol = 1e-5)
    assert not any(q.requires_grad for q in Wl.parameters())


def test_null_check():
    g = torch.Generator().manual_seed(0)
    k, key = 32, torch.randint(0, 2, (32,), generator = g).float()
    # fair independent bits: M ~ Bin(32, 1/2), mean 16, variance 8
    r, _ = null_check(torch.randn(20000, k, generator = g), key, k)
    assert abs(r['mean of M (H0: 16)'] - 16) < 0.1 and abs(r['variance of M (H0: 8)'] - 8) < 0.4
    assert r['chi-square p-value'] > 1e-3
    # one shared factor makes the bits correlated: heavier tails, M is not binomial
    y = torch.randn(20000, k, generator = g) + 1.5*torch.randn(20000, 1, generator = g)*(2*key - 1)
    r, _ = null_check(y, key, k)
    assert r['variance of M (H0: 8)'] > 16 and r['chi-square p-value'] < 1e-6


def main():
    tests = [test_thresholds, test_matches_and_key, test_fit_whitening, test_whitened_module_roundtrip, test_null_check]
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
    raise SystemExit(failed > 0)


if __name__ == '__main__':
    main()
