from math import comb
from pathlib import Path
import torch

# detection test: an image that is not watermarked gives random bits, so the number M of bits
# matching the key follows a binomial law Bin(k, 1/2), and the image is flagged when M >= tau


# false positive rate P(M >= tau) for an image that is not watermarked (exact)
def fpr(tau, k):
    return sum(comb(k, i) for i in range(tau, k + 1)) / 2**k


# smallest tau whose false positive rate is below target
def threshold(k, target):
    return next(t for t in range(k + 2) if fpr(t, k) <= target)


# probabilities of the binomial law Bin(k, 1/2)
def binom_pmf(k):
    return torch.tensor([comb(k, i) for i in range(k + 1)], dtype = torch.float64) / 2**k


# M = number of decoded bits (logit > 0) equal to the key
def matches(logits, key):
    return ((logits > 0).float() == key).sum(1)


# secret key, drawn once with a fixed seed and then always reloaded from disk
def load_or_make_key(path, k, seed=0):
    path = Path(path)
    if path.exists():
        key = torch.load(path)
        assert key.numel() == k, f"{path} holds a {key.numel()}-bit key, not {k}"
        return key
    key = torch.randint(0, 2, (k,), generator = torch.Generator().manual_seed(seed)).float()
    torch.save(key, path)
    return key
