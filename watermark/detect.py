from math import comb
from pathlib import Path
import torch

# Detection as a hypothesis test (step 5). H0: x is not watermarked, its decoded bits are i.i.d. Bernoulli(1/2),
# so the number M of bits matching the key m* follows Bin(k, 1/2). x is flagged when M >= tau.


# P(M >= tau | H0), exact
def fpr(tau, k):
    return sum(comb(k, i) for i in range(tau, k + 1)) / 2**k


# smallest tau with FPR(tau) <= target
def threshold(k, target):
    return next(t for t in range(k + 2) if fpr(t, k) <= target)


def binom_pmf(k):
    return torch.tensor([comb(k, i) for i in range(k + 1)], dtype = torch.float64) / 2**k


# M(x) = number of decoded bits 1[logit_i > 0] equal to the key
def matches(logits, key):
    return ((logits > 0).float() == key).sum(1)


# the secret key m*: drawn once with a fixed seed, then always reloaded from disk
def load_or_make_key(path, k, seed=0):
    path = Path(path)
    if path.exists():
        key = torch.load(path)
        assert key.numel() == k, f"{path} holds a {key.numel()}-bit key, not {k}"
        return key
    key = torch.randint(0, 2, (k,), generator = torch.Generator().manual_seed(seed)).float()
    torch.save(key, path)
    return key
