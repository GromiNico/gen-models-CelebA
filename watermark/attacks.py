import io
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Attack layer. Images are (B,3,H,W) in [-1, 1]. One transformation per batch, picked uniformly.
# Train-time attacks pass gradients (JPEG through a straight-through estimator); eval attacks are the
# real operations, on images rounded to uint8 before and after as a saved and edited file would be.


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1)*127.5).round().to(torch.uint8)

def from_uint8(u):
    return u.float()/127.5 - 1

def quantize(x):
    return from_uint8(to_uint8(x))


# real JPEG at quality q through PIL, on CPU, image by image (cheap at 64x64); no gradient
def jpeg(x, q):
    out = []
    for a in to_uint8(x.detach()).permute(0, 2, 3, 1).cpu().numpy():
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format = 'JPEG', quality = int(q))
        buf.seek(0)
        out.append(np.array(Image.open(buf).convert('RGB')))
    return from_uint8(torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2)).to(x.device)


# keep a fraction p of the area; resize=True scales the crop back to the input size
def crop(x, p, center=False, resize=True):
    H, W = x.shape[-2:]
    h, w = max(1, round(H*p**0.5)), max(1, round(W*p**0.5))
    i, j = ((H - h)//2, (W - w)//2) if center else (random.randint(0, H - h), random.randint(0, W - w))
    x = x[..., i:i+h, j:j+w]
    return F.interpolate(x, size = (H, W), mode = 'bilinear', align_corners = False) if resize else x


# down to a fraction s of the side, then back up (bilinear)
def resize(x, s):
    H, W = x.shape[-2:]
    y = F.interpolate(x, size = (max(1, round(H*s)), max(1, round(W*s))), mode = 'bilinear', align_corners = False, antialias = True)
    return F.interpolate(y, size = (H, W), mode = 'bilinear', align_corners = False)


# sigma in [-1, 1] pixel units
def noise(x, sigma):
    return (x + sigma*torch.randn_like(x)).clamp(-1, 1)


# multiply the [0, 1] image by c (as torchvision adjust_brightness), written in [-1, 1]: (x+1)*c - 1
def brightness(x, c):
    return ((x + 1)*c - 1).clamp(-1, 1)


TRAIN_ATTACKS = ['identity', 'jpeg', 'crop', 'resize', 'noise', 'brightness']

def train_attack(x, name):
    u = random.uniform
    if name == 'identity':
        return x
    if name == 'jpeg':
        # straight-through: forward is the real JPEG, backward is the identity
        return x + (jpeg(x, random.randint(50, 95)) - x).detach()
    if name == 'crop':
        return crop(x, u(0.3, 1))
    if name == 'resize':
        return resize(x, u(0.5, 1))
    if name == 'noise':
        return noise(x, u(0, 0.08))
    if name == 'brightness':
        return brightness(x, u(0.5, 2.0))
    raise ValueError(name)


# eval crops are center crops without resizing, as in Stable Signature; the extractor takes any size
EVAL_ATTACKS = {'none': lambda x: x}
EVAL_ATTACKS.update({f'jpeg{q}': (lambda x, q=q: jpeg(x, q)) for q in [90, 80, 70, 50]})
EVAL_ATTACKS.update({f'crop{p}': (lambda x, p=p: crop(x, p, center = True, resize = False)) for p in [0.75, 0.5, 0.25, 0.1]})
EVAL_ATTACKS.update({f'resize{s}': (lambda x, s=s: resize(x, s)) for s in [0.75, 0.5]})
EVAL_ATTACKS.update({f'noise{s}': (lambda x, s=s: noise(x, s)) for s in [0.02, 0.05, 0.1]})
EVAL_ATTACKS.update({f'bright{c}': (lambda x, c=c: brightness(x, c)) for c in [1.5, 2.0]})
EVAL_ATTACKS['combined'] = lambda x: jpeg(brightness(crop(x, 0.5, center = True, resize = False), 1.5), 80)

# the 5 attacks kept in the 2-day scope
EVAL_2DAY = ['jpeg80', 'jpeg50', 'crop0.5', 'noise0.05', 'combined']

def eval_attack(x, name):
    with torch.no_grad():
        return quantize(EVAL_ATTACKS[name](quantize(x)))


# one transformation for the whole batch, picked uniformly among names (identity included by default)
def attack(x, train=True, names=None):
    names = names or (TRAIN_ATTACKS if train else list(EVAL_ATTACKS))
    name = random.choice(names)
    return train_attack(x, name) if train else eval_attack(x, name)
