import torch
import torch.nn as nn
import torch.nn.functional as F


# conv 3x3 -> batchnorm -> relu, keeps the spatial size
def block(i, o):
    return [nn.Conv2d(i, o, 3, 1, 1, bias = False), nn.BatchNorm2d(o), nn.ReLU(True)]


# Encoder class (HiDDeN), takes images in [-1,1] and messages of k bits and hides each message in its image
class Encoder(nn.Module):
    def __init__(self, k=32, c=64, alpha=0.05):
        super().__init__()
        self.k, self.alpha = k, alpha
        self.feat = nn.Sequential(*block(3, c), *block(c, c), *block(c, c), *block(c, c))
        self.out = nn.Sequential(*block(c + k + 3, c), nn.Conv2d(c, 3, 1))

    def forward(self, x, m):
        # bits 0/1 become -1/+1 and are copied on every pixel, it gives a k x H x W message volume
        mv = (2*m.float() - 1).view(m.size(0), self.k, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        delta = self.out(torch.cat([self.feat(x), mv, x], 1))
        # each pixel moves by at most alpha, so no image loss is needed (as in Stable Signature)
        return (x + self.alpha*torch.tanh(delta)).clamp(-1, 1)


# Extractor class (HiDDeN), reads k logits from an image, bit i is 1 when logit i > 0
class Extractor(nn.Module):
    def __init__(self, k=32, c=64):
        super().__init__()
        layers = block(3, c)
        for _ in range(6):
            layers += block(c, c)
        # global average pooling so that any image size works (crops)
        self.net = nn.Sequential(*layers, *block(c, k), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(k, k)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


# whitened extractor W' = P(W - mu), mu and P are fitted by whiten.py
# on clean images its outputs are centered, of variance 1 and uncorrelated
# P is symmetric so bit i of W' still corresponds to bit i of the message
class Whitened(nn.Module):
    def __init__(self, W, mu, P):
        super().__init__()
        self.W = W
        self.register_buffer('mu', mu)
        self.register_buffer('P', P)

    def forward(self, x):
        return (self.W(x) - self.mu) @ self.P.T


# loads the frozen W' saved by whiten.py
def load_whitened(path, device='cpu'):
    ck = torch.load(path, map_location = device)
    W = Extractor(ck['cfg']['k'])
    W.load_state_dict(ck['W'])
    return Whitened(W, ck['mu'], ck['P']).to(device).eval().requires_grad_(False)


# one random message of k bits per image
def random_msg(b, k=32, device='cpu', generator=None):
    return torch.randint(0, 2, (b, k), device = device, generator = generator).float()


# binary cross entropy between the logits and the message bits, averaged over the bits and the images
def msg_loss(logits, m):
    return F.binary_cross_entropy_with_logits(logits, m.float())


# fraction of decoded bits equal to the message
def bit_acc(logits, m):
    return ((logits > 0).float() == m).float().mean()


# pixels are in [-1,1] so the peak to peak range is 2 and PSNR = 10 log10(4/MSE)
def psnr(x, y):
    mse = ((x - y)**2).flatten(1).mean(1)
    return (10*torch.log10(4/mse)).mean()
