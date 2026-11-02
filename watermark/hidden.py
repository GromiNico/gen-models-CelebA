import torch
import torch.nn as nn
import torch.nn.functional as F

# HiDDeN encoder E and extractor W (Zhu et al., ECCV 2018), with the residual tanh/alpha output of
# Stable Signature (Fernandez et al., ICCV 2023). Images in [-1, 1], messages m in {0, 1}^k.


# Conv 3x3 (stride 1, padding 1) -> BatchNorm -> ReLU, spatial size unchanged
def block(i, o):
    return [nn.Conv2d(i, o, 3, 1, 1, bias = False), nn.BatchNorm2d(o), nn.ReLU(True)]


class Encoder(nn.Module):
    # x: (B,3,H,W), m: (B,k) -> x_w = clamp(x + alpha*tanh(delta), -1, 1)
    def __init__(self, k=32, c=64, alpha=0.05):
        super().__init__()
        self.k, self.alpha = k, alpha
        self.feat = nn.Sequential(*block(3, c), *block(c, c), *block(c, c), *block(c, c))
        self.out = nn.Sequential(*block(c + k + 3, c), nn.Conv2d(c, 3, 1))

    def forward(self, x, m):
        # bits {0,1} -> {-1,+1}, replicated over every pixel: the (B,k,H,W) "message volume"
        mv = (2*m.float() - 1).view(m.size(0), self.k, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        delta = self.out(torch.cat([self.feat(x), mv, x], 1))
        # the distortion is bounded by alpha per pixel, so no image loss is needed
        return (x + self.alpha*torch.tanh(delta)).clamp(-1, 1)


class Extractor(nn.Module):
    # x: (B,3,H,W) of any size -> logits (B,k), decoded bit i is 1[logit_i > 0]
    def __init__(self, k=32, c=64):
        super().__init__()
        layers = block(3, c)
        for _ in range(6):
            layers += block(c, c)
        # global average pooling makes W independent of the input size (crops)
        self.net = nn.Sequential(*layers, *block(c, k), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(k, k)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


def random_msg(b, k=32, device='cpu', generator=None):
    return torch.randint(0, 2, (b, k), device = device, generator = generator).float()


# (1/k) sum_i BCE(sigmoid(logit_i), m_i), averaged over the batch
def msg_loss(logits, m):
    return F.binary_cross_entropy_with_logits(logits, m.float())


def bit_acc(logits, m):
    return ((logits > 0).float() == m).float().mean()


# pixels in [-1, 1]: peak-to-peak range 2, so PSNR = 10 log10(4 / MSE)
def psnr(x, y):
    mse = ((x - y)**2).flatten(1).mean(1)
    return (10*torch.log10(4/mse)).mean()
