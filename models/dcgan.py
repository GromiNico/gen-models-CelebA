import torch.nn as nn


# Generator class, takes a noise vector of size z=100 and creates a 3x64x64 image
class G(nn.Module):
    def __init__(self, z=100, c=64):
        super().__init__()
        
        def up(i, o, s=2, p=1):
            return [nn.ConvTranspose2d(i,o,4,s,p,bias = False), nn.BatchNorm2d(o), nn.ReLU(True)]
        
        self.net = nn.Sequential(*up(z, 8*c, 1,0), *up(8*c, 4*c), *up(4*c, 2*c), *up(2*c,c), nn.ConvTranspose2d(c, 3,4,2,1, bias = False), nn.Tanh())
        
    def forward(self, z):
        return self.net(z.view(z.size(0),-1,1,1))
        
class D(nn.Module):
    
    def __init__(self, c=64, norm = 'bn'):
        super().__init__()
        
        N = (lambda o: nn.BatchNorm2d(o)) if norm =='bn' else (lambda o: nn.GroupNorm(1,o))
        
        def down(i, o, n=True):
            return [nn.Conv2d(i,o,4,2,1, bias = False)] + ([N(o)] if n else []) + [nn.LeakyReLU(0.2, True)]
        
        self.net = nn.Sequential(*down(3, c, False), *down(c, 2*c), *down(2*c, 4*c), *down(4*c,8*c), nn.Conv2d(8*c, 1,4,1,0, bias = False))
        
    def forward(self, x):
        return self.net(x).view(-1)
        
        