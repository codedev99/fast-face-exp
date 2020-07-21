import torch
from torch import nn
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Taken from pytorch-spectral-normalization-gan repository by christiancosgrove
    See: https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class FFENet(nn.Module):
    def __init__(self, dropout=0.0):
        super(FFENet, self).__init__()
        self.layer1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, kernel_size = 5, stride = 1, padding=2, bias=False)),#48x48
            nn.LeakyReLU(0.2, inplace = True))
        self.layer2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 128, kernel_size = 5, stride = 2, padding=2, bias=False)),#24x24
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout(dropout))
        self.layer3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, kernel_size = 5, stride = 2, padding=2, bias=False)),#12x12
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout(dropout))
        self.layer4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(256, 512, kernel_size = 5, stride = 2, padding=2, bias=False)),#6x6
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True))
        self.layer5 = nn.Sequential(
            SpectralNorm(nn.Linear(512*6*6, 5)))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, 512*6*6)
        x = self.layer5(x)

        return x