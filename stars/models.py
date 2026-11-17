import torch
from torch import nn


class SpectrumCNN(nn.Module):
    """CNN 1-D pour spectres. n_out = nombre de classes ou de cibles de régression."""

    def __init__(self, n_out, channels=(16, 32, 64), kernel=7, pool=4):
        super().__init__()
        layers, c_in = [], 1
        for c_out in channels:
            layers += [nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2), nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(pool)]
            c_in = c_out
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(16), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(c_in * 16, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n_out))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)))


def _mlp(dims):
    """Couches linéaires avec ReLU entre elles, sans activation en sortie."""
    layers = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(dims[-2], dims[-1]))


class Autoencoder(nn.Module):
    def __init__(self, n_pix, latent_dim=16, hidden=(512, 128)):
        super().__init__()
        self.encoder = _mlp([n_pix, *hidden, latent_dim])
        self.decoder = _mlp([latent_dim, *reversed(hidden), n_pix])

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x)
