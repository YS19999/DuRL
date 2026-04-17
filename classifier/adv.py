import torch
import torch.nn as nn

class Adversarial(nn.Module):

    def __init__(self, ebd_dim):
        super(Adversarial, self).__init__()

        self.mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(ebd_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

    def forward(self, embedd):

        return self.mlp(embedd)