import datetime
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from classifier.base import BASE

class Classifier(BASE):

    def __init__(self, ebd_dim, args):
        super(Classifier, self).__init__(args)
        self.args = args
        self.ebd_dim = ebd_dim

        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

        self.reference = nn.Linear(self.ebd_dim, self.args.way * self.args.shot, bias=True)
        nn.init.kaiming_normal_(self.reference.weight, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.reference.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def get_distance(self, XS, XQ):

        dot_product = XQ.mm(XS.t()) / 0.1

        return dot_product

    def _label2onehot(self, Y):

        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    def get_sorted(self, XS, YS, XQ, YQ, XS_aug=None, XQ_aug=None):

        sorted_YS, indices_YS = torch.sort(YS)
        sorted_YQ, indices_YQ = torch.sort(YQ)

        XS = XS[indices_YS]
        XQ = XQ[indices_YQ]

        if XS_aug is not None:
            XS_aug = XS_aug[indices_YS]
            XQ_aug = XQ_aug[indices_YQ]
            return XS, sorted_YS, XQ, sorted_YQ, XS_aug, XQ_aug

        return XS, sorted_YS, XQ, sorted_YQ

    def _compute_mean(self, XS):

        mean_ = []
        for i in range(self.args.way):
            mean_.append(torch.mean(
                XS[i * self.args.shot:(i + 1) * self.args.shot], dim=0,
                keepdim=True))

        mean_ = torch.cat(mean_, dim=0)

        return mean_

    def Transformation_Matrix(self, XS):
        C = XS
        eps = 1e-6
        R = self.reference.weight

        power_R = ((R * R).sum(dim=1, keepdim=True)).sqrt()
        R = R / (power_R + eps)

        power_C = ((C * C).sum(dim=1, keepdim=True)).sqrt()
        C = C / (power_C + eps)

        P = torch.matmul(torch.pinverse(C), R)
        P = P.permute(1, 0) # [d, d]
        return P

    def forward(self, XS, YS, XQ, YQ):

        YS, YQ = self.reidx_y(YS, YQ)
        XS, YS, XQ, YQ = self.get_sorted(XS, YS, XQ, YQ)
        prototype = self._compute_mean(XS)

        P = self.Transformation_Matrix(XS)
        weight = P.view(P.size(0), P.size(1), 1)
        XS_transformed = F.conv1d(XS.squeeze(0).unsqueeze(2), weight).squeeze(2)
        XQ_transformed = F.conv1d(XQ.squeeze(0).unsqueeze(2), weight).squeeze(2)
        prototype_transformed = F.conv1d(prototype.squeeze(0).unsqueeze(2), weight).squeeze(2)

        similar = self.get_distance(XS_transformed, XQ_transformed)
        YS_onehot = self._label2onehot(YS)
        pred = similar.mm(YS_onehot.float())

        discriminative_loss = 0.0

        for j in range(self.args.way):
            for k in range(self.args.way):
                if j != k:
                    sim = -self._compute_cos(prototype_transformed[j].unsqueeze(0), prototype_transformed[k].unsqueeze(0))
                    discriminative_loss = discriminative_loss + sim

        loss = F.cross_entropy(pred, YQ) + 0.5 * discriminative_loss

        acc = BASE.compute_acc(pred, YQ)

        return acc, loss