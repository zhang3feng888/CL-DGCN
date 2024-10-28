import math
import torch
import torch as th
from aug import aug
from dgl.nn.pytorch.glob import AvgPooling
from ogb.graphproppred.mol_encoder import AtomEncoder
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from modules import MessageNorm, MLP
from layers import GENConv
from ogb.graphproppred.mol_encoder import BondEncoder


class DeepGCN(nn.Module):
    def __init__(
        self,
        hid_dim,
        out_dim,
        DEEP_num_layers,
        dropout=0.0,
    ):
        super(DeepGCN, self).__init__()

        self.DEEP_num_layers = DEEP_num_layers
        self.dropout = dropout
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.DEEP_num_layers):
            conv = GENConv(
                in_dim=hid_dim,
                out_dim=hid_dim,
            )

            self.gcns.append(conv)
            self.norms.append(nn.BatchNorm2d(hid_dim, affine=True))

        self.node_encoder = AtomEncoder(hid_dim)
        self.pooling = AvgPooling()
        self.output = nn.Linear(hid_dim, out_dim)

    def forward(self, g, edge_feats, node_feats=None):
        with g.local_scope():
            hv = node_feats
            he = edge_feats

            for layer in range(self.DEEP_num_layers):
                hv1 = self.norms[layer](hv)
                hv1 = F.relu(hv1)
                hv1 = F.dropout(hv1, p=self.dropout, training=self.training)
                hv = self.gcns[layer](g, hv1, he) + hv

            h_g = self.pooling(g, hv.transpose(0, 3))
            h_g = h_g.transpose(0, 3) * hv

            return self.output(h_g.transpose(1, 3)).transpose(1, 3)

class TemporalConvLayer(nn.Module):
    def __init__(self, c_in, c_out, dia=1):
        super(TemporalConvLayer, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(
            c_in, c_out, (2, 1), 1, dilation=dia, padding=(0, 0)
        )

    def forward(self, x):
        return torch.relu(self.conv(x))

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, in_dim)

    def forward(self, x):
        z = F.elu(self.fc1(x))
        return self.fc2(z)

class CL(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, DEEP_num_layers, temp):
        super(CL, self).__init__()
        self.encoder = DeepGCN(
            in_dim,
            hid_dim,
            DEEP_num_layers,
            dropout=0.2,
        )
        self.temp = temp
        self.proj = MLP(hid_dim, out_dim)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        s = th.mm(z1, z2.t())
        return s

    def get_loss(self, z1, z2):
        x = th.zeros_like(z1)
        z1 = z1[0, :, :, 0]
        z2 = z2[0, :, :, 0]
        f = lambda x: th.exp(x / self.temp)

        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -th.log(between_sim.diag() / x1)

        x = th.zeros_like(x)
        x[0, :, 0, 0] = loss
        x = th.zeros_like(x, device=z1.device)

        return x

    def forward(self, graph1, graph2, feat1, feat2):
        h1 = self.encoder(graph1, feat1, feat1).transpose(1, 3)
        h2 = self.encoder(graph2, feat2, feat2).transpose(1, 3)

        z1 = self.proj(h1)
        z2 = self.proj(h2)

        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)

        ret = z1 + z2 + l1 + l2
        ret = ret.transpose(1, 3)
        return ret

class SpatioConvLayer(nn.Module):
    def __init__(self, c, Lk, DEEP_num_layers):
        super(SpatioConvLayer, self).__init__()
        self.Lk = Lk
        self.cl = CL(c, c, c, DEEP_num_layers, 1.0)

    def init(self):
        stdv = 1.0 / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        g1, x1 = aug(self.Lk, x, 0.2, 0.2)
        g2, x2 = aug(self.Lk, x, 0.2, 0.2)

        output = self.cl(g1, g2, x1, x2)
        return torch.relu(output)

class FullyConvLayer(nn.Module):
    def __init__(self, c):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)

class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation=1, padding=(0, 0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation=1, padding=(0, 0))
        self.fc = FullyConvLayer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)

class STGCN_WAVE(nn.Module):
    def __init__(
        self, c, T, n, Lk, p, DEEP_num_layers, device, control_str="TST"
    ):
        super(STGCN_WAVE, self).__init__()
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.layers = nn.ModuleList([])
        cnt = 0
        diapower = 0
        for i in range(self.num_layers):
            i_layer = control_str[i]
            if i_layer == "T":
                self.layers.append(
                    TemporalConvLayer(c[cnt], c[cnt + 1], dia=2**diapower)
                )
                diapower += 1
                cnt += 1
            if i_layer == "S":
                self.layers.append(SpatioConvLayer(c[cnt], Lk, DEEP_num_layers))
        self.output = OutputLayer(c[cnt], T + 1 - 2 ** (diapower), n)
        for layer in self.layers:
            layer = layer.to(device)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return self.output(x)
