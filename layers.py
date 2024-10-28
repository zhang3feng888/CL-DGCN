import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from modules import MessageNorm, MLP
from ogb.graphproppred.mol_encoder import BondEncoder

class GENConv(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        beta=1.0,
        msg_norm=False,
        learn_msg_scale=False,
        mlp_layers=1,
        eps=1e-7,
    ):
        super(GENConv, self).__init__()

        self.eps = eps
        self.beta = beta

        channels = [in_dim]
        for _ in range(mlp_layers - 1):
            channels.append(in_dim * 2)
        channels.append(out_dim)

        self.mlp = MLP(channels)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.edge_encoder = BondEncoder(in_dim)

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata["h"] = node_feats.transpose(0, 3).transpose(1, 3)
            g.edata["h"] = g.edata["feat"]
            g.apply_edges(fn.u_add_e("h", "h", "m"))

            g.edata["m"] = F.relu(g.edata["m"]) + self.eps
            g.edata["a"] = edge_softmax(g, g.edata["m"] * self.beta)
            g.update_all(
                lambda edge: {"x": edge.data["m"] * edge.data["a"]},
                fn.sum("x", "m"),
            )

            if self.msg_norm is not None:
                g.ndata["m"] = self.msg_norm(node_feats, g.ndata["m"])

            feats = node_feats.transpose(0, 3).transpose(1, 3) + g.ndata["m"]
            return self.mlp(feats).transpose(1, 3).transpose(0, 3)
