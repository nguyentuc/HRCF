import numpy as np
import torch
import torch.nn as nn

import manifolds
import models.encoders as encoders
from utils.helper import default_device
import torch.nn.functional as F
from layers.layers import FermiDiracDecoder

class HRCFModel(nn.Module):

    def __init__(self, users_items, args):
        super(HRCFModel, self).__init__()

        self.c = torch.tensor([args.c]).to(default_device())
        self.manifold = getattr(manifolds, "Hyperboloid")()
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, "HRCF")(self.c, args)

        self.num_users, self.num_items = users_items
        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.num_layers = args.num_layers
        self.args = args
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.embedding = nn.Embedding(num_embeddings=self.num_users + self.num_items,
                                      embedding_dim=args.embedding_dim).to(default_device())

        self.embedding.state_dict()['weight'].uniform_(-args.scale, args.scale)
        self.embedding.weight = nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], self.c))

        self.embedding.weight = manifolds.ManifoldParameter(self.embedding.weight, True, self.manifold, self.c)

        self.alpha = args.alpha

    def encode(self, adj):
        x = self.embedding.weight
        if torch.cuda.is_available():
           adj = adj.to(default_device())
           x = x.to(default_device())
        h = self.encoder.encode(x, adj)
        return h

    def decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return [sqdist, probs]


    def geometric_regularizer(self, embeddings):
        embeddings_tan = self.manifold.logmap0(embeddings, c=1.0)
        item_embeddings = embeddings_tan[self.num_users:]
        item_mean_norm = ((1e-6 + item_embeddings.pow(2).sum(dim=1)).mean()).sqrt()
        return 1.0 / item_mean_norm


    def ranking_loss(self, pos_sqdist, neg_sqdist, ):
        loss = pos_sqdist - neg_sqdist + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss

    def compute_loss(self, embeddings, triples):
        train_edges = triples[:, [0, 1]]

        sampled_false_edges_list = [triples[:, [0, 2 + i]] for i in range(self.args.num_neg)]

        pos = self.decode(embeddings, train_edges)
        pos_sqdist, pos_probs = pos
        neg = self.decode(embeddings, sampled_false_edges_list[0])
        neg_sqdist, neg_probs = neg

        ranking_loss = self.ranking_loss(pos_sqdist, neg_sqdist)
        gr_loss = self.geometric_regularizer(embeddings)

        return ranking_loss + self.alpha*gr_loss

    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_in = h[i, :]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)
            emb_out = h[np.arange(num_users, num_users + num_items), :]
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)

            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])
        return probs_matrix
