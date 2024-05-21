"""
DistMult implementation.
"""

import torch
import torch.nn as nn


class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim):
        super(DistMult, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim

        self.entity_emb = nn.Embedding(self.num_entities,
                                       self.embed_dim)  # [n, d], n = num of entities, d = embedding dim
        self.relation_emb = nn.Embedding(self.num_relations,
                                         self.embed_dim)  # [n, d], d = diagonal elements of dxd matrix for relation r

        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, sample, device):
        positive_samples, negative_samples = sample

        head, relation, tail = positive_samples[:, 0].to(device), positive_samples[:, 1].to(device), positive_samples[:, 2].to(device)  # each [N]

        # each of shape [N, 1, d]
        head_emb = self.entity_emb(head).unsqueeze(1)
        relation_emb = self.relation_emb(relation).unsqueeze(1)
        tail_emb = self.entity_emb(tail).unsqueeze(1)

        positive_score = DistMult.get_score(head_emb, relation_emb, tail_emb)   # [N, 1]

        # negative_samples = (neg_heads, neg_tails, (maybe) head_filter, (maybe) tail_filter)
        # each negative_* of shape [N, num_neg_sample]
        negative_heads, negative_tails = negative_samples[0].to(device), negative_samples[1].to(device)

        batch, neg_sample_size = negative_heads.size(0), negative_heads.size(1)
        head = self.entity_emb(negative_heads.view(-1)).view(batch, neg_sample_size, -1)
        negative_head_score = DistMult.get_score(head, relation_emb, tail_emb)

        batch, neg_sample_size = negative_tails.size(0), negative_tails.size(1)
        tail = self.entity_emb(negative_tails.view(-1)).view(batch, neg_sample_size, -1)
        negative_tail_score = DistMult.get_score(head_emb, relation_emb, tail)

        return positive_score, negative_head_score, negative_tail_score

    @staticmethod
    def get_score(head, relation, tail):
        score = (head * relation * tail).sum(dim=2)
        return score
