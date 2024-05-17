"""
DistMult implementation.
"""

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def forward(self, sample, mode="train"):
        # sample.to(DEVICE)
        positive_sample, negative_samples = sample

        head, relation, tail = positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2]  # each [B]
        head_emb = self.entity_emb(head).to(DEVICE)
        relation_emb = self.relation_emb(relation).to(DEVICE)
        tail_emb = self.entity_emb(tail).to(DEVICE)

        # print("relation and tail", relation.shape, tail.shape)

        true_score = self._get_score(head_emb, relation_emb, tail_emb)

        negative_heads, negative_tails = negative_samples  # ensure it is a tuple
        # print("neg head and neg tail", negative_heads.shape, negative_tails.shape)

        if mode == "test":
            # negative_heads, negative_tails of size [B, K]
            neg_batch, neg_num_samples = negative_heads.shape
            relation = relation.repeat(neg_num_samples, 1).view(neg_batch, neg_num_samples)
            tail = tail.repeat(neg_num_samples, 1).view(neg_batch, neg_num_samples)

            relation_emb = self.relation_emb(relation)
            tail_emb = self.entity_emb(tail)

            # print("neg relation and tail", relation.shape, tail.shape)
            # print("neg relation emb and tail emb", relation_emb.shape, tail_emb.shape)

            head = self.entity_emb(negative_heads.view(neg_batch, neg_num_samples))
            # print("head emb", head.shape)

            tail = self.entity_emb(negative_tails.view(neg_batch, neg_num_samples))
            # print("tail emb", tail.shape)

        else:
            # train mode, negative_heads and negative_tails of shape [B]
            head = self.entity_emb(negative_heads.view(-1))
            tail = self.entity_emb(negative_tails.view(-1))

        head_pred_score = self._get_score(head, relation_emb, tail_emb)
        tail_pred_score = self._get_score(head, relation_emb, tail)

        return true_score, head_pred_score, tail_pred_score

    def _get_score(self, head, relation, tail):
        score = (head * relation * tail).sum(dim=-1)    # shape [B] or [B, K] depending on where it is called
        return score
