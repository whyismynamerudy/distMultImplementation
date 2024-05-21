"""
Handles reading in Knowledge Graph and creates a dataloader for it.
"""

import torch
from torch.utils.data import Dataset
# import numpy as np


# from https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/codes/dataloader.py#L96
def _get_true_head_and_tail(triples):
    true_head = {}
    true_tail = {}

    for head, relation, tail in triples:
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    for relation, tail in true_head:
        true_head[(relation, tail)] = torch.as_tensor(list(set(true_head[(relation, tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = torch.as_tensor(list(set(true_tail[(head, relation)])))

    return true_head, true_tail


def load_dict(dict_file):
    """
    Load in dictionary from *.dict file.

    :param dict_file: file containing the dict.
    :return: dictionary with mapping from object to id.
    """
    my_dict = dict()
    with open(dict_file, "r") as f:
        for line in f:
            id, obj = line.strip().split('\t')
            my_dict[obj] = int(id)
    return my_dict


def load_triples(txt_file, entity2id, relation2id):
    """
    Load in relations/triples from *.txt file.

    :param txt_file: file containing relation.
    :return: list of all relations/triples.
    """
    relations = list()
    with open(txt_file, "r") as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            relations.append((entity2id[h], relation2id[r], entity2id[t]))
    return relations


class TrainDataLoader(Dataset):
    def __init__(self, triples, num_entities, negative_sample_size=128):
        self.triples = triples
        self.triples_set = set(triples)
        self.num_entities = num_entities
        self.negative_sample_size = negative_sample_size
        self.true_head, self.true_tail = _get_true_head_and_tail(triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        """
        Return positive sample corresponding to idx with negative sample.

        In the paper, they sample negative triplets (head, tail) for each positive sample.

        :param idx: Index of desired triplet.
        :return:
        """
        positive_sample = self.triples[idx]  # positive sample

        heads = torch.LongTensor(self.random_sampling(self, positive_sample, True))
        tails = torch.LongTensor(self.random_sampling(self, positive_sample, False))

        # print("positive", torch.LongTensor(positive_sample).shape)
        # print("heads", heads.shape)
        # print("tails", tails.shape)

        # neg_head, neg_tail = list(positive_sample[:]), list(positive_sample[:])
        # random_head = int(torch.randint(low=0, high=self.num_entities, size=(1,)).flatten())
        # random_tail = int(torch.randint(low=0, high=self.num_entities, size=(1,)).flatten())
        #
        # neg_head[0] = random_head
        # while set(neg_head) in self.triples_set:
        #     random_head = int(torch.randint(low=0, high=self.num_entities, size=(1,)).flatten())
        #     neg_head[0] = random_head
        #
        # neg_tail[2] = random_tail
        # while set(neg_tail) in self.triples_set:
        #     random_tail = int(torch.randint(low=0, high=self.num_entities, size=(1,)).flatten())
        #     neg_tail[2] = random_tail

        return torch.LongTensor(positive_sample), (heads, tails)

    @staticmethod
    def collate_fn(batch):
        # batch contains list of tuples (pos_sample, random_head, random_tail)

        pos_samples = [sample[0] for sample in batch]
        neg_heads = [sample[1][0] for sample in batch]
        neg_tails = [sample[1][1] for sample in batch]

        pos = torch.stack([torch.tensor(_) for _ in pos_samples])       # [N, 3]
        neg_head = torch.stack([torch.tensor(_) for _ in neg_heads])    # [N, num_neg_sample]
        neg_tail = torch.stack([torch.tensor(_) for _ in neg_tails])    # [N, num_neg_sample]

        return pos, (neg_head, neg_tail)

    @staticmethod
    def random_sampling(self, positive_sample, is_head=True):
        head, relation, tail = positive_sample
        all_entities = torch.arange(0, self.num_entities-1)

        if is_head:
            true_entities = torch.tensor(self.true_head[(relation, tail)])
        else:
            true_entities = torch.tensor(self.true_tail[(head, relation)])

        mask = torch.isin(all_entities, true_entities, invert=True)

        negative_sample_pool = all_entities[mask]

        negative_samples = torch.randperm(len(negative_sample_pool))[
                           :self.negative_sample_size]

        return negative_sample_pool[negative_samples]
        # all_entities = torch.arange(0, self.num_entities - 1)
        #
        # if is_head:
        #     mask = np.isin(
        #         all_entities,
        #         self.true_head[(relation, tail)],
        #         assume_unique=True,
        #         invert=True
        #     )
        # else:
        #     mask = np.isin(
        #         all_entities,
        #         self.true_tail[(head, relation)],
        #         assume_unique=True,
        #         invert=True
        #     )
        #
        # negative_sample_pool = all_entities[mask]
        #
        # negative_samples = np.random.choice(
        #     negative_sample_pool,
        #     size=min(len(negative_sample_pool), self.negative_sample_size),
        #     replace=False
        # )
        #
        # return negative_samples


class TestDataLoader(Dataset):
    def __init__(self, triples, num_entities):
        self.triples = triples
        self.triples_set = set(triples)
        self.num_entities = num_entities

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        # need to return stuff here s.t it can be used for evaluation
        positive_sample = self.triples[idx] # [1, 3]

        head, relation, tail = positive_sample

        neg_heads = [(1, test_head) if (test_head, relation, tail) not in self.triples_set else
                     (0, test_head) for test_head in range(self.num_entities)]
        neg_head = torch.tensor(neg_heads)

        neg_tail = [(1, test_tail) if (head, relation, test_tail) not in self.triples_set else
                    (0, test_tail) for test_tail in range(self.num_entities)]
        neg_tail = torch.tensor(neg_tail)

        filter_bias = (torch.tensor(neg_head[:, 0]), torch.tensor(neg_tail[:, 0]))      # ([K, 1], [K, 1])
        neg_head, neg_tail = neg_head[:, 1], neg_tail[:, 1]     # [K, 1], [K, 1], K = num_entities

        return torch.tensor(positive_sample), torch.tensor(neg_head), torch.tensor(neg_tail), filter_bias

    @staticmethod
    def collate_fn(batch):
        positives = torch.stack([sample[0] for sample in batch])        # [N, 3]
        heads = torch.stack([sample[1] for sample in batch])            # [N, K]
        tails = torch.stack([sample[2] for sample in batch])            # [N, K]
        filter_heads = torch.stack([sample[3][0] for sample in batch])  # [N, K]
        filter_tails = torch.stack([sample[3][1] for sample in batch])  # [N, K]

        return positives, (heads, tails, filter_heads, filter_tails)
