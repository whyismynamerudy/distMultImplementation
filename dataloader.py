"""
Handles reading in Knowledge Graph and creates a dataloader for it.
"""

import torch
from torch.utils.data import Dataset


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
    def __init__(self, triples, num_entities):
        self.triples = triples
        self.triples_set = set(triples)
        self.num_entities = num_entities

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        """
        Return positive sample corresponding to idx with negative sample.

        In the paper, they sample 2 negative triplets (one head, one tail) for each positive sample.

        :param idx: Index of desired triplet.
        :return: Tensors pos_sample [1, 3], neg_head [1], and neg_tail [1].
        """
        positive_sample = self.triples[idx]  # positive sample

        neg_head, neg_tail = list(positive_sample[:]), list(positive_sample[:])
        random_head = int(torch.randint(low=0, high=self.num_entities, size=(1,)).flatten())
        random_tail = int(torch.randint(low=0, high=self.num_entities, size=(1,)).flatten())

        neg_head[0] = random_head
        while set(neg_head) in self.triples_set:
            random_head = int(torch.randint(low=0, high=self.num_entities, size=(1,)).flatten())
            neg_head[0] = random_head

        neg_tail[2] = random_tail
        while set(neg_tail) in self.triples_set:
            random_tail = int(torch.randint(low=0, high=self.num_entities, size=(1,)).flatten())
            neg_tail[2] = random_tail

        return positive_sample, (random_head, random_tail)

    @staticmethod
    def collate_fn(batch):
        # batch contains list of tuples (pos_sample, random_head, random_tail)

        pos_samples = [sample[0] for sample in batch]
        neg_heads = [sample[1][0] for sample in batch]
        neg_tails = [sample[1][1] for sample in batch]

        pos = torch.stack([torch.tensor(_) for _ in pos_samples])       # [N, 3]
        neg_head = torch.stack([torch.tensor(_) for _ in neg_heads])    # [N]
        neg_tail = torch.stack([torch.tensor(_) for _ in neg_tails])    # [N]

        return pos, (neg_head, neg_tail)


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
