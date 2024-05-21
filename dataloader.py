"""
Handles reading in Knowledge Graph and creates a dataloader for it.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


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
        true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

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
    def __init__(self, triples, num_entities, neg_sample_size=128):
        self.triples = triples
        # self.triples_set = set(triples)
        self.num_entities = num_entities
        self.neg_sample_size = neg_sample_size
        self.true_head, self.true_tail = _get_true_head_and_tail(self.triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        """
        idx = index we want to retrieve
        """
        positive_sample = torch.LongTensor(self.triples[idx])  # positive sample, [3]

        negative_head = torch.LongTensor(self.corrupt_sample(positive_sample, True))
        negative_tail = torch.LongTensor(self.corrupt_sample(positive_sample, False))

        return positive_sample, (negative_head, negative_tail)

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
        #
        # return positive_sample, (random_head, random_tail)

    def corrupt_sample(self, positive_sample, is_head):
        head, relation, tail = positive_sample
        # all_entities = np.arange(self.num_entities)  # -1?
        #
        # if is_head:
        #     mask = np.isin(all_entities, self.true_head[(relation, tail)], assume_unique=True, invert=True)
        # else:
        #     mask = np.isin(all_entities, self.true_tail[(head, relation)], assume_unique=True, invert=True)
        #
        # negative_samples = np.random.choice(all_entities[mask], size=min(len(all_entities[mask]), self.neg_sample_size),
        #                                     replace=False)

        negative_sample_list, negative_sample_size = [], 0
        while negative_sample_size < self.neg_sample_size:
            negative_sample = np.random.randint(self.num_entities, size=self.neg_sample_size * 2)
            if is_head:
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            else:
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )

            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_samples = np.concatenate(negative_sample_list)[:self.neg_sample_size]

        return negative_samples

    @staticmethod
    def collate_fn(batch):
        # batch contains list of tuples: [..., (positive_sample, (negative_head, negative_tail)),...]

        positives, negatives = zip(*batch)

        batched_positive = torch.stack(positives)  # [N, 3]

        # each batched_* below is of shape [N, num_negative_samples]
        batched_negative_heads = torch.stack([neg[0] for neg in negatives])
        batched_negative_tails = torch.stack([neg[1] for neg in negatives])

        return [batched_positive, (batched_negative_heads, batched_negative_tails)]


class TestDataLoader(Dataset):
    def __init__(self, triples, all_triples, num_entities):
        self.triples = triples
        self.all_triples = set(all_triples)
        self.num_entities = num_entities
        self.true_head, self.true_tail = _get_true_head_and_tail(self.all_triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        positive_sample = torch.LongTensor(self.triples[idx])  # [3]

        negative_head, head_filter = self.corrupt_sample(positive_sample, True)  # [K]
        negative_tail, tail_filter = self.corrupt_sample(positive_sample, False)  # [K]

        return positive_sample, (torch.LongTensor(negative_head),
                                 torch.LongTensor(negative_tail),
                                 torch.BoolTensor(head_filter),
                                 torch.BoolTensor(tail_filter))

    @staticmethod
    def collate_fn(batch):
        positives, negatives = zip(*batch)

        batched_positive = torch.stack(positives)  # [N, 3]

        # each batched_* below is of shape [N, K]
        batched_negative_heads = torch.stack([neg[0] for neg in negatives])
        batched_negative_tails = torch.stack([neg[1] for neg in negatives])
        batched_head_filter = torch.stack([neg[2] for neg in negatives])
        batched_tail_filter = torch.stack([neg[3] for neg in negatives])

        return [batched_positive, (batched_negative_heads,
                                   batched_negative_tails,
                                   batched_head_filter,
                                   batched_tail_filter)]

        # positives = torch.stack([sample[0] for sample in batch])  # [N, 3]
        # heads = torch.stack([sample[1] for sample in batch])  # [N, K]
        # tails = torch.stack([sample[2] for sample in batch])  # [N, K]
        # filter_heads = torch.stack([sample[3][0] for sample in batch])  # [N, K]
        # filter_tails = torch.stack([sample[3][1] for sample in batch])  # [N, K]
        #
        # return positives, (heads, tails, filter_heads, filter_tails)

    def corrupt_sample(self, positive_sample, is_head):
        # diff from train, return all entities as negative
        head, relation, tail = positive_sample

        if is_head:
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.all_triples
                   else (-1, head) for rand_head in range(self.num_entities)]
            tmp[head] = (0, head)
        else:
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.all_triples
                   else (-1, tail) for rand_tail in range(self.num_entities)]
            tmp[tail] = (0, tail)

        tmp = torch.LongTensor(tmp)
        filtr = tmp[:, 0].float()
        negative = tmp[:, 1]

        # all_entities = np.arange(self.num_entities)

        # if is_head:
        #     np.delete(all_entities, head)
        #     mask = np.in1d(all_entities, self.true_head[(relation, tail)], assume_unique=True, invert=True)
        # else:
        #     np.delete(all_entities, tail)
        #     mask = np.in1d(all_entities, self.true_tail[(head, relation)], assume_unique=True, invert=True)

        return negative, filtr
