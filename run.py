"""
Script for training DistMult.
"""
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataloader import *
from model import DistMult

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 32
BATCH_SIZE_TRAIN = 10
BATCH_SIZE_TEST = 10
NUM_EPOCHS = 1
LR = 1e-5


def train(model, train_dataloader, optimizer, num_epochs, device):
    model.train().to(device)
    full_len = len(train_dataloader)

    for e in range(num_epochs):
        for i, (positive, negatives) in enumerate(train_dataloader):
            print(f"Train: {i} / {full_len-1}")

            optimizer.zero_grad()

            positive.to(device)
            negatives[0].to(device)
            negatives[1].to(device)

            true_score, head_pred_score, tail_pred_score = model((positive, negatives))

            loss = F.margin_ranking_loss(true_score,
                                         head_pred_score,
                                         target=torch.ones_like(true_score),
                                         margin=1)

            loss += F.margin_ranking_loss(true_score,
                                          tail_pred_score,
                                          target=torch.ones_like(true_score),
                                          margin=1)

            loss = loss / BATCH_SIZE_TRAIN

            loss.backward()
            optimizer.step()


def test(model, test_loader, device):
    model.eval().to(device)

    mrr = 0
    hit_at_10 = 0
    num_samples = 0

    with torch.no_grad():
        full_len = len(test_loader)
        for i, (positive, negatives) in enumerate(test_loader):
            print(f"Test: {i} / {full_len-1}")

            positive.to(device)
            negatives[0].to(device)  # heads
            negatives[1].to(device)  # tails
            negatives[2].to(device)  # filter heads
            negatives[3].to(device)  # filter tails

            true_score, head_pred_score, tail_pred_score = model((positive, (negatives[0], negatives[1])), mode='test')

            head_ranks = get_ranks(positive, negatives[0], true_score, head_pred_score, negatives[2], 0)
            tail_ranks = get_ranks(positive, negatives[1], true_score, tail_pred_score, negatives[3], 2)

            mrr += (torch.sum(1.0 / head_ranks) + torch.sum(1.0 / tail_ranks)) / 2
            hit_at_10 += torch.sum(torch.where(head_ranks <= 10, torch.tensor([1.0]), torch.tensor([0.0])))
            num_samples += len(head_ranks)

    print("MRR: ", mrr / num_samples)
    print("HIT@10: ", hit_at_10 / num_samples)


def get_ranks(positive_sample, negative_samples, true_score, pred_score, filter, pos_idx):
    # print(positive_sample.shape, negative_samples.shape, pred_score.shape)
    # use filter to eliminate positive triplet from the pred_score
    pred_score = torch.where(filter.bool(), pred_score, torch.tensor(float('-inf')))
    # print(true_score.unsqueeze(1).shape, pred_score.shape)
    scores = torch.cat((true_score.unsqueeze(1), pred_score), dim=1)

    entities_in_question = torch.cat((positive_sample[:, pos_idx].unsqueeze(1), negative_samples), dim=1)

    sorted_scores = torch.argsort(scores, descending=True)
    # sorted_entities = torch.gather(entities_in_question, 1, sorted_scores)

    ranking = []
    for i in range(sorted_scores.size(0)):
        index = (sorted_scores[i, :] == positive_sample[i][pos_idx]).nonzero()
        ranking.append(index[0].item() + 1)  # index may contain multiple elements since we added the true value

    return torch.tensor(ranking)


def main():
    entities2id, relations2id = load_dict('./data/FB15k-237/entities.dict'), load_dict(
        './data/FB15k-237/relations.dict')
    train_data, test_data, val_data = (load_triples('./data/FB15k-237/train.txt', entities2id, relations2id),
                                       load_triples('./data/FB15k-237/test.txt', entities2id, relations2id),
                                       load_triples('./data/FB15k-237/valid.txt', entities2id, relations2id))

    train_data = TrainDataLoader(train_data, len(entities2id))
    train_dataloader = DataLoader(train_data,
                                  collate_fn=TrainDataLoader.collate_fn,
                                  batch_size=BATCH_SIZE_TRAIN,
                                  shuffle=True)

    test_data = TestDataLoader(test_data, len(entities2id))
    test_dataloader = DataLoader(test_data,
                                 collate_fn=TestDataLoader.collate_fn,
                                 batch_size=BATCH_SIZE_TEST,
                                 shuffle=True)

    model = DistMult(len(entities2id), len(relations2id), EMBED_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)

    train(model, train_dataloader, optimizer, NUM_EPOCHS, DEVICE)
    test(model, test_dataloader, DEVICE)


if __name__ == '__main__':
    main()
