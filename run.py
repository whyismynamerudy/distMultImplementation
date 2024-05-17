"""
Script for training DistMult.
"""
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from dataloader import *
from model import DistMult
import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONSTANT_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train(model, train_dataloader, optimizer, num_epochs, device):
    model.train().to(device)

    for e in range(num_epochs):
        epoch_loss = 0
        print(f"Epoch [{e + 1}/{num_epochs}]")

        for (positive, negatives) in tqdm(train_dataloader):
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

            loss = loss / len(positive)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Loss: {epoch_loss / num_epochs}")


def test(model, test_loader, device):
    model.eval().to(device)

    mrr = 0
    hit_at_10 = 0
    num_samples = 0

    with torch.no_grad():
        for (positive, negatives) in tqdm(test_loader):
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

    mrr, hit_at_10 = mrr / num_samples, hit_at_10 / num_samples

    print("MRR: ", mrr)
    print("HIT@10: ", hit_at_10)

    return mrr, hit_at_10


def get_ranks(positive_sample, negative_samples, true_score, pred_score, filter, pos_idx):
    # use filter to eliminate positive triplet from the pred_score
    pred_score = torch.where(filter.bool(), pred_score, torch.tensor(float('-inf'))).to(DEVICE)
    scores = torch.cat((true_score.unsqueeze(1), pred_score), dim=1).to(DEVICE)

    sorted_scores = torch.argsort(scores, descending=True).to(DEVICE)

    ranking = []
    for i in range(sorted_scores.size(0)):
        index = (sorted_scores[i, :] == positive_sample[i][pos_idx]).nonzero()
        ranking.append(index[0].item() + 1)

    return torch.tensor(ranking)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for training DistMult.')
    parser.add_argument('--embed_dim', type=int, default=32,
                        help='Dimension of entity embeddings (default: 32)')
    parser.add_argument('--batch_size_train', type=int, default=10,
                        help='Batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=10,
                        help='Batch size for testing (default: 10)')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs for training (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save trained models (default: models)')
    parser.add_argument('--do_test', action='store_true', help='Test the model on the test set after training')
    parser.add_argument('--do_hyperparameter_search', action='store_true', help='Perform hyperparameter search')
    parser.add_argument('--pretrained_model_path', type=str, default='', help='Path to pretrained model')
    args = parser.parse_args()

    if args.pretrained_model_path and not args.do_test:
        parser.error("When providing a pretrained model path, please use the '--do_test' flag to test the model.")

    return args


def save_model(model, save_dir, results, hyperparameters, model_name: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, f"{CONSTANT_DATETIME}_{model_name}")
    torch.save(model.state_dict(), model_path)

    results_path = os.path.join(save_dir, f"{CONSTANT_DATETIME}_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"MRR: {results[0]}\n")
        f.write(f"HIT@10: {results[1]}\n")
        f.write(f"Hyperparameters:\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
        f.write("\n *** \n")


def hyperparameter_search(train_dataloader, val_dataloader, entities2id, relations2id, save_dir,
                          mrr_or_hit=0):
    # mrr_or_hit is idx for the results returned by test func
    best_hyperparameters = None
    best_result = 0.0
    best_model = None

    for num_epochs in [16, 32, 64, 128]:
        for embed_dim in [32, 64, 128, 256]:
            for lr in [1e-3, 1e-4, 1e-5]:
                for weight_decay in [0.001, 0.01, 0.1]:
                    model = DistMult(len(entities2id), len(relations2id), embed_dim)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                    train(model, train_dataloader, optimizer, num_epochs, DEVICE)
                    results = test(model, val_dataloader, DEVICE)

                    if results[mrr_or_hit] > best_result:
                        best_result = results[mrr_or_hit]
                        best_hyperparameters = {
                            'embed_dim': embed_dim,
                            'lr': lr,
                            'weight_decay': weight_decay,
                            'num_epochs': num_epochs
                        }
                        best_model = model
                        if save_dir:
                            save_model(model, save_dir, results, best_hyperparameters, "best_model_hyperparam.pth")

    print("Best hyperparameters:", best_hyperparameters)
    return best_hyperparameters, best_model


def main():
    args = parse_arguments()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    EMBED_DIM = args.embed_dim
    BATCH_SIZE_TRAIN = args.batch_size_train
    BATCH_SIZE_TEST = args.batch_size_test
    NUM_EPOCHS = args.num_epochs
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay

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

    val_data = TestDataLoader(val_data, len(entities2id))
    val_dataloader = DataLoader(val_data,
                                collate_fn=TestDataLoader.collate_fn,
                                batch_size=BATCH_SIZE_TEST,
                                shuffle=True)

    test_data = TestDataLoader(test_data, len(entities2id))
    test_dataloader = DataLoader(test_data,
                                 collate_fn=TestDataLoader.collate_fn,
                                 batch_size=BATCH_SIZE_TEST,
                                 shuffle=True)

    if args.pretrained_model_path:
        if not args.do_test:
            print("Error: When providing a pretrained model path, please use the '--do_test' flag to test the model.")
            return

        model = DistMult(len(entities2id), len(relations2id), EMBED_DIM)
        model.load_state_dict(torch.load(args.pretrained_model_path))
        test(model, test_dataloader, DEVICE)
        return

    if args.do_hyperparameter_search:
        best_hyperparameters, best_model = hyperparameter_search(train_dataloader, val_dataloader, entities2id,
                                                                 relations2id, args.save_dir, 0)

        if args.do_test:
            test(best_model, test_dataloader, DEVICE)

    else:
        model = DistMult(len(entities2id), len(relations2id), EMBED_DIM)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        train(model, train_dataloader, optimizer, NUM_EPOCHS, DEVICE)
        mrr, hit_at_10 = test(model, val_dataloader, DEVICE)

        if args.save_dir:
            save_model(model, args.save_dir, (mrr, hit_at_10),
                       {'embed_dim': EMBED_DIM, 'lr': LR, 'weight_decay': WEIGHT_DECAY},
                       "best_model_from_arguments.pth")

        if args.do_test:
            test(model, test_dataloader, DEVICE)


if __name__ == '__main__':
    print("Running at ", CONSTANT_DATETIME)
    main()
