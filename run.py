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


# include loss for validation interval / epoch within the range and include those as well
# include validation MRR, HIT@10 as well during interval
def train(model, train_dataloader, valid_dataloader, optimizer, num_epochs, lambda_reg):
    model.train()
    optimizer.zero_grad()

    train_losses = []
    val_mrr, val_hit_at_10 = [], []

    for e in range(num_epochs):
        epoch_loss = 0
        num_samples = 0
        print("Epoch [{}/{}]".format(e + 1, num_epochs))

        for (positive, negatives) in tqdm(train_dataloader):
            positive.to(DEVICE)
            negatives[0].to(DEVICE)
            negatives[1].to(DEVICE)

            true_score, head_pred_score, tail_pred_score = model((positive, negatives))

            loss = (F.margin_ranking_loss(true_score,
                                          head_pred_score,
                                          target=torch.ones_like(true_score),
                                          margin=1) +
                    F.margin_ranking_loss(true_score,
                                          tail_pred_score,
                                          target=torch.ones_like(true_score),
                                          margin=1))

            reg = lambda_reg * (model.entity_emb.weight.norm(p=2) + model.relation_emb.weight.norm(p=2))

            batch_size = true_score.size(0)
            epoch_loss += loss.item()
            num_samples += batch_size

            loss = loss / batch_size + reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Loss: {epoch_loss / num_samples}")

        # if e % 5 == 0:
        #     train_losses.append(epoch_loss / num_samples)
        #     val_mrr_score, val_hit_score = validate(model, valid_dataloader)
        #     val_mrr.append(val_mrr_score)
        #     val_hit_at_10.append(val_hit_score)
        #     print("MRR: {}, Hit@10: {}".format(val_mrr_score, val_hit_score))

    return train_losses, val_mrr, val_hit_at_10


# def validate(model, dataloader):
#     model.eval()
#     total_loss = 0
#     num_samples = 0
#
#     mrr = 0
#     hit_at_10 = 0
#
#     with torch.no_grad():
#         for (positive, negatives) in tqdm(dataloader):
#             positive.to(DEVICE)
#             negatives[0].to(DEVICE)
#             negatives[1].to(DEVICE)
#             negatives[2].to(DEVICE)
#             negatives[3].to(DEVICE)
#
#             true_score, head_pred_score, tail_pred_score = model((positive, negatives))
#
#             head_ranks = get_ranks(positive, negatives[0], true_score, head_pred_score, negatives[2], 0)
#             tail_ranks = get_ranks(positive, negatives[1], true_score, tail_pred_score, negatives[3], 2)
#
#             mrr += (torch.sum(1.0 / head_ranks) + torch.sum(1.0 / tail_ranks)) / 2
#             hit_at_10 += torch.sum(
#                 torch.where(head_ranks <= 10, torch.tensor([1.0]).to(DEVICE), torch.tensor([0.0]).to(DEVICE)))
#             num_samples += len(head_ranks)
#
#             loss = F.margin_ranking_loss(true_score,
#                                          torch.mean(head_pred_score, 1),
#                                          target=torch.ones_like(true_score),
#                                          margin=1)
#             loss += F.margin_ranking_loss(true_score,
#                                           torch.mean(tail_pred_score, 1),
#                                           target=torch.ones_like(true_score),
#                                           margin=1)
#             total_loss += loss.item()
#
#     mrr, hit_at_10 = mrr / num_samples, hit_at_10 / num_samples
#
#     return mrr, hit_at_10


def test(model, test_loader):
    model.eval()

    mrr = 0
    hit_at_10 = 0
    num_samples = 0

    with torch.no_grad():
        e = 0
        for (positive, negatives) in tqdm(test_loader):
            positive.to(DEVICE)
            negatives[0].to(DEVICE)  # heads
            negatives[1].to(DEVICE)  # tails
            negatives[2].to(DEVICE)  # filter heads
            negatives[3].to(DEVICE)  # filter tails

            true_score, head_pred_score, tail_pred_score = model((positive, negatives))

            head_ranks = get_ranks(positive, negatives[0], true_score, head_pred_score, negatives[2], 0)
            tail_ranks = get_ranks(positive, negatives[1], true_score, tail_pred_score, negatives[3], 2)

            mrr += (torch.sum(1.0 / head_ranks) + torch.sum(1.0 / tail_ranks)) / 2
            hit_at_10 += torch.sum(
                torch.where(head_ranks <= 10, torch.FloatTensor([1.0]), torch.FloatTensor([0.0])))
            num_samples += true_score.size(0)

            if e < 10:
                e += 1
            else:
                break

    mrr, hit_at_10 = mrr / num_samples, hit_at_10 / num_samples

    print("MRR: ", mrr)
    print("HIT@10: ", hit_at_10)

    return mrr, hit_at_10


def get_ranks(positive_sample, negative_samples, true_score, pred_score, filter, pos_idx):
    pred_score = torch.where(filter, pred_score, torch.IntTensor([torch.iinfo(torch.int32).min]).to(DEVICE))
    scores = torch.cat((true_score, pred_score), dim=1).to(DEVICE)
    all_samples = torch.cat((positive_sample[:, pos_idx].unsqueeze(1), negative_samples), dim=1).to(DEVICE)

    sorted_scores = torch.argsort(scores, descending=True).to(DEVICE)
    sorted_samples = torch.gather(all_samples, 1, sorted_scores)

    ranking = []
    for i in range(sorted_scores.size(0)):
        index = (sorted_samples[i, :] == positive_sample[i][pos_idx]).nonzero()
        ranking.append(index[0].item() + 1)

    return torch.IntTensor(ranking)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for training DistMult.')

    parser.add_argument('--embed_dim', type=int, default=32,
                        help='Dimension of entity embeddings (default: 32)')

    parser.add_argument('--batch_size_train', type=int, default=10,
                        help='Batch size for training (default: 10)')

    parser.add_argument('--batch_size_test', type=int, default=10,
                        help='Batch size for testing (default: 10)')

    parser.add_argument('--neg_sample_size', type=int, default=128)

    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs for training (default: 1)')

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')

    parser.add_argument('--lambda_reg', type=float, default=1e-3,
                        help='Regularization parameter (default: 1e-3)')

    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')

    parser.add_argument('--do_test', action='store_true', default=False,
                        help='Test the model on the test set after training')

    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save trained models (default: models)')

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
        f.write(f"Details:\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
        f.write("\n *** \n")


def hyperparameter_search(train_dataloader, val_dataloader, entities2id, relations2id, save_dir, batch_train,
                          batch_test,
                          mrr_or_hit=0):
    # mrr_or_hit is idx for the results returned by test func
    best_hyperparameters = None
    best_result = 0.0
    best_model = None

    for num_epochs in [50, 100, 150, 200]:
        for embed_dim in [364, 128, 256, 512]:
            for lr in [1e-3, 1e-4, 1e-5]:
                for weight_decay in [0.0001, 0.001, 0.01]:
                    model = DistMult(len(entities2id), len(relations2id), embed_dim).to(DEVICE)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                    train_losses, val_mrr, val_hit_at_10 = train(model, train_dataloader, val_dataloader,
                                                                             optimizer, num_epochs)
                    results = test(model, val_dataloader)

                    if results[mrr_or_hit] > best_result:
                        best_result = results[mrr_or_hit]
                        best_hyperparameters = {
                            'embed_dim': embed_dim,
                            'lr': lr,
                            'weight_decay': weight_decay,
                            'num_epochs': num_epochs,
                            'batch_train': batch_train,
                            'batch_test': batch_test,
                            'train_losses': train_losses,
                            'val_mrr': val_mrr,
                            'val_hit_at_10': val_hit_at_10,
                        }
                        best_model = model
                        if save_dir:
                            save_model(model, save_dir, results, best_hyperparameters, "best_model.pth")

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
    LAMBDA_REG = args.lambda_reg
    NEG_SAMPLE_SIZE = args.neg_sample_size

    print("Beginning loading data...")

    entities2id, relations2id = load_dict('./data/FB15k-237/entities.dict'), load_dict(
        './data/FB15k-237/relations.dict')
    train_data, test_data, val_data = (load_triples('./data/FB15k-237/train.txt', entities2id, relations2id),
                                       load_triples('./data/FB15k-237/test.txt', entities2id, relations2id),
                                       load_triples('./data/FB15k-237/valid.txt', entities2id, relations2id))

    train_data = TrainDataLoader(train_data, len(entities2id), NEG_SAMPLE_SIZE)
    train_dataloader = DataLoader(train_data,
                                  collate_fn=TrainDataLoader.collate_fn,
                                  batch_size=BATCH_SIZE_TRAIN,
                                  shuffle=True)

    print("Loaded Train.")

    val_data = TestDataLoader(val_data, train_data+val_data+test_data, len(entities2id))
    val_dataloader = DataLoader(val_data,
                                collate_fn=TestDataLoader.collate_fn,
                                batch_size=BATCH_SIZE_TEST,
                                shuffle=True)

    print("Loaded Val.")

    test_data = TestDataLoader(test_data, train_data+val_data+test_data, len(entities2id))
    test_dataloader = DataLoader(test_data,
                                 collate_fn=TestDataLoader.collate_fn,
                                 batch_size=BATCH_SIZE_TEST,
                                 shuffle=True)

    print("Loaded Test.")

    print("Loaded all Data.")

    # if pretrained model provided, load it in directly and test
    if args.pretrained_model_path:
        if not args.do_test:
            print("Error: When providing a pretrained model path, please use the '--do_test' flag to test the model.")
            return

        model = DistMult(len(entities2id), len(relations2id), EMBED_DIM)
        model.load_state_dict(torch.load(args.pretrained_model_path))
        test(model, test_dataloader)
        return

    if args.do_hyperparameter_search:
        best_hyperparameters, best_model = hyperparameter_search(train_dataloader, val_dataloader, entities2id,
                                                                 relations2id, args.save_dir, BATCH_SIZE_TRAIN,
                                                                 BATCH_SIZE_TEST, 0)

        if args.do_test:
            test(best_model, test_dataloader)

    else:
        model = DistMult(len(entities2id), len(relations2id), EMBED_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        hyperparameters = {
            'batch_train': BATCH_SIZE_TRAIN,
            'batch_test': BATCH_SIZE_TEST,
            'num_epochs': NUM_EPOCHS,
            'embed_dim': EMBED_DIM,
            'learning_rate': LR,
            'weight_decay': WEIGHT_DECAY,
            'lambda_reg': LAMBDA_REG,
            'neg_sample_size': NEG_SAMPLE_SIZE
        }
        print("Running with: ", hyperparameters)

        train_losses, val_mrr, val_hit_at_10 = train(model, train_dataloader, val_dataloader, optimizer,
                                                                 NUM_EPOCHS, LAMBDA_REG)
        mrr, hit_at_10 = test(model, val_dataloader)

        save_model(model, args.save_dir, (mrr, hit_at_10),
                   {
                       'embed_dim': EMBED_DIM,
                       'lr': LR,
                       'weight_decay': WEIGHT_DECAY,
                       'num_epochs': NUM_EPOCHS,
                       'batch_train': BATCH_SIZE_TRAIN,
                       'batch_test': BATCH_SIZE_TEST,
                       'train_losses': train_losses,
                       'val_mrr': val_mrr,
                       'val_hit_at_10': val_hit_at_10,
                   },
                   "model.pth")

        if args.do_test:
            test(model, test_dataloader)


if __name__ == '__main__':
    print("Running at {} on {}".format(CONSTANT_DATETIME, "CUDA" if torch.cuda.is_available() else "CPU"))
    main()
