import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from Node_Encoders import Node_Encoder
from Node_Aggregators import Node_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import sys
from GraphConsis import GraphConsis

def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphConsis model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--load_from_checkpoint', type=bool, default=False, help='Load from checkpoint or not')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--data', type = str)
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device(args.device)

    embed_dim = args.embed_dim

    path_data = '../data/' + args.data + ".pkl"
    data_file = open(path_data, 'rb')

    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, social_adj_lists, item_adj_lists, ratings_list = pickle.load(
        data_file)

    traindata = np.array(traindata)
    validdata = np.array(validdata)
    testdata = np.array(testdata)

    train_u = traindata[:, 0]
    train_v = traindata[:, 1]
    train_r = traindata[:, 2]

    valid_u = validdata[:, 0]
    valid_v = validdata[:, 1]
    valid_r = validdata[:, 2]

    test_u = testdata[:, 0]
    test_v = testdata[:, 1]
    test_r = testdata[:, 2]

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_v),
                                              torch.FloatTensor(valid_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings + 1, embed_dim).to(device)
    #node_feature
    node_agg = Node_Aggregator(v2e, r2e, u2e, embed_dim, r2e.num_embeddings - 1, cuda=device)
    node_enc = Node_Encoder(u2e, v2e, embed_dim, history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,
                              social_adj_lists, item_adj_lists, node_agg, cuda=device)

    # model
    graphconsis = GraphConsis(node_enc, r2e).to(device)
    optimizer = torch.optim.Adam(graphconsis.parameters(), lr=args.lr, weight_decay = 0.0001)

    # load from checkpoint
    if args.load_from_checkpoint == True:
        checkpoint = torch.load('models/' + args.data + '.pt')
        graphconsis.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):

        train(graphconsis, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae = test(graphconsis, device, valid_loader)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
            torch.save({
            'epoch': epoch,
            'model_state_dict': graphconsis.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'models/' + args.data + '.pt')
        else:
            endure_count += 1
        print("rmse on valid set: %.4f, mae:%.4f " % (expected_rmse, mae))
        rmse, mae = test(graphconsis, device, test_loader)
        print('rmse on test set: %.4f, mae:%.4f '%(rmse, mae))

        if endure_count > 5:
            break
    print('finished')


if __name__ == "__main__":
    main()
