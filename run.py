#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    run.py
  Author:       locke
  Date created: 2018/10/5 下午2:37
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import time
import argparse
import gc
import random
import math
import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default="data/DBP15K/zh_en", required=False, help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--rate", type=float, default=0.3, help="training set rate")

    parser.add_argument("--save", default="output", help="the output dictionary of the model and embedding")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")

    parser.add_argument("--seed", type=int, default=2018, help="random seed")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--check_point", type=int, default=100, help="check point")

    parser.add_argument("--hidden_units", type=str, default="128,128,128", help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
    parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
    parser.add_argument("--instance_normalization", action="store_true", default=False, help="enable instance normalization")

    parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for layers")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")
    parser.add_argument("--dist", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')")

    parser.add_argument("--margin_CG", type=int, default=3, help="margin for cross-graph model")
    parser.add_argument("--margin_KE", type=int, default=3, help="margin for knowledge embedding model")
    parser.add_argument("--k_CG", type=int, default=25, help="negtive sampling number for cross-graph model")
    parser.add_argument("--k_KE", type=int, default=2, help="negtive sampling number for knowledge embedding model")
    parser.add_argument("--update_num", type=int, default=5, help="number of epoch for updating negtive samples")

    parser.add_argument("--wo_K", action="store_true", default=False, help="baseline w/o Knowledge embedding model")
    parser.add_argument("--wo_NNS", action="store_true", default=False, help="baseline w/o NNS")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    K_CG = args.k_CG
    K_KE = args.k_KE
    

    # Load data
    lang_list = [1, 2]
    ent2id_dict, ills, triples, r_hs, r_ts, ids = read_raw_data(args.file_dir, lang_list)
    np.random.shuffle(ills)
    train_ill = np.array(ills[:int(len(ills) // 1 * args.rate)], dtype=np.int32)
    test_ill = np.array(ills[int(len(ills) // 1 * args.rate):], dtype=np.int32)

    test_left = torch.LongTensor(test_ill[:, 0].squeeze()).to(device)
    test_right = torch.LongTensor(test_ill[:, 1].squeeze()).to(device)
    
    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(r_hs)

    print("-----dataset summary-----")
    print("dataset:\t", args.file_dir)
    print("triple num:\t", len(triples))
    print("entity num:\t", ENT_NUM)
    print("relation num:\t", REL_NUM)
    print("train ill num:\t", train_ill.shape[0], "\ttest ill num:\t", test_ill.shape[0])
    print("-------------------------")

    input_dim = int(args.hidden_units.strip().split(",")[0])

    entity_emb = nn.Embedding(ENT_NUM, input_dim)
    nn.init.normal_(entity_emb.weight, std=1.0 / math.sqrt(ENT_NUM))
    entity_emb.requires_grad = True
    entity_emb = entity_emb.to(device)

    relation_emb = nn.Embedding(REL_NUM, input_dim)
    nn.init.xavier_uniform_(relation_emb.weight)
    relation_emb.requires_grad = True
    relation_emb = relation_emb.to(device)

    input_idx = torch.LongTensor(np.arange(ENT_NUM)).to(device)
    
    adj = get_adjr(ENT_NUM, triples, norm=True)
    adj = adj.to(device)


    # Set model
    n_units = [int(x) for x in args.hidden_units.strip().split(",")]
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    cross_graph_model = GAT(n_units=n_units, n_heads=n_heads, dropout=args.dropout, attn_dropout=args.attn_dropout, instance_normalization=args.instance_normalization, diag=True).to(device)

    params = [{"params": filter(lambda p: p.requires_grad, list(cross_graph_model.parameters()) + [entity_emb.weight, relation_emb.weight])}]
    optimizer = optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)
    print(cross_graph_model)
    print(optimizer)


    # Train
    print("training...")
    t_total = time.time()
    epoch_KE, epoch_CG = 0, 0
    for epoch in range(args.epochs):
        t_epoch = time.time()
        cross_graph_model.train()
        optimizer.zero_grad()

        attention_enhanced_emb = cross_graph_model(entity_emb(input_idx), adj)
        
        if args.wo_K:
            print("w\\o K")

        if args.wo_K or epoch % 2 == 0:
            if epoch_CG == 0:
                train_left = torch.LongTensor((np.ones((train_ill.shape[0], K_CG)) * (train_ill[:, 0].reshape((train_ill.shape[0], 1)))).reshape((train_ill.shape[0] * K_CG,))).to(device)
                train_right = torch.LongTensor((np.ones((train_ill.shape[0], K_CG)) * (train_ill[:, 1].reshape((train_ill.shape[0], 1)))).reshape((train_ill.shape[0] * K_CG,))).to(device)
                print("\ttrain pos/neg_pairs shape: {}".format(train_left.shape))
            if epoch_CG % args.update_num == 0:
                if args.wo_NNS:
                    print("w\\o NNS")
                if args.wo_NNS or epoch_CG == 0:
                    neg_left = torch.LongTensor(np.random.choice(ENT_NUM, train_ill.shape[0] * K_CG)).to(device)
                    neg_right = torch.LongTensor(np.random.choice(ENT_NUM, train_ill.shape[0] * K_CG)).to(device)
                else:
                    with torch.no_grad():
                        neg_left, neg_right = nearest_neighbor_sampling(attention_enhanced_emb.cpu(), torch.LongTensor(train_ill[:, 0]), torch.LongTensor(train_ill[:, 1]), K_CG)
                        neg_left, neg_right = neg_left.to(device), neg_right.to(device)
            epoch_CG += 1

            # Cross-graph model alignment loss
            loss_CG = F.triplet_margin_loss(torch.cat((attention_enhanced_emb[train_left], attention_enhanced_emb[train_right]), dim=0),
                                               torch.cat((attention_enhanced_emb[train_right], attention_enhanced_emb[train_left]), dim=0),
                                               torch.cat((attention_enhanced_emb[neg_left], attention_enhanced_emb[neg_right]), dim=0),
                                               margin=args.margin_CG, p=args.dist)

            loss_CG.backward()
            print("loss_CG in epoch {:d}: {:f}, time: {:.4f} s".format(epoch, loss_CG.item(), time.time() - t_epoch))

        else:
            if epoch_KE == 0:
                true_triples = torch.cat(tuple([torch.LongTensor(triples) for _ in range(K_KE)]), dim=0).to(device)
                print("\ttrain pos/neg_triples shape: {}".format(true_triples.shape))
            if epoch_KE % args.update_num == 0:
                neg_triples = torch.cat(tuple([torch.LongTensor(multi_typed_sampling(triples, triples, r_hs, r_ts, ids, x)) for x in range(K_KE)]), dim=0).to(device)
            epoch_KE += 1

            # Knowledge embedding model loss
            X_1 = F.normalize(attention_enhanced_emb[true_triples[:, 0]] + relation_emb(true_triples[:, 1]) - attention_enhanced_emb[true_triples[:, 2]], p=args.dist)
            X_2 = F.normalize(attention_enhanced_emb[neg_triples[:, 0]] + relation_emb(neg_triples[:, 1]) - attention_enhanced_emb[neg_triples[:, 2]], p=args.dist)
            Y = torch.ones(X_1.size(0), 1).to(device)
            loss_KE = F.margin_ranking_loss(X_1.sum(1).view(-1, 1), X_2.sum(1).view(-1, 1), Y, args.margin_KE)

            loss_KE.backward()
            print("loss_KE in epoch {:d}: {:f}, time: {:.4f} s".format(epoch, loss_KE, time.time() - t_epoch))

        optimizer.step()


        # Test
        if (epoch + 1) % args.check_point == 0:
            print("\nepoch {:d}, checkpoint!".format(epoch))
            
            with torch.no_grad():
                t_test = time.time()
                cross_graph_model.eval()
                attention_enhanced_emb = cross_graph_model(entity_emb(input_idx), adj)

                top_k = [1, 5, 10, 50, 100]
                if "100" in args.file_dir:
                    Lvec = attention_enhanced_emb[test_left].cpu().data.numpy()
                    Rvec = attention_enhanced_emb[test_right].cpu().data.numpy()
                    acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l = multi_get_hits(Lvec, Rvec, top_k=top_k)
                    del attention_enhanced_emb
                    gc.collect()
                else:
                    acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
                    acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
                    test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
                    if args.dist == 2:
                        distance = pairwise_distances(attention_enhanced_emb[test_left], attention_enhanced_emb[test_right])
                    elif args.dist == 1:
                        distance = torch.FloatTensor(scipy.spatial.distance.cdist(attention_enhanced_emb[test_left].cpu().data.numpy(), attention_enhanced_emb[test_right].cpu().data.numpy(), metric="cityblock"))
                    else:
                        raise NotImplementedError
                    for idx in range(test_left.shape[0]):
                        values, indices = torch.sort(distance[idx, :], descending=False)
                        rank = (indices == idx).nonzero().squeeze().item()
                        mean_l2r += (rank + 1)
                        mrr_l2r += 1.0 / (rank + 1)
                        for i in range(len(top_k)):
                            if rank < top_k[i]:
                                acc_l2r[i] += 1
                    for idx in range(test_right.shape[0]):
                        _, indices = torch.sort(distance[:, idx], descending=False)
                        rank = (indices == idx).nonzero().squeeze().item()
                        mean_r2l += (rank + 1)
                        mrr_r2l += 1.0 / (rank + 1)
                        for i in range(len(top_k)):
                            if rank < top_k[i]:
                                acc_r2l[i] += 1
                    mean_l2r /= test_left.size(0)
                    mean_r2l /= test_right.size(0)
                    mrr_l2r /= test_left.size(0)
                    mrr_r2l /= test_right.size(0)
                    for i in range(len(top_k)):
                        acc_l2r[i] = round(acc_l2r[i] / test_left.size(0), 4)
                        acc_r2l[i] = round(acc_r2l[i] / test_right.size(0), 4)
                    del distance, attention_enhanced_emb
                    gc.collect()
                print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r, mean_l2r, mrr_l2r, time.time() - t_test))
                print("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(top_k, acc_r2l, mean_r2l, mrr_r2l, time.time() - t_test))

        if args.cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("optimization finished!")
    print("total time elapsed: {:.4f} s".format(time.time() - t_total))

    if args.save != "":
        time_str = time.strftime("%Y%m%d-%H%M", time.gmtime())
        torch.save(cross_graph_model, args.save + "/%s_model.pkl" % (time_str))
        with torch.no_grad():
            cross_graph_model.eval()
            attention_enhanced_emb = cross_graph_model(entity_emb(input_idx), adj)
            np.save(args.save + "/%s_ent_vec.npy" % (time_str), attention_enhanced_emb.cpu().detach().numpy())
            np.save(args.save + "/%s_rel_vec.npy" % (time_str), relation_emb.weight.cpu().detach().numpy())
        print("model and embeddings saved!")


if __name__ == "__main__":
    main()
