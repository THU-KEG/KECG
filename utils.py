#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  File name:    utils.py
  Author:       locke
  Date created: 2018/10/5 下午2:38
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, time, multiprocessing
import math
import random
import numpy as np
import scipy
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def read_raw_data(file_dir, l=[1, 2]):
    print('loading raw data...')
    def read_file(file_paths):
        tups = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    tups.append(tuple([int(x) for x in params]))
        return tups
    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids
    ent2id_dict, ids = read_dict([file_dir + "/ent_ids_" + str(i) for i in l])
    ills = read_file([file_dir + "/ill_ent_ids"])
    triples = read_file([file_dir + "/triples_" + str(i) for i in l])
    r_hs, r_ts = {}, {}
    for (h, r, t) in triples:
        if r not in r_hs:
            r_hs[r] = set()
        if r not in r_ts:
            r_ts[r] = set()
        r_hs[r].add(h)
        r_ts[r].add(t)
    assert len(r_hs) == len(r_ts)
    return ent2id_dict, ills, triples, r_hs, r_ts, ids


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def multi_cal_neg(pos_triples, task, triples, r_hs_dict, r_ts_dict, ids, neg_scope):
    neg_triples = list()
    for idx, tas in enumerate(task):
        (h, r, t) = pos_triples[tas]
        h2, r2, t2 = h, r, t
        temp_scope, num = neg_scope, 0
        while True:
            choice = random.randint(0, 999)
            if choice < 500:
                if temp_scope:
                    h2 = random.sample(r_hs_dict[r], 1)[0]
                else:
                    for id in ids:
                        if h2 in id:
                            h2 = random.sample(id, 1)[0]
                            break
            else:
                if temp_scope:
                    t2 = random.sample(r_ts_dict[r], 1)[0]
                else:
                    for id in ids:
                        if t2 in id:
                            t2 = random.sample(id, 1)[0]
                            break
            if (h2, r2, t2) not in triples:
                break
            else:
                num += 1
                if num > 10:
                    temp_scope = False
        neg_triples.append((h2, r2, t2))
    return neg_triples


def multi_typed_sampling(pos_triples, triples, r_hs_dict, r_ts_dict, ids, neg_scope):
    t_ = time.time()
    triples = set(triples)
    tasks = div_list(np.array(range(len(pos_triples)), dtype=np.int32), 10)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(multi_cal_neg, (pos_triples, task, triples, r_hs_dict, r_ts_dict, ids, neg_scope)))
    pool.close()
    pool.join()
    neg_triples = []
    for res in reses:
        neg_triples.extend(res.get())
    return neg_triples


def nearest_neighbor_sampling(emb, left, right, K):
    t = time.time()
    neg_left = []
    distance = pairwise_distances(emb[right], emb[right])
    for idx in range(right.shape[0]):
        _, indices = torch.sort(distance[idx, :], descending=False)
        neg_left.append(right[indices[1 : K+1]])
    neg_left = torch.cat(tuple(neg_left), dim=0)
    neg_right = []
    distance = pairwise_distances(emb[left], emb[left])
    for idx in range(left.shape[0]):
        _, indices = torch.sort(distance[idx, :], descending=False)
        neg_right.append(left[indices[1 : K+1]])
    neg_right = torch.cat(tuple(neg_right), dim=0)
    return neg_left, neg_right


def get_adjr(ent_size, triples, norm=False):
    print('getting a sparse tensor r_adj...')
    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 0
        M[(tri[0], tri[2])] += 1
    ind, val = [], []
    for (fir, sec) in M:
        ind.append((fir, sec))
        ind.append((sec, fir))
        val.append(M[(fir, sec)])
        val.append(M[(fir, sec)])
    for i in range(ent_size):
        ind.append((i, i))
        val.append(1)
    if norm:
        ind = np.array(ind, dtype=np.int32)
        val = np.array(val, dtype=np.float32)
        adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    else:
        M = torch.sparse_coo_tensor(torch.LongTensor(ind).t(), torch.FloatTensor(val), torch.Size([ent_size, ent_size]))
        return M


# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)


def multi_cal_rank(task, sim, top_k, l_or_r):
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    for i in range(len(task)):
        ref = task[i]
        if l_or_r == 0:
            rank = (sim[i, :]).argsort()
        else:
            rank = (sim[:, i]).argsort()
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, num, mrr


def multi_get_hits(Lvec, Rvec, top_k=(1, 5, 10, 50, 100)):
    result = []
    sim = pairwise_distances(torch.FloatTensor(Lvec), torch.FloatTensor(Rvec)).numpy()
    for i in [0, 1]:
        top_total = np.array([0] * len(top_k))
        mean_total, mrr_total = 0.0, 0.0
        s_len = Lvec.shape[0] if i == 0 else Rvec.shape[0]
        tasks = div_list(np.array(range(s_len)), 10)
        pool = multiprocessing.Pool(processes=len(tasks))
        reses = list()
        for task in tasks:
            if i == 0:
                reses.append(pool.apply_async(multi_cal_rank, (task, sim[task, :], top_k, i)))
            else:
                reses.append(pool.apply_async(multi_cal_rank, (task, sim[:, task], top_k, i)))
        pool.close()
        pool.join()
        for res in reses:
            mean, num, mrr = res.get()
            mean_total += mean
            mrr_total += mrr
            top_total += np.array(num)
        acc_total = top_total / s_len
        for i in range(len(acc_total)):
            acc_total[i] = round(acc_total[i], 4)
        mean_total /= s_len
        mrr_total /= s_len
        result.append(acc_total)
        result.append(mean_total)
        result.append(mrr_total)
    return result


if __name__ == '__main__':
    pass
