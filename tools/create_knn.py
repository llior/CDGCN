#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import inspect
import argparse
import warnings
from utils import (write_meta, set_random_seed, mkdir_if_no_exists,
                   BasicDataset, Timer, build_knns)



def parse_args():
    parser = argparse.ArgumentParser(description='Baseline Clustering')
    parser.add_argument("--name",
                        type=str,
                        default='part1_test',
                        help="image features")
    parser.add_argument("--prefix",
                        type=str,
                        default='./data',
                        help="prefix of dataset")
    parser.add_argument("--dim",
                        type=int,
                        default=256,
                        help="dimension of feature")
    parser.add_argument("--no_normalize",
                        action='store_true',
                        help="whether to normalize feature")
    # args for different methods
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--knn', default=80, type=int)
    parser.add_argument('--knn_method',
                        default='faiss',
                        choices=['faiss', 'faiss_gpu', 'hnsw'])
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)


    ds = BasicDataset(name=args.name,
                      prefix=args.prefix,
                      dim=args.dim,
                      normalize=not args.no_normalize)
    ds.info()
    feats = ds.features

    with Timer('{}'.format(args.knn_method)):
        knn_prefix = os.path.join(args.prefix, 'knns', args.name)
        knn_prefix = os.path.join(knn_prefix, '{}_k_{}'.format(args.knn_method, args.knn))
        if feats.shape[0]-1<=args.knn:
            warnings.warn("The number of features exceeds the K {}->{}".format(args.knn, feats.shape[0]))
            args.knn = feats.shape[0] - 1
        #使用faiss-gpu能显著提升计算速度，但是进行了近似求解使得某些时候邻居id出现-1的情况，而使用faiss不会
        knns = build_knns(knn_prefix, feats, args.knn_method, args.knn, num_process=4, is_rebuild=True)

