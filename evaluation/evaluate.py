#!/usr/bin/env python
# -*- coding: utf-8 -*-
# xmuspeech wangjie 2023.3.1
import inspect
import argparse
import numpy as np

import evaluation.metrics as metrics
from utils import Timer, TextColors, lab2cls, clusters2labels, intdict2ndarray,get_cluster_idxs



def _read_meta(fn):
    labels = list()
    lb_set = set()
    with open(fn) as f:
        for lb in f.readlines():
            lb = int(lb.strip())
            labels.append(lb)
            lb_set.add(lb)
    return np.array(labels), lb_set

def filtered_evaluate(gt_labels, pred_labels, metric, max_size):

    if isinstance(gt_labels, str) and isinstance(pred_labels, str):
        print('[gt_labels] {}'.format(gt_labels))
        print('[pred_labels] {}'.format(pred_labels))
        gt_labels, gt_lb_set = _read_meta(gt_labels)
        pred_labels, pred_lb_set = _read_meta(pred_labels)
        clusters = lab2cls(pred_labels)
        pred_idx2lb = clusters2labels(clusters)
        pred_labels = intdict2ndarray(pred_idx2lb)

        gt_clusters = lab2cls(gt_labels)
        gt_idx2lb = clusters2labels(gt_clusters)
        gt_labels = intdict2ndarray(gt_idx2lb)

        filtered_cluster_idxs = get_cluster_idxs(gt_clusters, size=max_size, type="bigger")

        remain_idxs = np.setdiff1d(np.arange(len(gt_labels)),
                                   np.array(filtered_cluster_idxs))
        remain_idxs = np.array(remain_idxs)

        print('==> evaluation (filter {} bigsize clusters)'.format(
            len(filtered_cluster_idxs)))
        print('#inst: gt({}) vs pred({})'.format(len(gt_labels[remain_idxs]),
                                                 len(pred_labels[remain_idxs])))
        print('#cls: gt({}) vs pred({})'.format(len(set(gt_labels[remain_idxs])),
                                                len(set(pred_labels[remain_idxs]))))

    metric_func = metrics.__dict__[metric]

    with Timer('evaluate with {}{}{}'.format(TextColors.FATAL, metric,
                                             TextColors.ENDC)):
        result = metric_func(gt_labels[remain_idxs], pred_labels[remain_idxs])
    if isinstance(result, np.float):
        print('{}{}: {:.4f}{}'.format(TextColors.OKGREEN, metric, result,
                                      TextColors.ENDC))
    else:
        ave_pre, ave_rec, fscore = result
        print('{}ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}{}'.format(
            TextColors.OKGREEN, ave_pre, ave_rec, fscore, TextColors.ENDC))

def mv_single_eval(gt_labels, pred_labels, metric='pairwise'):
    if isinstance(gt_labels, str) and isinstance(pred_labels, str):

        print('[gt_labels] {}'.format(gt_labels))
        print('[pred_labels] {}'.format(pred_labels))
        gt_labels, gt_lb_set = _read_meta(gt_labels)
        pred_labels, pred_lb_set = _read_meta(pred_labels)
        clusters = lab2cls(pred_labels)
        pred_idx2lb = clusters2labels(clusters)
        pred_labels = intdict2ndarray(pred_idx2lb)
        single_cluster_idxs = get_cluster_idxs(clusters, size=20)

        remain_idxs = np.setdiff1d(np.arange(len(pred_labels)),
                                   np.array(single_cluster_idxs))
        remain_idxs = np.array(remain_idxs)

        print('==> evaluation (removing {} single clusters)'.format(
            len(single_cluster_idxs)))
        print('#inst: gt({}) vs pred({})'.format(len(gt_labels[remain_idxs]),
                                                 len(pred_labels[remain_idxs])))
        print('#cls: gt({}) vs pred({})'.format(len(set(gt_labels[remain_idxs])),
                                                len(set(pred_labels[remain_idxs]))))

    metric_func = metrics.__dict__[metric]

    with Timer('evaluate with {}{}{}'.format(TextColors.FATAL, metric,
                                             TextColors.ENDC)):
        result = metric_func(gt_labels[remain_idxs], pred_labels[remain_idxs])
    if isinstance(result, np.float):
        print('{}{}: {:.4f}{}'.format(TextColors.OKGREEN, metric, result,
                                      TextColors.ENDC))
    else:
        ave_pre, ave_rec, fscore = result
        print('{}ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}{}'.format(
            TextColors.OKGREEN, ave_pre, ave_rec, fscore, TextColors.ENDC))

def evaluate(gt_labels, pred_labels, metric='pairwise'):
    if isinstance(gt_labels, str) and isinstance(pred_labels, str):
        print('==> evaluation')
        print('[gt_labels] {}'.format(gt_labels))
        print('[pred_labels] {}'.format(pred_labels))
        gt_labels, gt_lb_set = _read_meta(gt_labels)
        pred_labels, pred_lb_set = _read_meta(pred_labels)

        print('#inst: gt({}) vs pred({})'.format(len(gt_labels),
                                                 len(pred_labels)))
        print('#cls: gt({}) vs pred({})'.format(len(gt_lb_set),
                                                len(pred_lb_set)))

    metric_func = metrics.__dict__[metric]

    with Timer('evaluate with {}{}{}'.format(TextColors.FATAL, metric,
                                             TextColors.ENDC)):
        result = metric_func(gt_labels, pred_labels)
    if isinstance(result, np.float):
        print('{}{}: {:.4f}{}'.format(TextColors.OKGREEN, metric, result,
                                      TextColors.ENDC))
    else:
        ave_pre, ave_rec, fscore = result
        print('{}ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}{}'.format(
            TextColors.OKGREEN, ave_pre, ave_rec, fscore, TextColors.ENDC))



if __name__ == '__main__':
    metric_funcs = inspect.getmembers(metrics, inspect.isfunction)
    metric_names = [n for n, _ in metric_funcs]

    parser = argparse.ArgumentParser(description='Evaluate Cluster')
    parser.add_argument('--gt_labels', type=str, required=True)
    parser.add_argument('--pred_labels', type=str, required=True)
    parser.add_argument('--metric', default='pairwise', choices=metric_names)
    parser.add_argument('--max_size', type=int, default=None, help="The size of cluster will not be evaluated")
    args = parser.parse_args()

    evaluate(args.gt_labels, args.pred_labels, args.metric)

    mv_single_eval(args.gt_labels, args.pred_labels, args.metric)
    if args.max_size:
        filtered_evaluate(args.gt_labels, args.pred_labels, args.metric, args.max_size)

    pass