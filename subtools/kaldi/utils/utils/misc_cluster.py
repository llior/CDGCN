#!/usr/bin/env python
# -*- coding: utf-8 -*-


def filter_clusters(clusters, min_size=None, max_size=None):
    if min_size is not None:
        clusters = [c[:] for c in clusters if len(c) >= min_size]
    if max_size is not None:
        clusters = [c[:] for c in clusters if len(c) <= max_size]
    return clusters


def get_cluster_idxs(clusters, size=1, type="smaller"):
    idxs = []
    #"bigger" indicate that the size of cluster which is bigger than 1 will be return
    for c in clusters:
        if (len(c) <= size and type=="smaller") or (len(c)>size and type=="bigger"):
            idxs.extend(c)
    return idxs
