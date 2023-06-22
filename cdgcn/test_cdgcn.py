from __future__ import division

import os
import torch
import numpy as np
import torch.nn.functional as F
from .cluster_infomap import Link_cluster_infomap
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from cdgcn.datasets import build_dataset, build_dataloader
from cdgcn.online_evaluation import online_evaluate

from utils import (clusters2labels, intdict2ndarray, get_cluster_idxs, filter_clusters,
                   write_meta)
from proposals.graph import graph_clustering_dynamic_th
from evaluation import evaluate
from CommunityDetection.community_utils import Link_cluster_leiden, Link_cluster_louvain, overlap_community_detection
import igraph as ig

def merge_single_cluster(dataset, clusters, min_size=20, max_size=500):

    dense_clusters = filter_clusters(clusters=clusters, min_size=min_size+1, max_size=max_size)
    single_clusters = filter_clusters(clusters=clusters, max_size=min_size)
    if len(dense_clusters)==0:
        print("Warming: The number of dense_clusters is 0. return cluster without mergence")
        return clusters
    elif len(single_clusters)==0:
        print("The number of single_clusters is 0. return cluster without mergence")
        return clusters
    X_center = np.array([
        dataset.features[dense_clusters[i]].mean(axis=0)
        for i in range(len(dense_clusters))
    ])
    X_single = np.array([
        dataset.features[single_clusters[i]].mean(axis=0)
        for i in range(len(single_clusters))
    ])
    merged_labels = np.argmax(np.matmul(X_single, X_center.T), axis=1)
    for index, label in enumerate(merged_labels):
        dense_clusters[label].extend(single_clusters[index])

    return dense_clusters


def test(model, dataset, cfg, logger):
    if cfg.load_from:
        print('load from {}'.format(cfg.load_from))
        load_checkpoint(model, cfg.load_from, strict=True, logger=logger)

    losses = []
    edges = []
    scores = []

    if cfg.gpus == 1:
        data_loader = build_dataloader(dataset,
                                       cfg.batch_size_per_gpu,
                                       cfg.workers_per_gpu,
                                       train=False)

        model = MMDataParallel(model, device_ids=range(cfg.gpus))
        if cfg.cuda:
            model.cuda()

        model.eval()
        for i, (data, cid, node_list) in enumerate(data_loader):
            with torch.no_grad():
                _, _, h1id, gtmat = data
                pred, loss = model(data, return_loss=True)
                losses += [loss.item()]
                pred = F.softmax(pred, dim=1)
                if i % cfg.log_config.interval == 0:
                    if dataset.ignore_label:
                        logger.info('[Test] Iter {}/{}'.format(
                            i, len(data_loader)))
                    else:
                        acc, p, r = online_evaluate(gtmat, pred)
                        logger.info(
                            '[Test] Iter {}/{}: Loss {:.4f}, '
                            'Accuracy {:.4f}, Precision {:.4f}, Recall {:.4f}'.
                            format(i, len(data_loader), loss, acc, p, r))

                node_list = node_list.numpy()
                bs = len(cid)
                h1id_num = len(h1id[0])
                for b in range(bs):
                    cidb = cid[b].int().item()
                    nlst = node_list[b]
                    center_idx = nlst[cidb]
                    for j, n in enumerate(h1id[b]):
                        edges.append([center_idx, nlst[n.item()]])
                        scores.append(pred[b * h1id_num + j, 1].item())
    else:
        raise NotImplementedError

    if not dataset.ignore_label:
        avg_loss = sum(losses) / len(losses)
        logger.info('[Test] Overall Loss {:.4f}'.format(avg_loss))

    return np.array(edges), np.array(scores), len(dataset)


def test_cdgcn(model, cfg, logger):
    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.test_data, k, v)
    dataset = build_dataset(cfg.test_data)
    ##modify by wanjie
    if "diarization" in cfg.keys():
        pred_scores_dir = os.path.join(cfg.work_dir, 'pred_edges_scores')
        if not os.path.exists(pred_scores_dir):
            os.makedirs(pred_scores_dir)
        ofn_pred = os.path.join(pred_scores_dir, 'pred_edges_scores_{}.npz'.format(cfg["test_name"]))
    else:
        ofn_pred = os.path.join(cfg.work_dir, 'pred_edges_scores.npz')
    if os.path.isfile(ofn_pred) and not cfg.force:
        data = np.load(ofn_pred)
        edges = data['edges']
        scores = data['scores']
        inst_num = data['inst_num']
        if inst_num != len(dataset):
            logger.warn(
                'instance number in {} is different from dataset: {} vs {}'.
                format(ofn_pred, inst_num, len(dataset)))
    else:
        edges, scores, inst_num = test(model, dataset, cfg, logger)

    gt_labels = dataset.labels

    ## 3.perform community detection
    cd_method = cfg.cd_params["method"]

    secondary_partition = []
    if cd_method == "BFS":
        # 1.produce predicted labels
        cd_clusters = graph_clustering_dynamic_th(edges,
                                               scores,
                                               max_sz=cfg.max_sz,
                                               step=cfg.step,
                                               pool=cfg.pool,
                                               max_iter=getattr(cfg, "max_iter", 100))
        cd_pred_idx2lb = clusters2labels(cd_clusters)
        cd_pred_labels = intdict2ndarray(cd_pred_idx2lb)
        ## 2.merge singel cluster
        if hasattr(cfg, "min_sz"):
            min_sz = cfg.min_sz
        else:
            min_sz = 20
        merged_clusters = merge_single_cluster(dataset, cd_clusters, min_size=min_sz, max_size=cfg.max_sz)
        merged_pred_idx2lb = clusters2labels(merged_clusters)
        merged_pred_labels = intdict2ndarray(merged_pred_idx2lb)

        # evaluation
        if not dataset.ignore_label:
            print('==> evaluation. The number of cluster is {}'.format(len(cd_clusters)))
            for metric in cfg.metrics:
                evaluate(gt_labels, cd_pred_labels, metric)

            single_cluster_idxs = get_cluster_idxs(cd_clusters, size=1)
            remain_idxs = np.setdiff1d(np.arange(len(dataset)),
                                       np.array(single_cluster_idxs))
            remain_idxs = np.array(remain_idxs)
            number_cluters = len(set(cd_pred_labels[remain_idxs]))
            print('==> evaluation (removing {} single clusters). The number of cluster is {}'.format(
                len(single_cluster_idxs), number_cluters))

            print('==> evaluation BFS recluster cluster. The number of cluster is {}'.format(len(merged_clusters)))

        if cfg.save_output:
            print('save predicted edges and scores to {}'.format(ofn_pred))
            np.savez_compressed(ofn_pred,
                                edges=edges,
                                scores=scores,
                                inst_num=inst_num)

            # save merged_pred_labels file
            merged_ofn_meta = os.path.join(cfg.work_dir, 'merged_pred_labels.txt')
            write_meta(merged_ofn_meta, merged_pred_idx2lb, inst_num=inst_num)

            # save cd_pred_labels file
            cd_labels_dir = os.path.join(cfg.work_dir, "{}_labels".format(cd_method))
            if not os.path.exists(cd_labels_dir):
                os.makedirs(cd_labels_dir)

            cd_ofn_meta = os.path.join(cd_labels_dir, '{}_pred_labels.txt'.format(cfg["test_name"]))
            write_meta(cd_ofn_meta, cd_pred_idx2lb, inst_num=inst_num)

    elif cd_method in ["infomap", "Louvain", "Leiden"]:
        if cd_method == "infomap":
            min_sim = cfg.cd_params["min_sim"]
            cd_clusters = Link_cluster_infomap(edges, scores, min_sim=min_sim)
            cd_pred_idx2lb = clusters2labels(cd_clusters)
            cd_pred_labels = intdict2ndarray(cd_pred_idx2lb)
            partition = [int(x) for x in cd_pred_labels]
            print('==> evaluation {} cluster. The number of cluster is {}'.format(cd_method, len(cd_clusters)))


        elif cd_method == "Leiden":
            resolution = cfg.cd_params["resolution"]
            if hasattr(cfg.cd_params, "kneighbour"):
                kneighbour = cfg.cd_params["kneighbour"]
            else:
                kneighbour = None
            VertexPartition = cfg.cd_params["VertexPartition"]
            cd_clusters, partition = Link_cluster_leiden(edges, scores, int(inst_num), min_sim=0,
                                              resolution=resolution, n_iter=10,
                                              VertexPartition=VertexPartition, kneighbour=kneighbour)
            print("the kneighbour of graph is {}".format(kneighbour))
            cd_pred_idx2lb = clusters2labels(cd_clusters)
            cd_pred_labels = intdict2ndarray(cd_pred_idx2lb)
            print('==> evaluation {} cluster. The number of cluster is {}'.format(cd_method, len(cd_clusters)))


        elif cd_method == "Louvain":
            resolution = cfg.cd_params["resolution"]
            VertexPartition = cfg.cd_params["VertexPartition"]
            cd_clusters = Link_cluster_louvain(edges, scores, int(inst_num), min_sim=0, resolution=0.3)
            cd_pred_idx2lb = clusters2labels(cd_clusters)
            cd_pred_labels = intdict2ndarray(cd_pred_idx2lb)
            print('==> evaluation {} cluster. The number of cluster is {}'.format(cd_method, len(cd_clusters)))

        secondary_partition = overlap_community_detection(partition, edges, scores, 0.2)
        if cfg.save_output:
            print('save predicted edges and scores to {}'.format(ofn_pred))
            np.savez_compressed(ofn_pred,
                                edges=edges,
                                scores=scores,
                                inst_num=inst_num)


            # save cd_pred_labels file
            cd_labels_dir = os.path.join(cfg.work_dir, "{}_labels".format(cd_method))
            if not os.path.exists(cd_labels_dir):
                os.makedirs(cd_labels_dir)
            cd_ofn_meta = os.path.join(cd_labels_dir, '{}_pred_labels.txt'.format(cfg["test_name"]))
            with open(cd_ofn_meta, "w") as label_file:
                label_file.writelines([str(x) + "\n" for x in partition])
            # save secondary speaker labels file
            if len(secondary_partition)!=0:
                cd_secondary_labels_dir = os.path.join(cfg.work_dir, "{}_labels/secondary".format(cd_method))
                if not os.path.exists(cd_secondary_labels_dir):
                    os.makedirs(cd_secondary_labels_dir)
                cd_secondary_ofn_meta = os.path.join(cd_secondary_labels_dir, '{}.txt'.format(cfg["test_name"]))
                with open(cd_secondary_ofn_meta, "w") as label_file:
                    label_file.writelines([str(x) + "\n" for x in secondary_partition])
    else:
        raise Exception("The method not defined")
    pass



