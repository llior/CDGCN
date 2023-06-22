#copyright wangjie xmuspeech 9.20
#Version 2.0
# add graph2adj function

import igraph as ig
import leidenalg
import louvain
import numpy as np
import scipy.sparse as sp

def create_graph(edges, scores, inst_num, min_sim=0):
    g = ig.Graph()
    g.add_vertices(inst_num)
    g.add_edges(edges)
    g.es['weight'] = scores

    return g
# edges.shape=(n*k, 2), scores.shape=(n*k), where n is the number of samples, k is the K-neighbour of KNN
# the bigger the resolution, the more the number of cluster
# VertexPartition including: CPMVertexPartition, RBConfigurationVertexPartition, RBERVertexPartition
def Link_cluster_leiden(edges, scores, inst_num, min_sim=0, resolution=0.9, n_iter=10, VertexPartition="CPMVertexPartition", kneighbour=None):
    if kneighbour==None:
        g = create_graph(edges, scores, inst_num, min_sim)
    else:
        g = k_neighbor_Graph(edges, scores, k=kneighbour)
    if VertexPartition=="ModularityVertexPartition":
        print("perform ModularityVertexPartition")
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights="weight", n_iterations=n_iter)
    else:
        partition = leidenalg.find_partition(g, eval("leidenalg.{}".format(VertexPartition)),
                                            resolution_parameter=resolution,
                                            weights='weight',
                                            n_iterations=n_iter)
    clusters_dict = dict()
    for id, pred_label in enumerate(partition._membership):
        if pred_label not in clusters_dict:
            clusters_dict[pred_label] = [id]
        else:
            clusters_dict[pred_label].append(id)
    return list(clusters_dict.values()), partition._membership

def Link_cluster_louvain(edges, scores, inst_num, min_sim=0, resolution=0.9):

    g = create_graph(edges, scores, inst_num, min_sim)

    partition = louvain.find_partition(g, louvain.CPMVertexPartition,
                                       resolution_parameter=resolution,
                                       weights='weight')
    clusters_dict = dict()
    for id, pred_label in enumerate(partition._membership):
        if pred_label not in clusters_dict:
            clusters_dict[pred_label] = [id]
        else:
            clusters_dict[pred_label].append(id)
    return list(clusters_dict.values())

def get_second_spk(first_label, node_stat):
    secondary_label = 100
    secondary_weighted_degree = 0
    for label, weighted_degree in node_stat.items():
        if label!=first_label and weighted_degree>secondary_weighted_degree:

            secondary_weighted_degree = weighted_degree
            secondary_label = label
    return secondary_label

#input the result of initial clustering
def overlap_community_detection(partition, edges, scores, threshold):
    overlap_rates = np.zeros(len(partition))
    nebr_lab_statistics = [dict() for _ in range(len(partition))]
    nebr_degree_statistics = [dict() for _ in range(len(partition))]
    secondary_partition = np.zeros(len(partition), dtype=np.int)
    for edge,score in zip(edges, scores):
        nodeLabel0 = partition[edge[0]]
        nodeLabel1 = partition[edge[1]]

        nebr_lab_statistics[edge[0]][nodeLabel1] = nebr_lab_statistics[edge[0]].get(nodeLabel1, 0)+score
        nebr_lab_statistics[edge[1]][nodeLabel0] = nebr_lab_statistics[edge[1]].get(nodeLabel0, 0)+score
        nebr_degree_statistics[edge[0]][nodeLabel1] = nebr_degree_statistics[edge[0]].get(nodeLabel1, 0)+1
        nebr_degree_statistics[edge[1]][nodeLabel0] = nebr_degree_statistics[edge[1]].get(nodeLabel0, 0)+1
    #normalize the degree of node
    # for node_id in range(len(nebr_lab_statistics)):
    #     for nebr_lab, degree in nebr_lab_statistics[node_id].items():
    #         nebr_lab_statistics[node_id][nebr_lab] /= nebr_degree_statistics[node_id][nebr_lab]

    for node_id, node_stat in enumerate(nebr_lab_statistics):
        secondary_partition[node_id] = get_second_spk(partition[node_id], node_stat)

    return secondary_partition

# Thresolds affinity matrix to leave p maximum non-zero elements in each row
def Threshold(A, p):
    N = A.shape[0]
    Ap = np.zeros((N,N))
    # avoid list index out of range
    if p >= N:
        p = N-1
    for i in range(N):
        thr = sorted(A[i,:], reverse=True)[p]
        Ap[i,A[i,:]>thr] = A[i,A[i,:]>thr]
    return Ap

def Graph_to_Adjacency(edges, scores):
    max_nodeId = max([ x for y in edges for x in y])+1
    adj = np.zeros((max_nodeId, max_nodeId))
    for edge,score in zip(edges, scores):
        adj[edge[0]][edge[1]] = score
    return adj

def k_neighbor_Graph(edges, scores, k=20):
    adj = Graph_to_Adjacency(edges, scores)
    knn_adj = Threshold(adj, k)
    knn_G = ig.Graph.Weighted_Adjacency(knn_adj)

    return knn_G
