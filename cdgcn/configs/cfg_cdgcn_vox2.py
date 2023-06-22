import os.path as osp

# data locations
work_dir = './xv_data/resnet34se/work_dir/cfg_cdgcn_vox2'
prefix = './xv_data/resnet34se'
train_name = 'vox2_resnet34se'
test_name = 'dihard_resnet34se'
knn = 300
knn_method = 'faiss_gpu'
cd_params = dict(
    method = "Leiden",
    VertexPartition = "RBConfigurationVertexPartition",
    resolution=0.6,
)

train_data = dict(
    feat_path=osp.join(prefix, 'features', '{}.bin'.format(train_name)),
    label_path=osp.join(prefix, 'labels', '{}.meta'.format(train_name)),
    knn_graph_path=osp.join(prefix, 'knns', train_name,
                            '{}_k_{}.npz'.format(knn_method, knn)),
    k_at_hop=[300, 10],
    active_connection=20,
    is_norm_feat=True,
    is_sort_knns=True,
)

test_data = dict(
    feat_path=osp.join(prefix, 'features', '{}.bin'.format(test_name)),
    knn_graph_path=osp.join(prefix, 'knns', test_name,
                            'faiss_k_{}.npz'.format(knn)),
    k_at_hop=[300, 10],
    active_connection=20,
    is_norm_feat=True,
    is_sort_knns=True,
    is_test=True,
)

# model
model = dict(type='cdgcn', kwargs=dict(feature_dim=256))

# training args
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer_config = {}

lr_config = dict(
    policy='step',
    step=[1, 2, 3],
)

batch_size_per_gpu = 2
total_epochs = 4
workflow = [('train', 1)]

# testing args
max_sz = 500 #不影响
step = 0.5
pool = 'max'
max_iter = 200

metrics = ['pairwise', 'bcubed', 'nmi']
metrics = ['pairwise']

# misc
workers_per_gpu = 2  # Worker to pre-fetch data for each single GPU

checkpoint_config = dict(interval=1)

log_level = 'INFO'
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'),
])
