from .cluster_dataset import ClusterDataset, ClusterOnlineDataset
from .build_dataloader import build_dataloader


def build_dataset(cfg):
    if cfg.get("online", False):
        return ClusterOnlineDataset(cfg)
    else:
        return ClusterDataset(cfg)
