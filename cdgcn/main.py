from __future__ import division

import os
import torch
import argparse

from mmcv import Config

from utils import (create_logger, set_random_seed, rm_suffix,
                   mkdir_if_no_exists)

from cdgcn.models import build_model
from cdgcn import build_handler


def parse_args():
    parser = argparse.ArgumentParser(
        description='Linkage-based Face Clustering via GCN')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--phase', choices=['test', 'train'], default='test')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--resume_from',
                        default=None,
                        help='the checkpoint file to resume from')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='node id of mulit-gpu')
    parser.add_argument('--diarization',
                        action="store_true",
                        help='perform diarization for dataset')
    parser.add_argument('--test_name', type=str, help='if None we will use cfg[test_name] in config file')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus(only applicable to non-distributed training)')
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--save_output', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg["local_rank"]=args.local_rank
    # init DDP training(modify by wangjie
    if args.distributed:
        import torch.distributed as dist
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend="nccl")
    # Add option --testname for batch processing(modify by wangjie
    if args.diarization:
        cfg['diarization']=True
    if args.test_name:
        cfg['test_name'] = args.test_name
        cfg['test_data']['feat_path'] = os.path.join(cfg['prefix'], 'features', '{}.bin'.format(cfg['test_name']))
        #cfg['test_data']['label_path'] = os.path.join(cfg['prefix'], 'labels', '{}.meta'.format(cfg['test_name']))
        cfg['test_data']['knn_graph_path'] = os.path.join(cfg['prefix'], 'knns', cfg['test_name'],
                                                          'faiss_k_{}.npz'.format(cfg['knn']))
    # set cuda
    cfg.cuda = not args.no_cuda and torch.cuda.is_available()
    # set cudnn_benchmark & cudnn_deterministic
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('cudnn_deterministic', False):
        torch.backends.cudnn.deterministic = True

    # update configs according to args
    if not hasattr(cfg, 'work_dir'):
        if args.work_dir is not None:
            cfg.work_dir = args.work_dir
        else:
            cfg_name = rm_suffix(os.path.basename(args.config))
            cfg.work_dir = os.path.join('./data/work_dir', cfg_name)
    mkdir_if_no_exists(cfg.work_dir, is_folder=True)

    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from

    cfg.gpus = args.gpus
    cfg.distributed = args.distributed
    cfg.save_output = args.save_output
    cfg.force = args.force

    logger = create_logger()

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_model(cfg.model['type'], **cfg.model['kwargs'])
    handler = build_handler(args.phase)
    handler(model, cfg, logger)

if __name__ == '__main__':
    main()
