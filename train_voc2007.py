import argparse
import os
import time
import uuid
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from backbone.base import Base as BackboneBase
from config.train_voc2007_config import TrainVOC2007Config as Config
from dataset.base import Base as DatasetBase
from extention.lr_scheduler import WarmUpMultiStepLR
from logger import Logger as Log
from model import Model
from roi.pooler import Pooler


def _train(dataset_name: str, backbone_name: str, path_to_data_dir: str, path_to_checkpoints_dir: str, path_to_resuming_checkpoint: Optional[str]):
    
    dataset = DatasetBase.from_name(dataset_name)(path_to_data_dir, DatasetBase.Mode.TRAIN, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                            sampler=DatasetBase.NearestRatioRandomSampler(dataset.image_ratios, num_neighbors=Config.BATCH_SIZE),
                            num_workers=0, collate_fn=DatasetBase.padding_collate_fn, pin_memory=True)

    #为便于调试，num_works置为0

    Log.i('Found {:d} samples'.format(len(dataset)))

    backbone = BackboneBase.from_name(backbone_name)(pretrained=True)
    
    model = nn.DataParallel(
        Model(
            backbone, dataset.num_classes(), pooler_mode=Config.POOLER_MODE,
            anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
            rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N,
            anchor_smooth_l1_loss_beta=Config.ANCHOR_SMOOTH_L1_LOSS_BETA, proposal_smooth_l1_loss_beta=Config.PROPOSAL_SMOOTH_L1_LOSS_BETA
        ).cuda()
    )

    # 便于调试
    # model =  Model(
    #         backbone, dataset.num_classes(), pooler_mode=Config.POOLER_MODE,
    #         anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
    #         rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N,
    #         anchor_smooth_l1_loss_beta=Config.ANCHOR_SMOOTH_L1_LOSS_BETA, proposal_smooth_l1_loss_beta=Config.PROPOSAL_SMOOTH_L1_LOSS_BETA
    #     ).cuda()

    ''' 训练用参数：

        IMAGE_MIN_SIDE: float = 600.0
        IMAGE_MAX_SIDE: float = 1000.0

        ANCHOR_RATIOS: List[Tuple[int, int]] = [(1, 2), (1, 1), (2, 1)]
        ANCHOR_SIZES: List[int] = [128, 256, 512]
        POOLER_MODE: Pooler.Mode = Pooler.Mode.ALIGN

        RPN_PRE_NMS_TOP_N: int = 12000
        RPN_POST_NMS_TOP_N: int = 2000

        ANCHOR_SMOOTH_L1_LOSS_BETA: float = 1.0
        PROPOSAL_SMOOTH_L1_LOSS_BETA: float = 1.0

        BATCH_SIZE: int = 1
        LEARNING_RATE: float = 0.001
        MOMENTUM: float = 0.9
        WEIGHT_DECAY: float = 0.0005
        STEP_LR_SIZES: List[int] = [50000, 70000]
        STEP_LR_GAMMA: float = 0.1
        WARM_UP_FACTOR: float = 0.3333
        WARM_UP_NUM_ITERS: int = 500

        NUM_STEPS_TO_DISPLAY: int = 20
        NUM_STEPS_TO_SNAPSHOT: int = 10000
        NUM_STEPS_TO_FINISH: int = 90000
    '''
    #动量的意义：
    #1.降低病态条件数带来的振荡
    #2.减少随机梯度带来的方差（权值的衰减也有这个用处）

    #优化算法的两种衰减：
    #1. 权值的衰减：表现为在总的损失函数后面再加上权值的L2范数
    #2.学习率的衰减：表现为通过学习率调节器以不同策略随着学习步增加，对学习率进行衰减调节

    #optimizer = optim.Adam(model.parameters())

    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE,
                          momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    scheduler = WarmUpMultiStepLR(optimizer, milestones=Config.STEP_LR_SIZES, gamma=Config.STEP_LR_GAMMA,
                                  factor=Config.WARM_UP_FACTOR, num_iters=Config.WARM_UP_NUM_ITERS)

    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    summary_writer = SummaryWriter(os.path.join(path_to_checkpoints_dir, 'summaries'))
    should_stop = False

    num_steps_to_display = Config.NUM_STEPS_TO_DISPLAY
    num_steps_to_snapshot = Config.NUM_STEPS_TO_SNAPSHOT
    num_steps_to_finish = Config.NUM_STEPS_TO_FINISH

    if path_to_resuming_checkpoint is not None:
        step = model.module.load(path_to_resuming_checkpoint, optimizer, scheduler)
        Log.i(f'Model has been restored from file: {path_to_resuming_checkpoint}')

    device_count = torch.cuda.device_count()
    #BATCH_SIZE默认是1
    assert Config.BATCH_SIZE % device_count == 0, 'The batch size is not divisible by the device count'
    Log.i('Start training with {:d} GPUs ({:d} batches per GPU)'.format(torch.cuda.device_count(),
                                                                        Config.BATCH_SIZE // torch.cuda.device_count()))

    while not should_stop:
        for _, (_, image_batch, _, bboxes_batch, labels_batch) in enumerate(dataloader):
            #训练使用的数据集采用voc2007
            batch_size = image_batch.shape[0]#(1,)
            image_batch = image_batch.cuda()#(1,3,h,w)
            bboxes_batch = bboxes_batch.cuda()#(1,gt_n,4)
            labels_batch = labels_batch.cuda()#(1,gt_n)

            anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = \
                model.train().forward(image_batch, bboxes_batch, labels_batch)
            #rpn的损失
            anchor_objectness_loss = anchor_objectness_losses.mean()
            anchor_transformer_loss = anchor_transformer_losses.mean()
            #detection的损失
            proposal_class_loss = proposal_class_losses.mean()
            proposal_transformer_loss = proposal_transformer_losses.mean()
            loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            summary_writer.add_scalar('train/anchor_objectness_loss', anchor_objectness_loss.item(), step)
            summary_writer.add_scalar('train/anchor_transformer_loss', anchor_transformer_loss.item(), step)
            summary_writer.add_scalar('train/proposal_class_loss', proposal_class_loss.item(), step)
            summary_writer.add_scalar('train/proposal_transformer_loss', proposal_transformer_loss.item(), step)
            summary_writer.add_scalar('train/loss', loss.item(), step)
            step += 1

            if step == num_steps_to_finish:
                should_stop = True

            if step % num_steps_to_display == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                samples_per_sec = batch_size * steps_per_sec
                eta = (num_steps_to_finish - step) / steps_per_sec / 3600
                avg_loss = sum(losses) / len(losses)
                lr = scheduler.get_lr()[0]
                #lr = optimizer.param_groups[0]['lr']
                Log.i(f'[Step {step}] Avg. Loss = {avg_loss:.6f}, Learning Rate = {lr} ({samples_per_sec:.2f} samples/sec; ETA {eta:.1f} hrs)')

            #test
            if step == 10:
                path_to_checkpoint = model.module.save(path_to_checkpoints_dir, step, optimizer, scheduler)
                #path_to_checkpoint = model.module.save(path_to_checkpoints_dir, step, optimizer)

                Log.i(f'Model has been saved to {path_to_checkpoint}')

            if step % num_steps_to_snapshot == 0 or should_stop:
                path_to_checkpoint = model.module.save(path_to_checkpoints_dir, step, optimizer, scheduler)
                #path_to_checkpoint = model.module.save(path_to_checkpoints_dir, step, optimizer)

                Log.i(f'Model has been saved to {path_to_checkpoint}')

            if should_stop:
                break

    Log.i('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        # 'voc2007', 'coco2017'等
        parser.add_argument('-s', '--dataset', type=str, default='voc2007',choices=DatasetBase.OPTIONS, help='name of dataset')
        # OPTIONS = ['resnet18', 'resnet50', 'resnet101']
        parser.add_argument('-b', '--backbone', type=str, default='resnet101',choices=BackboneBase.OPTIONS, help='name of backbone model')
        parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to data directory')
        parser.add_argument('-o', '--outputs_dir', type=str, default='./outputs', help='path to outputs directory')
        parser.add_argument('-r', '--resume_checkpoint', type=str, help='path to resuming checkpoint')
        parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
        parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
        parser.add_argument('--anchor_smooth_l1_loss_beta', type=float, help='default: {:g}'.format(Config.ANCHOR_SMOOTH_L1_LOSS_BETA))
        parser.add_argument('--proposal_smooth_l1_loss_beta', type=float, help='default: {:g}'.format(Config.PROPOSAL_SMOOTH_L1_LOSS_BETA))
        parser.add_argument('--batch_size', type=int, help='default: {:g}'.format(Config.BATCH_SIZE))
        parser.add_argument('--learning_rate', type=float, help='default: {:g}'.format(Config.LEARNING_RATE))
        parser.add_argument('--momentum', type=float, help='default: {:g}'.format(Config.MOMENTUM))
        parser.add_argument('--weight_decay', type=float, help='default: {:g}'.format(Config.WEIGHT_DECAY))
        parser.add_argument('--step_lr_sizes', type=str, help='default: {!s}'.format(Config.STEP_LR_SIZES))
        parser.add_argument('--step_lr_gamma', type=float, help='default: {:g}'.format(Config.STEP_LR_GAMMA))
        parser.add_argument('--warm_up_factor', type=float, help='default: {:g}'.format(Config.WARM_UP_FACTOR))
        parser.add_argument('--warm_up_num_iters', type=int, help='default: {:d}'.format(Config.WARM_UP_NUM_ITERS))
        parser.add_argument('--num_steps_to_display', type=int, help='default: {:d}'.format(Config.NUM_STEPS_TO_DISPLAY))
        parser.add_argument('--num_steps_to_snapshot', type=int, help='default: {:d}'.format(Config.NUM_STEPS_TO_SNAPSHOT))
        parser.add_argument('--num_steps_to_finish', type=int, help='default: {:d}'.format(Config.NUM_STEPS_TO_FINISH))
        args = parser.parse_args()

        dataset_name = args.dataset
        backbone_name = args.backbone
        path_to_data_dir = args.data_dir
        path_to_outputs_dir = args.outputs_dir
        path_to_resuming_checkpoint = args.resume_checkpoint

        path_to_checkpoints_dir = os.path.join(path_to_outputs_dir, 'checkpoints-{:s}-{:s}-{:s}-{:s}'.format(
            time.strftime('%Y%m%d%H%M%S'), dataset_name, backbone_name, str(uuid.uuid4()).split('-')[0]))
        os.makedirs(path_to_checkpoints_dir)

        Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
                     anchor_ratios=args.anchor_ratios, anchor_sizes=args.anchor_sizes, pooler_mode=args.pooler_mode,
                     rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n,
                     anchor_smooth_l1_loss_beta=args.anchor_smooth_l1_loss_beta, proposal_smooth_l1_loss_beta=args.proposal_smooth_l1_loss_beta,
                     batch_size=args.batch_size, learning_rate=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                     step_lr_sizes=args.step_lr_sizes, step_lr_gamma=args.step_lr_gamma,
                     warm_up_factor=args.warm_up_factor, warm_up_num_iters=args.warm_up_num_iters,
                     num_steps_to_display=args.num_steps_to_display, num_steps_to_snapshot=args.num_steps_to_snapshot, num_steps_to_finish=args.num_steps_to_finish)

        Log.initialize(os.path.join(path_to_checkpoints_dir, 'train.log'))
        Log.i('Arguments:')
        for k, v in vars(args).items():
            Log.i(f'\t{k} = {v}')
        Log.i(Config.describe())

        _train(dataset_name, backbone_name, path_to_data_dir, path_to_checkpoints_dir, path_to_resuming_checkpoint)

    main()
