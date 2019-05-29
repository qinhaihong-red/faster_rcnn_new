from typing import Tuple, List, Optional, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from bbox import BBox
from extention.functional import beta_smooth_l1_loss
from support.layer.nms import nms


class RegionProposalNetwork(nn.Module):
    #以下参数用于infer:
    #num_features_out=1024
    #anchor_ratios=[(1, 2), (1, 1), (2, 1)]
    #anchor_sizes=[64, 128, 256, 512]
    #rpn_pre_nms_top_n=6000
    #rpn_post_nms_top_n=1000
    #anchor_smooth_l1_loss_beta=None
    def __init__(self, num_features_out: int, anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int],
                 pre_nms_top_n: int, post_nms_top_n: int, anchor_smooth_l1_loss_beta: float):
        super().__init__()

        self._anchor_ratios = anchor_ratios
        self._anchor_sizes = anchor_sizes
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self._anchor_smooth_l1_loss_beta = anchor_smooth_l1_loss_beta

        num_anchor_ratios = len(self._anchor_ratios)#3
        num_anchor_sizes = len(self._anchor_sizes)#3. 对于infer是4
        num_anchors = num_anchor_ratios * num_anchor_sizes#9. 对于infer是12
        
        #先把输入的features进行卷积：
        #Conv@features:(bn,1024,anchors_y,anchors_x)-->(bn,512,anchors_y,anchors_x)@ksp(3,1,1)
        self._features = nn.Sequential(
            nn.Conv2d(in_channels=num_features_out, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        #再分别对每个anchor卷积出2个objectness和4个预测坐标：
        #per_anchors=num_anchors，每个点上的anchor数目.
        #Conv@anchor_obj:(bn,512,h,w)-->(bn,per_anchors * 2,h,w)@ksp(1,1,0)
        #Conv@ahcor_coor:(bn,512,h,w)-->(bn,per_anchors * 4,h,w)@ksp(1,1,0)
        self._anchor_objectness = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self._anchor_transformer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)
    
    def forward(self, features: Tensor,
                anchor_bboxes: Optional[Tensor] = None, gt_bboxes_batch: Optional[Tensor] = None,
                image_width: Optional[int]=None, image_height: Optional[int]=None) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        
        #输入的features是h,w是原图的h/16,w/16，也就是anchors_y和anchors_x
        batch_size = features.shape[0]

        features = self._features(features)#(bn,1024,anchors_y,anchors_x)->(bn,512,anchors_y,anchors_x)@ksp(3,1,1)
        anchor_objectnesses = self._anchor_objectness(features)# (bn,18,anchors_y,anchors_x)@ksp(1,1,0) 这里假设每个点的anchor数为9
        anchor_transformers = self._anchor_transformer(features)#(bn,36,anchors_y,anchors_x)@ksp(1,1,0)

        #输出所有anchor的obj(x2)和coor(x4).
        #这个处理类似于yolo的(1,10647,85),但是yolo由于输入图像的尺寸固定，因此输出也是固定的.
        #yolo的obj和预测坐标，是用一个向量表示的，rcnn这里是分开表示的.
        #这里由于图像尺寸不固定，anchor的obj和coor输出也不固定.
        #anchors_n=anchors_x * anchors_y * 9
        anchor_objectnesses = anchor_objectnesses.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)#(bn,anchors_n,2)
        anchor_transformers = anchor_transformers.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)#(bn,anchors_n,4)

        #self.training是nn.Module的成员变量.
        #对于infer，在前面已经置model为eval,因此会改变training为False.
        if not self.training:
            return anchor_objectnesses, anchor_transformers
        else:
            # remove cross-boundary
            # NOTE: The length of `inside_indices` is guaranteed to be a multiple of `anchor_bboxes.shape[0]` as each batch in `anchor_bboxes` is the same
            inside_indices = BBox.inside(anchor_bboxes, left=0, top=0, right=image_width, bottom=image_height).nonzero().unbind(dim=1)
            inside_anchor_bboxes = anchor_bboxes[inside_indices].view(batch_size, -1, anchor_bboxes.shape[2])
            inside_anchor_objectnesses = anchor_objectnesses[inside_indices].view(batch_size, -1, anchor_objectnesses.shape[2])
            inside_anchor_transformers = anchor_transformers[inside_indices].view(batch_size, -1, anchor_transformers.shape[2])

            # find labels for each `anchor_bboxes`
            labels = torch.full((batch_size, inside_anchor_bboxes.shape[1]), -1, dtype=torch.long, device=inside_anchor_bboxes.device)
            ious = BBox.iou(inside_anchor_bboxes, gt_bboxes_batch)
            anchor_max_ious, anchor_assignments = ious.max(dim=2)
            gt_max_ious, gt_assignments = ious.max(dim=1)
            anchor_additions = ((ious > 0) & (ious == gt_max_ious.unsqueeze(dim=1))).nonzero()[:, :2].unbind(dim=1)
            labels[anchor_max_ious < 0.3] = 0
            labels[anchor_additions] = 1
            labels[anchor_max_ious >= 0.7] = 1

            # select 256 x `batch_size` samples
            fg_indices = (labels == 1).nonzero()
            bg_indices = (labels == 0).nonzero()
            fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 128 * batch_size)]]
            bg_indices = bg_indices[torch.randperm(len(bg_indices))[:256 * batch_size - len(fg_indices)]]
            selected_indices = torch.cat([fg_indices, bg_indices], dim=0)
            selected_indices = selected_indices[torch.randperm(len(selected_indices))].unbind(dim=1)

            inside_anchor_bboxes = inside_anchor_bboxes[selected_indices]
            gt_bboxes = gt_bboxes_batch[selected_indices[0], anchor_assignments[selected_indices]]
            gt_anchor_objectnesses = labels[selected_indices]
            gt_anchor_transformers = BBox.calc_transformer(inside_anchor_bboxes, gt_bboxes)
            batch_indices = selected_indices[0]

            anchor_objectness_losses, anchor_transformer_losses = self.loss(inside_anchor_objectnesses[selected_indices],
                                                                            inside_anchor_transformers[selected_indices],
                                                                            gt_anchor_objectnesses,
                                                                            gt_anchor_transformers,
                                                                            batch_size, batch_indices)

            return anchor_objectnesses, anchor_transformers, anchor_objectness_losses, anchor_transformer_losses

    def loss(self, anchor_objectnesses: Tensor, anchor_transformers: Tensor,
             gt_anchor_objectnesses: Tensor, gt_anchor_transformers: Tensor,
             batch_size: int, batch_indices: Tensor) -> Tuple[Tensor, Tensor]:
        cross_entropies = torch.empty(batch_size, dtype=torch.float, device=anchor_objectnesses.device)
        smooth_l1_losses = torch.empty(batch_size, dtype=torch.float, device=anchor_transformers.device)

        for batch_index in range(batch_size):
            selected_indices = (batch_indices == batch_index).nonzero().view(-1)

            cross_entropy = F.cross_entropy(input=anchor_objectnesses[selected_indices],
                                            target=gt_anchor_objectnesses[selected_indices])

            fg_indices = gt_anchor_objectnesses[selected_indices].nonzero().view(-1)
            smooth_l1_loss = beta_smooth_l1_loss(input=anchor_transformers[selected_indices][fg_indices],
                                                 target=gt_anchor_transformers[selected_indices][fg_indices],
                                                 beta=self._anchor_smooth_l1_loss_beta)

            cross_entropies[batch_index] = cross_entropy
            smooth_l1_losses[batch_index] = smooth_l1_loss

        return cross_entropies, smooth_l1_losses


    #这是一个相当独立的函数，根据特征图生成anchors：(anchors_x * anchors_y * 9,4)
    def generate_anchors(self, image_width: int, image_height: int, num_x_anchors: int, num_y_anchors: int) -> Tensor:
        #image_width是原图经过min/max缩放过的，即至少有一边符合最大边或最小边
        #(anchors_x,anchors_y)=(img_width/16,img_height/16)
        #anchor坐标是基于原图的
        center_ys = np.linspace(start=0, stop=image_height, num=num_y_anchors + 2)[1:-1]#(anchors_y,)
        center_xs = np.linspace(start=0, stop=image_width, num=num_x_anchors + 2)[1:-1]# (anchors_x,)
        ratios = np.array(self._anchor_ratios)
        ratios = ratios[:, 0] / ratios[:, 1]#[0.5,1,2]@(3,)
        sizes = np.array(self._anchor_sizes)#[128,256,512]@(3,) #假设3个size. infer中实际用到了4个size，多了一个64.

        # NOTE: it's important to let `center_ys` be the major index (i.e., move horizontally and then vertically) for consistency with 2D convolution
        # giving the string 'ij' returns a meshgrid with matrix indexing, i.e., with shape (#center_ys, #center_xs, #ratios)
        center_ys, center_xs, ratios, sizes = np.meshgrid(center_ys, center_xs, ratios, sizes, indexing='ij')#这个函数可谓画龙点睛
        # 返回：(anchors_y,anchors_x,3,3) ,如果indexing为默认，即indexing='xy'，则返回数据的形状为:(num_anchor_x，num_anchor_y，3，3)

        #anchors_n=anchors_x * anchors_y * 9
        center_ys = center_ys.reshape(-1)#(anchors_n,)
        center_xs = center_xs.reshape(-1)#(anchors_n,)
        ratios = ratios.reshape(-1)#(anchors_n,)
        sizes = sizes.reshape(-1)#(anchors_n,)

        widths = sizes * np.sqrt(1 / ratios)#(anchors_n,)
        heights = sizes * np.sqrt(ratios)#(anchors_n,)

        center_based_anchor_bboxes = np.stack((center_xs, center_ys, widths, heights), axis=1)#(anchors_n,4)
        center_based_anchor_bboxes = torch.from_numpy(center_based_anchor_bboxes).float()#(anchors_n,4)
        anchor_bboxes = BBox.from_center_base(center_based_anchor_bboxes)#把 中心宽高 转换为 左上右下 的形式:(anchors_n,4)

        return anchor_bboxes

    def generate_proposals(self, anchor_bboxes: Tensor, objectnesses: Tensor, transformers: Tensor, image_width: int, image_height: int) -> Tensor:
        #anchor_bboxes@(bn,anchors_n,4)
        #objectnesses@(bn,anchors_n,2)
        #transformers@(bn,anchors_n,4)
        #anchors_n=anchors_x * anchors_y * 9

        batch_size = anchor_bboxes.shape[0]

        proposal_bboxes = BBox.apply_transformer(anchor_bboxes, transformers)#对anchor进行变换
        proposal_bboxes = BBox.clip(proposal_bboxes, left=0, top=0, right=image_width, bottom=image_height)#(bn,anchors_n,4)
        proposal_probs = F.softmax(objectnesses[:, :, 1], dim=-1)#(bn,anchors_n)

        _, sorted_indices = torch.sort(proposal_probs, dim=-1, descending=True)#(bn,anchors_n)
        nms_proposal_bboxes_batch = []

        for batch_index in range(batch_size):
            sorted_bboxes = proposal_bboxes[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]#(pre_nms_n,4)
            sorted_probs  =  proposal_probs[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]#(pre_nms_n,)
            threshold = 0.7
            kept_indices = nms(sorted_bboxes, sorted_probs, threshold)#(nms_n,)
            nms_bboxes = sorted_bboxes[kept_indices][:self._post_nms_top_n]#(post_nms_n,4) , post_nms_n<=_post_nms_top_n
            nms_proposal_bboxes_batch.append(nms_bboxes)

        #从一批图像中，找到proposal_boxes最多的，记录数量为max_nms_n
        #其它图像的proposal_boxes要对齐max_nms_n(dim0方向，即增加行数)
        #就是说，一批内每幅图像的proposal_boxes都要相等
        max_nms_proposal_bboxes_length = max([len(it) for it in nms_proposal_bboxes_batch])
        padded_proposal_bboxes = []

        for nms_proposal_bboxes in nms_proposal_bboxes_batch:
            padded_proposal_bboxes.append(
                torch.cat([
                    nms_proposal_bboxes,
                    torch.zeros(max_nms_proposal_bboxes_length - len(nms_proposal_bboxes), 4).to(nms_proposal_bboxes)
                ])# 增加为每幅图像的proposal_boxes增加 delta 行，每行4列 . 其中 delta = max_nms_proposal_bboxes_length - len(nms_proposal_bboxes) 
            )

        padded_proposal_bboxes = torch.stack(padded_proposal_bboxes, dim=0)#(bn,max_nms_n,4)
        return padded_proposal_bboxes
