from typing import Tuple, List, Optional, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from bbox import BBox
from extention.functional import beta_smooth_l1_loss
from support.layer.nms import nms


class RegionProposalNetwork(nn.Module):
    #以下具体的参数用于基于coco2017训练的infer:
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
        #Conv@features:(bn,1024,ga_y,ga_x)-->(bn,512,ga_y,ga_x)@ksp(3,1,1)
        self._features = nn.Sequential(
            nn.Conv2d(in_channels=num_features_out, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        #再分别对每个anchor卷积出2个objectness和4个预测坐标：
        #per_anchors=num_anchors，每个点上的anchor数目.
        #Conv@anchor_obj:(bn,512,h,w)-->(bn,pa_n * 2,h,w)@ksp(1,1,0)
        #Conv@ahcor_coor:(bn,512,h,w)-->(bn,pa_n * 4,h,w)@ksp(1,1,0)
        self._anchor_objectness = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self._anchor_transformer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)
    
    # anchor_bboxes@(b,ga_n,4) -->ab@(b,ga_n,4)
    # gt_bboxes_batch@(b,gt_n,4)对应图像的gt_box-->gb@(b,gt_n,4)
    def forward(self, features: Tensor,
                anchor_bboxes: Optional[Tensor] = None, gt_bboxes_batch: Optional[Tensor] = None,
                image_width: Optional[int]=None, image_height: Optional[int]=None) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        
        #输入的features是h,w是原图的h/16,w/16，也就是anchors_y和anchors_x
        #下面用 ga_y,ga_x,ga_n 指代 anchors_y, anchors_x, anchors_n
        batch_size = features.shape[0]

        features = self._features(features)#(bn,1024,ga_y,ga_x)->(bn,512,ga_y,ga_x)@ksp(3,1,1)
        anchor_objectnesses = self._anchor_objectness(features)# (bn, 18,ga_y,ga_x)@ksp(1,1,0) 这里假设每个点的anchor数为9
        anchor_transformers = self._anchor_transformer(features)#(bn, 36,ga_y,ga_x)@ksp(1,1,0)

        #输出所有anchor的obj(x2)和coor(x4).
        #这个处理类似于yolo的(1,10647,85),但是yolo由于输入图像的尺寸固定，因此输出也是固定的.
        #yolo的obj和预测坐标，是用一个向量表示的，rcnn这里是分开表示的.
        #这里由于图像尺寸不固定，anchor的obj和coor输出也不固定.
        #ga_n=anchors_n=ga_x * ga_y * 9
        anchor_objectnesses = anchor_objectnesses.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)#ao@(bn,ga_n,2)
        anchor_transformers = anchor_transformers.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)#at@(bn,ga_n,4)

        #self.training是nn.Module的成员变量.
        #对于infer，在前面已经置model为eval,因此会改变training为False.
        if not self.training:
            return anchor_objectnesses, anchor_transformers
        else:
            #NOTE 总的训练流程是：ious->labels->bg/fg->selected_indices

            # remove cross-boundary
            # NOTE: The length of `inside_indices` is guaranteed to be a multiple of `anchor_bboxes.shape[0]` as each batch in `anchor_bboxes` is the same
            ## 1.先过滤掉处于边缘的ga
            inside_indices = BBox.inside(anchor_bboxes, left=0, top=0, right=image_width, bottom=image_height).nonzero().unbind(dim=1)
            inside_anchor_bboxes = anchor_bboxes[inside_indices].view(batch_size, -1, anchor_bboxes.shape[2])#ab@(bn,ga_in,4)
            inside_anchor_objectnesses = anchor_objectnesses[inside_indices].view(batch_size, -1, anchor_objectnesses.shape[2])#ao@(bn,ga_in,2)
            inside_anchor_transformers = anchor_transformers[inside_indices].view(batch_size, -1, anchor_transformers.shape[2])#at@(bn,ga_in,4)

            # find labels for each `anchor_bboxes`
            ## 2.为所有的ga_in anchor分配后景/前景label（0或1），默认为-1（忽略）
            #注意labels的第一列是batch_size，通常为1
            #labels@(bn,ga_in)
            labels = torch.full((batch_size, inside_anchor_bboxes.shape[1]), -1, dtype=torch.long, device=inside_anchor_bboxes.device)
            ious = BBox.iou(inside_anchor_bboxes, gt_bboxes_batch)#(bn,ga_in,gt_n) ：!!!NOTE iou在rpn中只用到一次，在detection中也只用到一次
            # NOTE 计算每个anchor与所有gt_boxes的iou是关键!
            # 对于返回的ious，先忽略第一列（表示批内的编号，对于batch_size为1的情况，这个编号总是为0），
            # 则返回的ious的每一行表示每个生成anchor与所有gt_boxes(每列表示1个gt_box)的ious
            # 对于anchor与gt_box不相交的，则这个iou为0. 总之，可以为每个anchor生成与所有gt_boxex对应的ious.
            
            anchor_max_ious, anchor_assignments = ious.max(dim=2)#(bn,ga_in) 
            gt_max_ious,     gt_assignments     = ious.max(dim=1)#(bn,gt_n)
            
            #下面的计算是要找出label必须置1的anchor的索引.这是从gt_box的角度考虑.
            selector = (ious == gt_max_ious.unsqueeze(dim=1))#(bn,ga_in,gt_n)
            selector = (ious > 0) & selector#(bn,ga_in,gt_n)
            selector = selector.nonzero()#       (nonzero_num,3)，比如(5,3)
            selector = selector[:, :2]#只取前两列，(nonzero_num,2)，比如(5,2)
            anchor_additions=selector.unbind(dim=1)#使用unbind把selector分解为具有2个元素的tuple,每个元素对应上面的1列,这样主要是为了下面的多维数组索引

            #anchor_additions2 = ((ious > 0) & (ious == gt_max_ious.unsqueeze(dim=1))).nonzero()[:, :2].unbind(dim=1)
            result=(selector == ((ious > 0) & (ious == gt_max_ious.unsqueeze(dim=1))).nonzero()[:, :2]).sum()
  
            #NOTE 根据计算出ious确定anchor的labels,再由labels确定前景(1)/后景(0),再从前景/后景中选择一批数据训练:
            # ious@(ab&gb)-->anchors_labels@(0,1,-1)-->fg(1)&bg(0)-->selected_indices
            
            #注意下面的labels设置并没有覆盖所有的labels，没有覆盖到的，默认是初始化时候的-1 ：
            # [0,0.3)置0 
            # [0.3,0.7)置-1 
            # [0.7,1]置1      
            labels[anchor_max_ious < 0.3] = 0#labels置0的条件. 并不是除了置1的就是置0，还有默认的置-1的
            labels[anchor_additions] = 1#anchor_additions表示满足labels必须置1的多维索引
            labels[anchor_max_ious >= 0.7] = 1

            # select `batch_size` x 256 samples
            ## 3.选择 batch_size x 256个样本训练 
            fg_indices = (labels == 1).nonzero()#(fg_n,2) 注意第一列表示批.第二列对应ga_in索引.
            bg_indices = (labels == 0).nonzero()#(bg_n,2)
            len_fg=len(fg_indices)#(fg_n,)
            len_bg=len(bg_indices)#(bg_n,)
            randperm_fg=torch.randperm(len_fg)#随机排列所有的前景(fg_n,)
            randperm_bg=torch.randperm(len_bg)#随机排列所有的后景(bg_n,)

            fg_indices = fg_indices[randperm_fg[:min(len_fg, 128 * batch_size)]]#选出的前景(sfg_n,2)
            bg_indices = bg_indices[randperm_bg[:256 * batch_size - len_fg]]#选出的后景(sbg_n,2).总计sfg+sbg=256
            
            # NOTE：selected_indices 是下面选择ao/at与gao/gat的关键
            # 类似于YOLOv3中通过build_target构造mask，但是明显YOLOv3那里更简明
            # 3.1 确定selected_indices
            selected_indices = torch.cat([fg_indices, bg_indices], dim=0)#合并随机前景与后景@(256,2) 第一列是批,第二列是ga_in内索引
            randperm_selected=torch.randperm(len(selected_indices))#再随机排列
            selected_indices = selected_indices[randperm_selected]
            selected_indices =selected_indices.unbind(dim=1)#分解为2个元素的tuple，每个元素是上面的一列：([批内索引]，[anchor内索引])

            #3.2 由selected_indices确定gao/gat(计算变换)
            selected_inside_anchor_bboxes = inside_anchor_bboxes[selected_indices]#(256,4)
            #selected_indices[0]表示选中的批, anchor_assignments[selected_indices]表示对应的gt_boxes
            selected_gt_bboxes = gt_bboxes_batch[selected_indices[0], anchor_assignments[selected_indices]]#(256,4)
            gt_anchor_objectnesses = labels[selected_indices]#(256,) 或为0或为1
            gt_anchor_transformers = BBox.calc_transformer(selected_inside_anchor_bboxes, selected_gt_bboxes)#(256，4). 逆变换：计算由gt到anchor的变换
            
            #3.3 由selected_indices确定ao/at
            inside_anchor_objectnesses=inside_anchor_objectnesses[selected_indices]#(256,2)
            inside_anchor_transformers=inside_anchor_transformers[selected_indices]#(256,4)

            batch_indices = selected_indices[0]#记录被选中gt的批索引. 由于通常batch_size为1，因此batch_indices全为0

            anchor_objectness_losses, anchor_transformer_losses = self.loss(inside_anchor_objectnesses,
                                                                            inside_anchor_transformers,
                                                                            gt_anchor_objectnesses,
                                                                            gt_anchor_transformers,
                                                                            batch_size, batch_indices)#计算 ao,at 与 gao，gat之间的损失
            #ao,at,aol,atl
            return anchor_objectnesses, anchor_transformers, anchor_objectness_losses, anchor_transformer_losses

    def loss(self, anchor_objectnesses: Tensor, anchor_transformers: Tensor,
             gt_anchor_objectnesses: Tensor, gt_anchor_transformers: Tensor,
             batch_size: int, batch_indices: Tensor) -> Tuple[Tensor, Tensor]:
       
        cross_entropies = torch.empty(batch_size, dtype=torch.float, device=anchor_objectnesses.device)
        smooth_l1_losses = torch.empty(batch_size, dtype=torch.float, device=anchor_transformers.device)

        for batch_index in range(batch_size):
            #一批一批计算误差.
            #这里因为通常batch_size==1，所以batch_indices总是等与batch_index.相当于对于这唯一的一批是个全选的效果.
            selected_indices = (batch_indices == batch_index).nonzero().view(-1)#(256,). 
            
            #1.使用交叉熵函数计算ao与gao之间损失
            #因为后景对应的label是0，前景对应的label是1，所以ao的第一列(除去batch内索引列)是后景，第二列是前景.
            cross_entropy = F.cross_entropy(input=anchor_objectnesses[selected_indices],
                                            target=gt_anchor_objectnesses[selected_indices])
            
            #对于transformer只计算前景:由nonzero计算出前景的索引
            #2.使用smooth_l1函数计算前景变换at与gat之间的损失
            fg_indices = gt_anchor_objectnesses[selected_indices].nonzero().view(-1)#(fg_n,)
            smooth_l1_loss = beta_smooth_l1_loss(input=anchor_transformers[selected_indices][fg_indices],
                                                 target=gt_anchor_transformers[selected_indices][fg_indices],
                                                 beta=self._anchor_smooth_l1_loss_beta)

            cross_entropies[batch_index] = cross_entropy
            smooth_l1_losses[batch_index] = smooth_l1_loss

        return cross_entropies, smooth_l1_losses


    #这是一个相当独立的函数，生成原图上的anchors：(anchors_x * anchors_y * 9,4)
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

        #ga_n=anchors_n=anchors_x * anchors_y * 9
        #ab@(bn,ga_n,4)
        #ao@(bn,ga_n,2)
        #at@(bn,ga_n,4)

        batch_size = anchor_bboxes.shape[0]

        #对所有anchors进行变换. 注意anchors是 左上右下 模式，进入函数后要变换为 中心宽高 模式. 返回前pb又变换成 左上右下 模式.
        proposal_bboxes = BBox.apply_transformer(anchor_bboxes, transformers)
        proposal_bboxes = BBox.clip(proposal_bboxes, left=0, top=0, right=image_width, bottom=image_height)#(bn,ga_n,4)
        #objectnesses[:, :, 1]表示前景列的置信，而 objectnesses[:, :, 0] 表示后景列置信
        proposal_probs = F.softmax(objectnesses[:, :, 1], dim=-1)#(bn,ga_n) softmax是增函数.对所有前景置信执行softmax.
        _, sorted_indices = torch.sort(proposal_probs, dim=-1, descending=True)#(bn,ga_n).对所有前景置信降序排列.
        
        nms_proposal_bboxes_batch = []
        for batch_index in range(batch_size):
            sorted_bboxes = proposal_bboxes[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]#(pre_nms_n,4)
            sorted_probs  =  proposal_probs[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]#(pre_nms_n,)
            threshold = 0.7
            kept_indices = nms(sorted_bboxes, sorted_probs, threshold)#(nms_n,) 在rpn中执行一次. 在detection.generate_detections中执行（推断时）
            nms_bboxes = sorted_bboxes[kept_indices][:self._post_nms_top_n]#(post_nms_n,4) , post_nms_n<=_post_nms_top_n
            nms_proposal_bboxes_batch.append(nms_bboxes)

        #从一批图像中，找到proposal_boxes最多的，记录数量为max_nms_n
        #其它图像的proposal_boxes要对齐max_nms_n(dim0方向，即增加行数)
        #就是说，一批内每幅图像的proposal_boxes数量都要相等
        max_nms_proposal_bboxes_length = max([len(it) for it in nms_proposal_bboxes_batch])
        padded_proposal_bboxes = []

        for nms_proposal_bboxes in nms_proposal_bboxes_batch:
            padded_proposal_bboxes.append(
                torch.cat([
                    nms_proposal_bboxes,
                    torch.zeros(max_nms_proposal_bboxes_length - len(nms_proposal_bboxes), 4).to(nms_proposal_bboxes)
                ])#为每幅图像的proposal_boxes增加 delta 行，每行4列 . 其中 delta = max_nms_proposal_bboxes_length - len(nms_proposal_bboxes) 
            )

        padded_proposal_bboxes = torch.stack(padded_proposal_bboxes, dim=0)#(bn,max_nms_n,4)->外部符号记为(bn,gp_n,4)，gp_n表示经过generate_proposal处理过的
        return padded_proposal_bboxes