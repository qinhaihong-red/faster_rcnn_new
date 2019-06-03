import os
from typing import Union, Tuple, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from backbone.base import Base as BackboneBase
from bbox import BBox
from extention.functional import beta_smooth_l1_loss
from roi.pooler import Pooler
from rpn.region_proposal_network import RegionProposalNetwork
from support.layer.nms import nms


class Model(nn.Module):

    def __init__(self, backbone: BackboneBase, num_classes: int, pooler_mode: Pooler.Mode,
                 anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int],
                 rpn_pre_nms_top_n: int, rpn_post_nms_top_n: int,
                 anchor_smooth_l1_loss_beta: Optional[float] = None, proposal_smooth_l1_loss_beta: Optional[float] = None):
        super().__init__()
        
        #所用的backbone为resnet18/50/101
        #所有的resnet共10部分(0到9)
        #其中features是0到6（其0到4的参数冻结）部分，包括layer1到layer3的3个block. 
        #hidden是layer4.
        #以下使用resnet101为例说明:
        #num_features_out=1024
        #num_hidden_out=2048
        self.features, hidden, num_features_out, num_hidden_out = backbone.features()
        self._bn_modules = nn.ModuleList([it for it in self.features.modules() if isinstance(it, nn.BatchNorm2d)] +
                                         [it for it in hidden.modules() if isinstance(it, nn.BatchNorm2d)])

        # NOTE: It's crucial to freeze batch normalization modules for few batches training, which can be done by following processes
        #       (1) Change mode to `eval`
        #       (2) Disable gradient (we move this process into `forward`)
        for bn_module in self._bn_modules:
            for parameter in bn_module.parameters():
                parameter.requires_grad = False
        #以下参数用于infer:
        #num_features_out=1024
        #anchor_ratios=[(1, 2), (1, 1), (2, 1)]
        #anchor_sizes=[64, 128, 256, 512]
        #rpn_pre_nms_top_n=6000
        #rpn_post_nms_top_n=1000
        #anchor_smooth_l1_loss_beta=None
        self.rpn = RegionProposalNetwork(num_features_out, anchor_ratios, anchor_sizes, rpn_pre_nms_top_n, rpn_post_nms_top_n, anchor_smooth_l1_loss_beta)
        
        
        self.detection = Model.Detection(pooler_mode, hidden, num_hidden_out, num_classes, proposal_smooth_l1_loss_beta)
    #gt_bboxes_batch@(b,gt_n,4) 一批图像所包含的gt_boxes. 每幅图像的gt_n都相同，也就是经过0填充对齐过了的.
    #gt_classes_batch@(b,gt_n) gt_boxes所对应的类标签
    def forward(self, image_batch: Tensor,
                gt_bboxes_batch: Tensor = None, 
                gt_classes_batch: Tensor = None) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor],Tuple[Tensor, Tensor, Tensor, Tensor]]:
        # disable gradient for each forwarding process just in case model was switched to `train` mode at any time
        for bn_module in self._bn_modules:
            bn_module.eval()
        
        #以resnet101为例
        features = self.features(image_batch)#(batch_n,3,h,w)->(batch_n,1024,h/16,w/16)

        batch_size, _, image_height,    image_width  =  image_batch.shape
        _, _,       features_height, features_width  =  features.shape

        #在原图上生成所有anchors坐标@(ga_n,4)，以 左上右下 形式表达
        #这一步是最先执行的、相对独立的
        #符号体系如下：
        #ga_x=num_x_anchors=features_width, ga_y=num_y_anchors=features_height
        #ga_n=ga_x * ga_y * 9
        anchor_bboxes = self.rpn.generate_anchors(image_width, image_height, num_x_anchors=features_width, num_y_anchors=features_height)
        anchor_bboxes = anchor_bboxes.to(features)#tensor的.to方法，转dtype或者device,或者两个都转. 这里是把anchor转为与features相同的dtype和device.
        anchor_bboxes = anchor_bboxes.repeat(batch_size, 1, 1)#(bn,anchors_n,4)相当于增加一个批的维度

        #self.training是nn.Module的成员变量.
        #对于infer，在前面已经置model为eval,因此会改变training为False.
        if self.training:
            #训练使用voc2007数据集
            #gt_bboxes_batch@(1,gt_n,4),表示一批内每幅图像包含的gt_boxes数量
            #注意每幅图像的gt_boxes数量对齐本批内包含最多boxes的图像，这是在collate_fn中处理的.这里的批尺寸为1，相当于对齐自身.
            
            #1.rpn.forward@(f,ab,gb,w,h)->ao,at,aol,atl
            anchor_objectnesses, anchor_transformers, anchor_objectness_losses, anchor_transformer_losses = self.rpn.forward(features, anchor_bboxes, gt_bboxes_batch, image_width, image_height)
            
            #这里proposal_bboxes是经过detach过的（require_grad为False），再传入下面的detection.forward
            #2.rpn.generate_proposals@(ab,ao,at,w,h)->pb
            proposal_bboxes = self.rpn.generate_proposals(anchor_bboxes, anchor_objectnesses, anchor_transformers, image_width, image_height).detach()  # it's necessary to detach `proposal_bboxes` here
            
            #3.detection.forward@(f,pb,gc,gb)->pc,pt,pcl,ptl
            proposal_classes, proposal_transformers, proposal_class_losses, proposal_transformer_losses = self.detection.forward(features, proposal_bboxes, gt_classes_batch, gt_bboxes_batch)
            
            return anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses
        else:
            #ga_n=ga_x * ga_y * 9
            #返回 (bn, ga_n, 2) 与 (bn, ga_n, 4)
            anchor_objectnesses, anchor_transformers = self.rpn.forward(features)
            #返回经过rpn变换、排序、nms后的proposal_bboxes:(bn,gp_n,4),以 左上右下 模式表示
            proposal_bboxes = self.rpn.generate_proposals(anchor_bboxes, anchor_objectnesses, anchor_transformers, image_width, image_height)
            
            #(bn,gp_n,92) (bn,gp_n,92*4)
            proposal_classes, proposal_transformers = self.detection.forward(features, proposal_bboxes)
            
            #generate_detections只有在infer时用到
            #(gd_n,4) (gd_n,) (gd_n,) (gd_n,)
            detection_bboxes, detection_classes, detection_probs, detection_batch_indices = self.detection.generate_detections(proposal_bboxes, proposal_classes, proposal_transformers, image_width, image_height)
            return detection_bboxes, detection_classes, detection_probs, detection_batch_indices

    def save(self, path_to_checkpoints_dir: str, step: int, optimizer: Optimizer, scheduler: _LRScheduler) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir, f'model-{step}.pth')
        checkpoint = {
            'state_dict': self.state_dict(),
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str, optimizer: Optimizer = None, scheduler: _LRScheduler = None) -> 'Model':
        checkpoint = torch.load(path_to_checkpoint)
        self.load_state_dict(checkpoint['state_dict'])
        step = checkpoint['step']
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return step

    class Detection(nn.Module):

        def __init__(self, pooler_mode: Pooler.Mode, hidden: nn.Module, num_hidden_out: int, num_classes: int, proposal_smooth_l1_loss_beta: float):
            super().__init__()
            self._pooler_mode = pooler_mode
            self.hidden = hidden
            self.num_classes = num_classes
            self._proposal_class = nn.Linear(num_hidden_out, num_classes)
            self._proposal_transformer = nn.Linear(num_hidden_out, num_classes * 4)
            self._proposal_smooth_l1_loss_beta = proposal_smooth_l1_loss_beta
            self._transformer_normalize_mean = torch.tensor([0., 0., 0., 0.], dtype=torch.float)
            self._transformer_normalize_std = torch.tensor([.1, .1, .2, .2], dtype=torch.float)

        def forward(self, features: Tensor, proposal_bboxes: Tensor,
                    gt_classes_batch: Optional[Tensor] = None, gt_bboxes_batch: Optional[Tensor] = None) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
            
            #features@(bn,1024,h/16,w/16)
            #pb@(bn,gp_n,4)
            
            batch_size = features.shape[0]

            if not self.training:
                #(bn,gp_n)
                proposal_batch_indices = torch.arange(end=batch_size, dtype=torch.long, device=proposal_bboxes.device).view(-1, 1).repeat(1, proposal_bboxes.shape[1])
                #ROI pooling:(gp_n,1024,7,7)
                pool = Pooler.apply(features, proposal_bboxes.view(-1, 4), proposal_batch_indices.view(-1), mode=self._pooler_mode)
                
                hidden = self.hidden(pool)#(n4,2048,4,4)
                hidden = F.adaptive_max_pool2d(input=hidden, output_size=1)#(n4,2048,1,1)
                hidden = hidden.view(hidden.shape[0], -1)#(n4,2048)

                proposal_classes = self._proposal_class(hidden)#作分类的线性变换：(n4,num_cls)
                proposal_transformers = self._proposal_transformer(hidden)#框回归：(n4,num_cls*4)

                proposal_classes = proposal_classes.view(batch_size, -1, proposal_classes.shape[-1])#(batch_n,n4,num_cls)
                proposal_transformers = proposal_transformers.view(batch_size, -1, proposal_transformers.shape[-1])#(batch_n,n4,num_cls*4)
                return proposal_classes, proposal_transformers
            else:
                #NOTE 总的处理流程类似rpn的训练forward: ious->labels->fg/bg->selected_indices.
                
                # find labels for each `proposal_bboxes`
                # 1.为每个pb找到类label@(bn,pb_n)，默认值为-1
                labels = torch.full((batch_size, proposal_bboxes.shape[1]), -1, dtype=torch.long, device=proposal_bboxes.device)
                # ious@(bn,pb_n,gb_n) ious是确定labels的关键
                ious = BBox.iou(proposal_bboxes, gt_bboxes_batch)#NOTE iou在detection.forward中只用到一次，在rpn.forward中也只用到一次
                proposal_max_ious, proposal_assignments = ious.max(dim=2)#(bn,pb_n)
                labels[proposal_max_ious < 0.5] = 0#背景类
                fg_masks = proposal_max_ious >= 0.5#(bn,pb_n) 前景类的条件
                if len(fg_masks.nonzero()) > 0:
                    labels[fg_masks] = gt_classes_batch[fg_masks.nonzero()[:, 0], proposal_assignments[fg_masks]]#为前景类设置 类label

                #现在labels共分为3种：
                #a. 为0 ，对应背景proposal
                #b. 为cls_label(>0),对应前景proposal
                #c. 为-1的默认值，忽略

                # select 128 x `batch_size` samples
                #2.确定selected_indices, 选出 batch_size x 128 个训练样本
                fg_indices = (labels > 0).nonzero()#(fg_n,2)
                bg_indices = (labels == 0).nonzero()#(bg_n,2)
                fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 32 * batch_size)]]
                bg_indices = bg_indices[torch.randperm(len(bg_indices))[:128 * batch_size - len(fg_indices)]]
                selected_indices = torch.cat([fg_indices, bg_indices], dim=0)#(bn*128,2)
                #分裂为元素为2的tuple，每个元素是上面的一列：([批内索引]，[pb内索引])
                selected_indices = selected_indices[torch.randperm(len(selected_indices))].unbind(dim=1)

                proposal_bboxes = proposal_bboxes[selected_indices]#(bn*128,4)
                gt_bboxes = gt_bboxes_batch[selected_indices[0], proposal_assignments[selected_indices]]#(bn*128,4)
                gt_proposal_classes = labels[selected_indices]#(bn*128,)
                gt_proposal_transformers = BBox.calc_transformer(proposal_bboxes, gt_bboxes)#(bn*128,4)
                batch_indices = selected_indices[0]#(bn*128)

                #features@(bn,1024,w/16,h/16) proposal_bboxes@(bn*128,4)
                #pool@(bn*128,1024,7,7)
                pool = Pooler.apply(features, proposal_bboxes, proposal_batch_indices=batch_indices, mode=self._pooler_mode)
                hidden = self.hidden(pool)#(bn*128,2048,4,4)
                hidden = F.adaptive_max_pool2d(input=hidden, output_size=1)#(bn*128,2048,1,1)
                hidden = hidden.view(hidden.shape[0], -1)#(bn*128,2048)

                #self._proposal_class@Linear(2048,num_cls) self._proposal_transformer@Linear(2048,4*num_cls)
                proposal_classes = self._proposal_class(hidden)#(bn*128,num_cls)
                proposal_transformers = self._proposal_transformer(hidden)#(bn*128,num_cls)

                proposal_class_losses, proposal_transformer_losses = self.loss(proposal_classes, proposal_transformers,
                                                                               gt_proposal_classes, gt_proposal_transformers,
                                                                               batch_size, batch_indices)

                return proposal_classes, proposal_transformers, proposal_class_losses, proposal_transformer_losses

        def loss(self, proposal_classes: Tensor, proposal_transformers: Tensor,
                 gt_proposal_classes: Tensor, gt_proposal_transformers: Tensor,
                 batch_size, batch_indices) -> Tuple[Tensor, Tensor]:
            
            # proposal_classes@(bn*128,num_cls)
            # proposal_transformers@(bn*128,4*num_cls)
            # gt_proposal_classes@(128,)
            # gt_proposal_transformers@(bn*128,4*num_cls)
            # batch_size 通常为1，因此batch_indices就是1*128个0

            #由于proposal_transformers对于所有num_cls都生成了4个变换，总计有4*num_cls个预测变换，这里要根据 gt_proposal_classes 只选择有label的预测变换，因此：
            #proposal_transformers：(bn*128,84)-->(bn*128,4)
            proposal_transformers = proposal_transformers.view(-1, self.num_classes, 4)[torch.arange(end=len(proposal_transformers), dtype=torch.long), gt_proposal_classes]
            #对 gt_proposal_transformers 进行正则化处理
            transformer_normalize_mean = self._transformer_normalize_mean.to(device=gt_proposal_transformers.device)
            transformer_normalize_std = self._transformer_normalize_std.to(device=gt_proposal_transformers.device)
            gt_proposal_transformers = (gt_proposal_transformers - transformer_normalize_mean) / transformer_normalize_std  # scale up target to make regressor easier to learn

            cross_entropies = torch.empty(batch_size, dtype=torch.float, device=proposal_classes.device)
            smooth_l1_losses = torch.empty(batch_size, dtype=torch.float, device=proposal_transformers.device)

            for batch_index in range(batch_size):
                #batch_indices@(bn*128)
                #由于bn常取1,因此下面的selected_indices相当于选择整个批
                selected_indices = (batch_indices == batch_index).nonzero().view(-1)

                #1.使用交叉熵计算pc和gpc之间的损失
                cross_entropy = F.cross_entropy(input=proposal_classes[selected_indices],
                                                target=gt_proposal_classes[selected_indices])

                #2.使用smooth_l1计算前景pt和gpt之间的损失
                fg_indices = gt_proposal_classes[selected_indices].nonzero().view(-1)
                smooth_l1_loss = beta_smooth_l1_loss(input=proposal_transformers[selected_indices][fg_indices],
                                                     target=gt_proposal_transformers[selected_indices][fg_indices],
                                                     beta=self._proposal_smooth_l1_loss_beta)

                cross_entropies[batch_index] = cross_entropy
                smooth_l1_losses[batch_index] = smooth_l1_loss

            return cross_entropies, smooth_l1_losses
        
        #proposal_bboxes @(bn,gp_n,4) 
        #proposal_classes@(bn,gp_n,num_cls) 
        #proposal_transformers@(bn,gp_n,4*num_cls)  
        def generate_detections(self, proposal_bboxes: Tensor, proposal_classes: Tensor, proposal_transformers: Tensor, image_width: int, image_height: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            
            batch_size = proposal_bboxes.shape[0]

            proposal_transformers = proposal_transformers.view(batch_size, -1, self.num_classes, 4)#(bn,gp_n,num_cls,4)
            #因为在计算损失的时候，对变换进行了正则化处理，因此这里要逆正则化处理
            transformer_normalize_std = self._transformer_normalize_std.to(device=proposal_transformers.device)
            transformer_normalize_mean = self._transformer_normalize_mean.to(device=proposal_transformers.device)
            proposal_transformers = proposal_transformers * transformer_normalize_std + transformer_normalize_mean#(bn,gp_n,num_cls,4)

            proposal_bboxes = proposal_bboxes.unsqueeze(dim=2).repeat(1, 1, self.num_classes, 1)#(bn,gp_n,num_cls,4)
            detection_bboxes = BBox.apply_transformer(proposal_bboxes, proposal_transformers)#(bn,gp_n,num_cls,4)
            detection_bboxes = BBox.clip(detection_bboxes, left=0, top=0, right=image_width, bottom=image_height)
            detection_probs = F.softmax(proposal_classes, dim=-1)#(bn,gp_n,num_cls)

            all_detection_bboxes = []
            all_detection_classes = []
            all_detection_probs = []
            all_detection_batch_indices = []

            for batch_index in range(batch_size):
                for c in range(1, self.num_classes):
                    #1.先筛选出每个类
                    class_bboxes = detection_bboxes[batch_index, :, c, :]#(gp_n,4)
                    class_probs = detection_probs[batch_index, :, c]#(gp_n,)
                    #2.对同一类的boxes执行nms
                    threshold = 0.3
                    kept_indices = nms(class_bboxes, class_probs, threshold)#(nc,)
                    class_bboxes = class_bboxes[kept_indices]#(nc,4)
                    class_probs = class_probs[kept_indices]#(nc,)

                    #3.记录筛选后的同类boxes以及对应的类标签、置信、批内索引.
                    all_detection_bboxes.append(class_bboxes)
                    all_detection_classes.append(torch.full((len(kept_indices),), c, dtype=torch.int))
                    all_detection_probs.append(class_probs)
                    all_detection_batch_indices.append(torch.full((len(kept_indices),), batch_index, dtype=torch.long))

            #以这种形式返回的话，使用zip(boxes,classes,...)展开，通过循环就可以取得一个个的预测框：(box,cls,prob,bn).
            all_detection_bboxes = torch.cat(all_detection_bboxes, dim=0)#(gd_n,4)
            all_detection_classes = torch.cat(all_detection_classes, dim=0)#(gd_n,)
            all_detection_probs = torch.cat(all_detection_probs, dim=0)#(gd_n,)
            all_detection_batch_indices = torch.cat(all_detection_batch_indices, dim=0)#(gd_n,)
            return all_detection_bboxes, all_detection_classes, all_detection_probs, all_detection_batch_indices
