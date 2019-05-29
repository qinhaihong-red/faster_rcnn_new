import argparse
import os
import random
import torch

from PIL import ImageDraw
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from backbone.base import Base as BackboneBase
from bbox import BBox
from model import Model
from roi.pooler import Pooler
from config.eval_config import EvalConfig as Config


def _infer(path_to_input_image: str, path_to_output_image: str, path_to_checkpoint: str, dataset_name: str, backbone_name: str, prob_thresh: float):
    dataset_class = DatasetBase.from_name(dataset_name)
    backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
    model = Model(backbone, dataset_class.num_classes(), pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
    model.load(path_to_checkpoint)
    '''
    默认选项：
    pooler_mode=Config.POOLER_MODE= Pooler.Mode.ALIGN
    anchor_ratios=Config.ANCHOR_RATIOS= [(1, 2), (1, 1), (2, 1)]
    anchor_sizes=对于infer,这里默认增加了一个64，因此最后就是[64，128, 256, 512]

    用于Eval的RPN_NMS：
        RPN_PRE_NMS_TOP_N: int = 6000
        RPN_POST_NMS_TOP_N: int = 300

    '''

    with torch.no_grad():
        #预处理，使得输入图像至少一边满足min_side或max_side
        #yolo需要固定图像尺寸，这里并不需要.
        image = transforms.Image.open(path_to_input_image)
        image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

        #先增加一个批的维度，再以eval模式下执行forward.
        #(gd_n,4) (gd_n,) (gd_n,)
        detection_bboxes, detection_classes, detection_probs, _ = model.eval().forward(image_tensor.unsqueeze(dim=0).cuda())
        detection_bboxes /= scale #原图像是经过乘scale的，因此这里对于detection_box要除scale。

        kept_indices = detection_probs > prob_thresh #0.6
        detection_bboxes = detection_bboxes[kept_indices]#(gd_thresh_n,4)
        detection_classes = detection_classes[kept_indices]#(gd_thresh_n,)
        detection_probs = detection_probs[kept_indices]#(gd_thresh_n,)

        draw = ImageDraw.Draw(image)

        for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
            color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
            bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]

            draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
            draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

        image.save(path_to_output_image)
        print(f'Output image is saved to {path_to_output_image}')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()

        '''
        python infer.py -s=coco2017 -b=resnet101 -c=model-180000.pth --image_min_side=800 --image_max_side=1333 --anchor_sizes="[64, 128, 256, 512]" --rpn_post_nms_top_n=1000 field.jpg out_field.jpg
        
        这个model-180000.pth对应的训练数据是coco2017, 因此不能使用voc2007.
        '''
        #指定映射参数
        parser.add_argument('-s', '--dataset', type=str, default='coco2017',choices=DatasetBase.OPTIONS, help='name of dataset')
        parser.add_argument('-b', '--backbone', type=str, default='resnet101',choices=BackboneBase.OPTIONS,  help='name of backbone model')
        parser.add_argument('-c', '--checkpoint', type=str, default='model-180000.pth', help='path to checkpoint')
        parser.add_argument('-p', '--probability_threshold', type=float, default=0.6, help='threshold of detection probability')
        parser.add_argument('--image_min_side', default=800,type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side',default=1333, type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_sizes', type=str,default="[64, 128, 256, 512]",help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
        parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, default=1000,help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
        #最后两个是位置参数，没有指定映射参数
        #parser.add_argument('input', type=str, default='field.jpg',help='path to input image')
        #parser.add_argument('output', type=str, default='out_filed.jpg',help='path to output result image')
        args = parser.parse_args()

        path_to_input_image = 'field.jpg'
        path_to_output_image = 'out_field.jpg'
        dataset_name = args.dataset
        backbone_name = args.backbone
        path_to_checkpoint = args.checkpoint
        prob_thresh = args.probability_threshold

        os.makedirs(os.path.join(os.path.curdir, os.path.dirname(path_to_output_image)), exist_ok=True)

        #Config是由类变量和类方法构成的属性配置器
        Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
                     anchor_ratios=args.anchor_ratios, anchor_sizes=args.anchor_sizes, pooler_mode=args.pooler_mode,
                     rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n)

        print('Arguments:')
        for k, v in vars(args).items():
            print(f'\t{k} = {v}')
        print(Config.describe())

        _infer(path_to_input_image, path_to_output_image, path_to_checkpoint, dataset_name, backbone_name, prob_thresh)

    main()
