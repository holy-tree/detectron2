import numpy as np
import torch
import logging
from typing import Dict, Tuple, List, Optional
import pprint
import cv2

import torch.nn as nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads



############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


################ Gradient reverse function
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.neg()
    
def grad_reverse(x):
    return GradReverse.apply(x)


def calc_mean_std(features):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        roi_proposal: int = 64,
        # dis_loss_weight: float = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.input_format = input_format
        self.vis_period = vis_period
        self.roi_proposal = roi_proposal
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # @yujheli: you may need to build your discriminator here

        self.dis_type = dis_type
        self.D_img = None
        # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels['res4']) # Need to know the channel
        
        # self.D_img = None
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]) # Need to know the channel
        # self.bceLoss_func = nn.BCEWithLogitsLoss()

    def build_discriminator(self):
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            "roi_proposal": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            # "dis_loss_ratio": cfg.xxx,
        }

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        # !!! new add
        # 取消正则化
        images = [(x["image"]).float().to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images
    
    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.D_img == None:
            self.build_discriminator()
        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)
        
        source_label = 0
        target_label = 1

        # GRL + Discrimination
        if branch == "domain":
            # self.D_img.train()
            # source_label = 0
            # target_label = 1
            # images = self.preprocess_image(batched_inputs)
            images_s, images_t = self.preprocess_image_train(batched_inputs)

            features = self.backbone(images_s.tensor)

            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            features_t = self.backbone(images_t.tensor)
            
            features_t = grad_reverse(features_t[self.dis_type])
            # features_t = grad_reverse(features_t['p2'])
            D_img_out_t = self.D_img(features_t)
            loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            # import pdb
            # pdb.set_trace()

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s
            losses["loss_D_img_t"] = loss_D_img_t
            return losses, [], [], None
        

        # self.D_img.eval()
        images = self.preprocess_image(batched_inputs)

        # images将四张图片放入一个tensor中 images.tensor(N,3,H,W)
        # from  torchvision import utils as vutils
        # for i in range(images.tensor.shape[0]):
        #     image = images.tensor[i].clone().detach().double().to(torch.device('cpu'))
        #     vutils.save_image(image, f'./image{i}.jpg', normalize=True)
        # quit()



        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # if not adainNet == None:
        #     # print(images.tensor.shape)
        #     content = images.tensor[:2]
        #     style = images.tensor[4:]
        #     g_ts = adainNet(content, style, flag=2)
        #     images.tensor[:2] = g_ts
        #     images.tensor = images.tensor[:4]

            # print(g_ts.shape)

            # image1 = content[0].clone().detach().double().to(torch.device('cpu'))
            # image2 = content[1].clone().detach().double().to(torch.device('cpu'))
            # image3 = g_ts[0].clone().detach().double().to(torch.device('cpu'))
            # image4 = g_ts[1].clone().detach().double().to(torch.device('cpu'))

            # image1_np = np.ascontiguousarray(image1.numpy().transpose((1,2,0)))
            # image2_np = np.ascontiguousarray(image2.numpy().transpose((1,2,0)))
            # image3_np = np.ascontiguousarray(image3.numpy().transpose((1,2,0)))
            # image4_np = np.ascontiguousarray(image4.numpy().transpose((1,2,0)))
            # cv2.imwrite("./content1.jpg", image1_np)
            # cv2.imwrite("./content2.jpg", image2_np)
            # cv2.imwrite("./content3.jpg", image3_np)
            # cv2.imwrite("./content4.jpg", image4_np)
            # quit()
        

        features = self.backbone(images.tensor)

        # print(features)
        # print(features['res4'].shape)
        # quit()


        if branch == "content_alignment":
            # print(images.__len__())
            # from  torchvision import utils as vutils
            # for i in range(images.tensor.shape[0]):
            #     image = images.tensor[i].clone().detach().double().to(torch.device('cpu'))
            #     vutils.save_image(image, f'./image{i}.jpg', normalize=True)
            # quit()
           

            # gap = features[self.dis_type].shape[0] // 2

            # feature_q = features[self.dis_type][:gap]
            # feature_k = features[self.dis_type][gap:]
            # loss = F.mse_loss(feature_q, feature_k)
            # losses = {'loss_cont_cons': loss}
            # print(features)
            gap = features[self.dis_type].shape[0] // 2
            loss = 0
            for key in features:
                loss += F.mse_loss(features[key][:gap], features[key][gap:])
            losses = {'loss_cont_cons': loss}
            return losses


        # TODO: remove the usage of if else here. This needs to be re-organized
        # 引入 adain 模块
        if branch == "supervised":
            # 不太理解这里为什么重新计算loss_D_img_s
            # !!! 以下三行代码进行注释
            # features_s = grad_reverse(features[self.dis_type])
            # D_img_out_s = self.D_img(features_s)
            # loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)
            
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None
        
        elif branch == "adain_supervised":
            # 将 source feature 转化为 target 风格
            for key in features:
                source_feature1 = features[key][:2]
                source_feature2 = features[key][2:4]
                target_feature = features[key][4:]
                features1 = adain(source_feature1, target_feature)
                # features2 = adain(source_feature2, target_feature)
                features2 = source_feature2
                features[key] = torch.cat([features1, features2], dim=0)

            images = self.preprocess_image(batched_inputs[:4])
            gt_instances = gt_instances[:4]

            # 不太理解这里为什么重新计算loss_D_img_s
            # !!! 以下三行代码进行注释
            # features_s = grad_reverse(features[self.dis_type])
            # D_img_out_s = self.D_img(features_s)
            # loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, box_features, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )



            # new add!
            # 添加 loss_ins_cons 实例级别的一致性损失
            gap = self.roi_proposal            
            img_strong_proposals_1 = box_features[:gap]
            img_strong_proposals_2 = box_features[gap:gap*2]
            img_weak_proposals_1 = box_features[gap*2:gap*3]
            img_weak_proposals_2 = box_features[gap*3:]

            loss = 0
            compute_proposals = 10
            loss += F.mse_loss(img_strong_proposals_1[:compute_proposals], img_weak_proposals_1[:compute_proposals])
            loss += F.mse_loss(img_strong_proposals_2[:compute_proposals], img_weak_proposals_2[:compute_proposals])
            loss_ins_cons = {'loss_ins_cons': loss}

            # print(F.mse_loss(img_strong_proposals_1[:compute_proposals], img_weak_proposals_1[:compute_proposals]))
            # print(F.mse_loss(img_strong_proposals_2[:compute_proposals], img_weak_proposals_2[:compute_proposals]))
            # print(F.mse_loss(img_strong_proposals_1[:compute_proposals], img_strong_proposals_2[:compute_proposals]))
            # print(F.mse_loss(img_weak_proposals_1[:compute_proposals], img_weak_proposals_2[:compute_proposals]))
            
            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)
            
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(loss_ins_cons)
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None
        
        elif branch == "supervised_target":
            
            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )


            # from  torchvision import utils as vutils
            # for i in range(images.tensor.shape[0]):
            #     image = images.tensor[i].clone().detach().double().to(torch.device('cpu'))
            #     vutils.save_image(image, f'./image{i}.jpg', normalize=True)
            # for i in range(images.tensor.shape[0]):
            #     image = images.tensor[i].clone().detach().double().to(torch.device('cpu'))
            #     image = np.ascontiguousarray(image.numpy().transpose((1,2,0)))
            #     cv2.imwrite(f"./unlabel_strong_image{i}.jpg", image)
            # quit()
            

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            return {}, proposals_rpn, proposals_roih, ROI_predictions
        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None


