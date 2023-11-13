import copy
import logging
import numpy as np
from PIL import Image
import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.dataset_mapper import DatasetMapper

from at.data.detection_utils import build_strong_augmentation

class DatasetMapperTwoCropSeparate(DatasetMapper):
    """
    This customized mapper produces two augmented images from a single image
    instance. This mapper makes sure that the two augmented images have the same
    cropping and thus the same size.

    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True, is_label=True):
        # 首先进行图像比例缩放，将所有图像的最短边缩放成最小的图像  并进行旋转
        self.augmentation = utils.build_augmentation(cfg, is_train)
        # include crop into self.augmentation

        # 将图像进行裁剪 默认不进行裁剪
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            self.compute_tight_boxes = True
        else:
            self.compute_tight_boxes = False
        

        self.strong_augmentation = build_strong_augmentation(cfg, is_train, is_label)

        self.img_format = cfg.INPUT.FORMAT
        # ！！！一下内容与一般检测工作无关，故注释
        # fmt: off
        # self.mask_on = cfg.MODEL.MASK_ON
        # self.mask_format = cfg.INPUT.MASK_FORMAT
        # self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        # self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        # if self.keypoint_on and is_train:
        #     self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
        #         cfg.DATASETS.TRAIN
        #     )
        # else:
        #     self.keypoint_hflip_indices = None

        # if self.load_proposals:
        #     self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        #     self.proposal_topk = (
        #         cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
        #         if is_train
        #         else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
        #     )

        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        #！！！ 以下内容为分割任务    故注释
        # if "sem_seg_file_name" in dataset_dict:
        #     sem_seg_gt = utils.read_image(
        #         dataset_dict.pop("sem_seg_file_name"), "L"
        #     ).squeeze(2)
        # else:
        #     sem_seg_gt = None
        

        aug_input = T.StandardAugInput(image)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image_weak_aug = aug_input.image
        image_shape = image_weak_aug.shape[:2]  # h, w

        # if sem_seg_gt is not None:
        #     dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        
        # 因为load_proposals = false
        # if self.load_proposals:
        #     utils.transform_proposals(
        #         dataset_dict,
        #         image_shape,
        #         transforms,
        #         proposal_topk=self.proposal_topk,
        #         min_box_size=self.proposal_min_box_size,
        #     )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            # dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        
        # 将dataset_dict中的annotations变为instances
        if "annotations" in dataset_dict:
            # for anno in dataset_dict["annotations"]:
            #     if not self.mask_on:
            #         anno.pop("segmentation", None)
            #     if not self.keypoint_on:
            #         anno.pop("keypoints", None)
            # 由于对图像进行了缩放和旋转，标签也因此变化
            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            # 无用
            # if self.compute_tight_boxes and instances.has("gt_masks"):
            #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            bboxes_d2_format = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = bboxes_d2_format
        



        # apply strong augmentation
        # We use torchvision augmentation, which is not compatiable with
        # detectron2, which use numpy format for images. Thus, we need to
        # convert to PIL format first.
        image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
        image_strong_aug = np.array(self.strong_augmentation(image_pil))
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        )
        
        dataset_dict_key = copy.deepcopy(dataset_dict)
        dataset_dict_key["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )
        assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
        assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
        return (dataset_dict, dataset_dict_key)