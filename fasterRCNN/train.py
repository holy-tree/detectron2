#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os, cv2, torch
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import pycocotools


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


# 注册数据集
#声明类别，尽量保持
CLASS_NAMES =["N"]
DATASET_CATEGORIES = [
    # {"name": "background", "id": 0, "isthing": 1, "color": [220, 20, 60]},
    {"name": "N", "id": 0, "isthing": 1, "color": [219, 142, 185]},
]
# 数据集路径
DATASET_ROOT = '/media/work/data/zbt/detectron2/datasets/COCO'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'JPEGImages', 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'JPEGImages', 'val')
TEST_PATH = os.path.join(DATASET_ROOT, 'JPEGImages', 'test')

TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
TEST_JSON = os.path.join(ANN_ROOT, 'test.json')


# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "coco_my_train": (TRAIN_PATH, TRAIN_JSON),
    "coco_my_val": (VAL_PATH, VAL_JSON),
    "coco_my_test": (TEST_PATH, TEST_JSON)
}



#===========以下有两种注册数据集的方法，本人直接用的第二个plain_register_dataset的方式 也可以用register_dataset的形式==================
#注册数据集（这一步就是将自定义数据集注册进Detectron2）
def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key, 
                                   metadate=get_dataset_instances_meta(), 
                                   json_file=json_file, 
                                   image_root=image_root)


def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file, 
                                  image_root=image_root, 
                                  evaluator_type="coco", 
                                  **metadate)


#=============================
# 注册数据集和元数据
# def plain_register_dataset():
#     #训练集
#     DatasetCatalog.register("coco_my_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
#     MetadataCatalog.get("coco_my_train").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
#                                                     evaluator_type='coco', # 指定评估方式
#                                                     json_file=TRAIN_JSON,
#                                                     image_root=TRAIN_PATH)

#     #DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
#     #验证/测试集
#     DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
#     MetadataCatalog.get("coco_my_val").set(thing_classes=CLASS_NAMES, # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
#                                                 evaluator_type='coco', # 指定评估方式
#                                                 json_file=VAL_JSON,
#                                                 image_root=VAL_PATH)
# 查看数据集标注，可视化检查数据集标注是否正确，
#这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
#可选择使用此方法
def checkout_dataset_annotation(name="coco_my_train"):
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('show', vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    args.config_file = "../configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"
    cfg.merge_from_file(args.config_file)   
    cfg.merge_from_list(args.opts)       

    # 更改配置参数
    cfg.DATASETS.TRAIN = ("coco_my_train",) # 训练数据集名称
    cfg.DATASETS.TEST = ("coco_my_test",)
    cfg.DATALOADER.NUM_WORKERS = 4  # 单线程

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 640 # 训练图片输入的最大尺寸
    cfg.INPUT.MAX_SIZE_TEST = 640 # 测试数据输入的最大尺寸
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 768) # 训练图片输入的最小尺寸，可以设定为多尺度训练
    cfg.INPUT.MIN_SIZE_TEST = 640
    #cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING，其存在两种配置，分别为 choice 与 range ：
    # range 让图像的短边从 512-768随机选择
    #choice ： 把输入图像转化为指定的，有限的几种图片大小进行训练，即短边只能为 512或者768
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'
    # 本句一定要看下注释！！！！！！！！
    # cfg.MODEL.RETINANET.NUM_CLASSES = 1  # 类别数+1（因为有background，也就是你的 cate id 从 1 开始，如果您的数据集Json下标从 0 开始，这个改为您对应的类别就行，不用再加背景类！！！！！）
    # cfg.MODEL.WEIGHTS="./output/model_final.pth"
    # cfg.MODEL.WEIGHTS = "/home/yourstorePath/model_final_5bd44e.pkl"    # 预训练模型权重
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.SOLVER.IMS_PER_BATCH = 1  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size

    # 根据训练数据总数目以及batch_size，计算出每个epoch需要的迭代次数
    #9000为你的训练数据的总数目，可自定义
    ITERS_IN_ONE_EPOCH = int(5000 / cfg.SOLVER.IMS_PER_BATCH)

    # 指定最大迭代次数
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 6) - 1 # 12 epochs，
    # cfg.SOLVER.MAX_ITER = 100
    # 初始学习率
    cfg.SOLVER.BASE_LR = 0.002
    # 优化器动能
    cfg.SOLVER.MOMENTUM = 0.9
    #权重衰减
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # 学习率衰减倍数
    cfg.SOLVER.GAMMA = 0.1
    # 迭代到指定次数，学习率进行衰减
    cfg.SOLVER.STEPS = (7000,)
    # 在训练之前，会做一个热身运动，学习率慢慢增加初始学习率
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # 热身迭代次数
    cfg.SOLVER.WARMUP_ITERS = 1000

    cfg.SOLVER.WARMUP_METHOD = "linear"
    # 保存模型文件的命名数据减1
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    # 迭代到指定次数，进行一次评估
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
    #cfg.TEST.EVAL_PERIOD = 100

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    
    cfg = setup(args)

    torch.cuda.set_device(1)

    register_dataset()
    # plain_register_dataset()
    # checkout_dataset_annotation()
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
