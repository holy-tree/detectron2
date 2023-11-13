import os
import io
import logging
import contextlib
from fvcore.common.timer import Timer
from iopath.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.pascal_voc import register_pascal_voc

logger = logging.getLogger(__name__)


def register_HS50():
    SPLITS = [
        ("HS50_train", "trainval"),
        ("HS50_test", "test"),
    ]
    for name, split in SPLITS:
        year = 2012
        register_pascal_voc(name, "/media/work/data1/zbt/dataset/two_domain/HS50_all_voc", split, year, class_names=["n"])
        # MetadataCatalog.get(name).evaluor_type = "pascal_voc_water"
        MetadataCatalog.get(name).thing_clatasses = ["n"]
        MetadataCatalog.get(name).evaluator_type = "coco"

def register_H60():
    SPLITS = [
        ("H60_train", "trainval"),
        ("H60_test", "test"),
    ]
    for name, split in SPLITS:
        year = 2012
        register_pascal_voc(name, "/media/work/data1/zbt/dataset/two_domain/H60_all_voc", split, year, class_names=["n"])
        # MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        MetadataCatalog.get(name).thing_classes = ["n"]
        MetadataCatalog.get(name).evaluator_type = "coco"

def register_TN():
    SPLITS = [
        ("TN_train", "trainval"),
        ("TN_test", "test"),
    ]
    for name, split in SPLITS:
        year = 2012
        register_pascal_voc(name, "/media/work/data1/zbt/dataset/TN-SCUI2020_data/VOC2007", split, year, class_names=["n"])
        # MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        MetadataCatalog.get(name).thing_classes = ["n"]
        MetadataCatalog.get(name).evaluator_type = "coco"

def register_XY():
    SPLITS = [
        ("XY_train", "trainval"),
        ("XY_test", "test"),
    ]
    for name, split in SPLITS:
        year = 2012
        register_pascal_voc(name, "/media/work/data1/zbt/dataset/xiangya/xyvoc", split, year, class_names=["n"])
        # MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        MetadataCatalog.get(name).thing_classes = ["n"]
        MetadataCatalog.get(name).evaluator_type = "coco"


def register_breast_nodule():
    SPLITS = [
        ("BN_train", "trainval"),
        ("BN_test", "test"),
    ]
    for name, split in SPLITS:
        year = 2012
        register_pascal_voc(name, "/media/work/data1/zbt/dataset/breast_nodule/breast_voc", split, year, class_names=["n"])
        # MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        MetadataCatalog.get(name).thing_classes = ["n"]
        MetadataCatalog.get(name).evaluator_type = "coco"

def register_breast_nodule_C_left():
    SPLITS = [
        ("BN_C_left_train", "trainval"),
        ("BN_C_left_test", "test"),
    ]
    for name, split in SPLITS:
        year = 2012
        register_pascal_voc(name, "/media/work/data1/zbt/dataset/breast_nodule/breast_voc_C_left", split, year, class_names=["n"])
        # MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        MetadataCatalog.get(name).thing_classes = ["n"]
        MetadataCatalog.get(name).evaluator_type = "coco"






def register_HS50_1500():
    SPLITS = [
        ("HS50_train_1500", "trainval"),
        ("HS50_test_1500", "test"),
    ]
    for name, split in SPLITS:
        year = 2012
        register_pascal_voc(name, "/media/work/data1/zbt/dataset/two_domain/HS50_all_voc_1500", split, year, class_names=["n"])
        # MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        MetadataCatalog.get(name).thing_classes = ["n"]
        MetadataCatalog.get(name).evaluator_type = "coco"

def register_H60_1500():
    SPLITS = [
        ("H60_train_1500", "trainval"),
        ("H60_test_1500", "test"),
    ]
    for name, split in SPLITS:
        year = 2012
        register_pascal_voc(name, "/media/work/data1/zbt/dataset/two_domain/H60_all_voc_1500", split, year, class_names=["n"])
        # MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        MetadataCatalog.get(name).thing_classes = ["n"]
        MetadataCatalog.get(name).evaluator_type = "coco"

def register_TN_1500():
    SPLITS = [
        ("TN_train_1500", "trainval"),
        ("TN_test_1500", "test"),
    ]
    for name, split in SPLITS:
        year = 2012
        register_pascal_voc(name, "/media/work/data1/zbt/dataset/TN-SCUI2020_data/VOC2007_1500", split, year, class_names=["n"])
        # MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        MetadataCatalog.get(name).thing_classes = ["n"]
        MetadataCatalog.get(name).evaluator_type = "coco"

register_HS50()
register_H60()
register_TN()
register_XY()
register_breast_nodule()
register_breast_nodule_C_left()

register_HS50_1500()
register_H60_1500()
register_TN_1500()