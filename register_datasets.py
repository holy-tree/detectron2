from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.data import MetadataCatalog


def register_HS50():
    SPLITS = [
        ("HS50_train", "trainval"),
        ("HS50_test", "test"),
    ]
    for name, split in SPLITS:
        year = 2012
        register_pascal_voc(name, "/media/work/data/zbt/dataset/two_domain/HS50_all_voc", split, year, class_names=["n"])
        # MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        MetadataCatalog.get(name).thing_classes = ["n"]
        MetadataCatalog.get(name).evaluator_type = "coco"

def register_H60():
    SPLITS = [
        ("H60_train", "trainval"),
        ("H60_test", "test"),
    ]
    for name, split in SPLITS:
        year = 2012
        register_pascal_voc(name, "/media/work/data/zbt/dataset/two_domain/H60_all_voc", split, year, class_names=["n"])
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
        register_pascal_voc(name, "/media/work/data/zbt/dataset/TN-SCUI2020_data/VOC2007", split, year, class_names=["n"])
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
        register_pascal_voc(name, "/media/work/data/zbt/dataset/breast_nodule/breast_voc", split, year, class_names=["n"])
        # MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        MetadataCatalog.get(name).thing_classes = ["n"]
        MetadataCatalog.get(name).evaluator_type = "coco"

register_HS50()
register_H60()
register_TN()
register_breast_nodule()