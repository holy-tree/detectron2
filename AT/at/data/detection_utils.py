import logging
import torchvision.transforms as transforms
from at.data.transforms.augementation_impl import (
    GaussianBlur,
)

def build_strong_augmentation(cfg, is_train, is_label):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        # 随机调整(概率为0.8)亮度、对比度、饱和度和色相
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        # # 将图像概率调整为灰度图
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        if is_label:
            randcrop_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomErasing(
                        p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                    ),
                    transforms.RandomErasing(
                        p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                    ),
                    transforms.RandomErasing(
                        p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                    ),
                    transforms.ToPILImage(),
                ]
            )
            augmentation.append(randcrop_transform)

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)