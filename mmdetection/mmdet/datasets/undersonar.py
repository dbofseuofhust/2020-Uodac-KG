from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class UnderSonarDataset(CocoDataset):
    CLASSES = ('target')