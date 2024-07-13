from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class CustomCocoDataset(BaseSegDataset):
    METAINFO = dict(
        CLASSES=(
            'Gelb-gruen-braunes Blatt',
            'Asymptomatische Trauben',
            'Symptomatische Trauben',
            'Gelb-gruenes Blatt',
            'Abgestorbenes Blatt',
            'Gelbes Blatt',
            'Gelber Rand'
        ),
        PALETTE=[
            [0, 0, 128],   # Navy
            [0, 0, 0],     # Black
            [0, 128, 0],   # Green
            [0, 128, 128], # Teal
            [128, 128, 0], # Olive
            [128, 0, 0],   # Maroon
            [128, 128, 128]# Grey
        ]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.CLASSES = self.METAINFO['CLASSES']
        self.PALETTE = self.METAINFO['PALETTE']
