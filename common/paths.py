
from os import path
from pathlib import Path


class PATHS:
    PROJECT_ROOT = path.join(*Path(path.dirname(path.abspath(__file__))).parts[: -1])
    OUT_ROOT = '/data/code/flowers-slassif/scan/results/flowers/'
    FLOWERS_DS_UNLABELED = '/data/datasets/DS_flowers_dataset/'
    FLOWERS_DS_KAGGLE = '/data/datasets/flowers-recognition-kaggle/flowers'
