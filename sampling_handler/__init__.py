__version__ = "0.0.0"
__author__ = "FAO"

from .sampling.sample_size import SampleSize
from .sampling.sample_design import SampleDesign
from .time_series.ts_extract import TimeSeriesExtraction
from .dataset_augmentation.augment import DatasetAugmentation
from .dataset_augmentation.augment import DatasetAugmentation as DataAugmentation  # backwards compatibility
from .sampling.kmeans_subsampling import KMeansSubSampling
from .ensemble.classification import EnsembleClassification
