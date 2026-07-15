#!/usr/bin/env python

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .factory import make_dataset, resolve_delta_timestamps
from .label_loader import LabelLoader
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .sampler import EpisodeAwareSampler
from .streaming_dataset import StreamingLeRobotDataset
from .training_with_labels import LabelAwareDataset, create_keyframe_repeat_sampler, wrap_dataset_with_labels

try:
    from .annotation_manager import AnnotationManager
except Exception:
    AnnotationManager = None

try:
    from .weighted_sampler import WeightedEpisodeAwareSampler
except Exception:
    WeightedEpisodeAwareSampler = None

__all__ = [
    "CODEBASE_VERSION",
    "EpisodeAwareSampler",
    "LabelAwareDataset",
    "LabelLoader",
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "MultiLeRobotDataset",
    "StreamingLeRobotDataset",
    "make_dataset",
    "resolve_delta_timestamps",
    "create_keyframe_repeat_sampler",
    "wrap_dataset_with_labels",
]

if AnnotationManager is not None:
    __all__.append("AnnotationManager")
if WeightedEpisodeAwareSampler is not None:
    __all__.append("WeightedEpisodeAwareSampler")
