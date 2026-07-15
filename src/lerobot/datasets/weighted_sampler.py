#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""支持采样权重的 Episode 感知采样器"""

import logging
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler


class WeightedEpisodeAwareSampler(Sampler):
    """
    支持采样权重的 Episode 感知采样器

    该采样器能够：
    1. 识别 episode 边界，避免跨 episode 采样
    2. 丢弃 episode 末尾的帧（避免无效的 action chunk）
    3. 根据 frame 级别的权重进行加权采样
    """

    def __init__(
        self,
        dataset_from_indices: List[int],
        dataset_to_indices: List[int],
        episode_indices_to_use: Optional[List[int]] = None,
        frame_weights: Optional[np.ndarray] = None,
        drop_n_last_frames: int = 0,
        shuffle: bool = True,
    ):
        """
        Args:
            dataset_from_indices: 每个 episode 的起始索引列表
            dataset_to_indices: 每个 episode 的结束索引列表
            episode_indices_to_use: 要使用的 episode 索引列表（None 表示使用全部）
            frame_weights: 每个 frame 的采样权重数组 (shape: [total_frames])
            drop_n_last_frames: 每个 episode 末尾丢弃的帧数
            shuffle: 是否随机打乱
        """
        self.dataset_from_indices = dataset_from_indices
        self.dataset_to_indices = dataset_to_indices
        self.drop_n_last_frames = drop_n_last_frames
        self.shuffle = shuffle

        if episode_indices_to_use is None:
            episode_indices_to_use = list(range(len(dataset_from_indices)))
        self.episode_indices_to_use = episode_indices_to_use

        # 构建有效的 frame 索引和对应的权重
        self.valid_indices = []
        self.weights = []

        for ep_idx in self.episode_indices_to_use:
            start = dataset_from_indices[ep_idx]
            end = dataset_to_indices[ep_idx] - drop_n_last_frames

            for frame_idx in range(start, end):
                self.valid_indices.append(frame_idx)
                # 如果提供了权重，使用权重；否则使用均匀权重
                if frame_weights is not None and frame_idx < len(frame_weights):
                    self.weights.append(frame_weights[frame_idx])
                else:
                    self.weights.append(1.0)

        self.weights = np.array(self.weights, dtype=np.float32)

        # 检查权重有效性
        if np.any(self.weights < 0):
            logging.warning("检测到负权重，将被设置为 0")
            self.weights = np.maximum(self.weights, 0)

        if np.sum(self.weights) == 0:
            logging.warning("所有权重为 0，使用均匀权重")
            self.weights = np.ones_like(self.weights)

        # 归一化权重为概率分布
        self.weights = self.weights / self.weights.sum()

        logging.info(
            f"WeightedEpisodeAwareSampler 初始化完成: "
            f"{len(self.valid_indices)} 个有效帧, "
            f"权重范围 [{self.weights.min():.6f}, {self.weights.max():.6f}]"
        )

    def __iter__(self) -> Iterator[int]:
        """返回采样索引的迭代器"""
        if self.shuffle:
            # 根据权重进行加权随机采样（无放回）
            indices = np.random.choice(
                len(self.valid_indices),
                size=len(self.valid_indices),
                replace=False,
                p=self.weights
            )
            sampled_indices = [self.valid_indices[i] for i in indices]
        else:
            # 不打乱，按顺序返回
            sampled_indices = self.valid_indices

        return iter(sampled_indices)

    def __len__(self) -> int:
        """返回采样器中有效帧的数量"""
        return len(self.valid_indices)

    def get_weight_statistics(self) -> dict:
        """
        获取权重统计信息

        Returns:
            包含权重统计的字典
        """
        # 还原为原始权重（未归一化）
        original_weights = self.weights * len(self.weights)

        return {
            "num_frames": len(self.valid_indices),
            "weight_min": float(original_weights.min()),
            "weight_max": float(original_weights.max()),
            "weight_mean": float(original_weights.mean()),
            "weight_std": float(original_weights.std()),
            "weight_median": float(np.median(original_weights)),
            "num_episodes": len(self.episode_indices_to_use),
        }
