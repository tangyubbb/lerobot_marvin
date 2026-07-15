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

"""数据集标注管理器，用于管理和查询 frame 级别的采样权重"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple


class AnnotationManager:
    """管理数据集标注文件，提供 frame 采样权重查询功能"""

    def __init__(self, annotation_path: str | Path):
        """
        初始化标注管理器

        Args:
            annotation_path: 标注文件路径
        """
        self.annotation_path = Path(annotation_path)
        self.annotations = self._load_annotations()
        self._validate_annotations()

    def _load_annotations(self) -> dict:
        """加载标注文件"""
        if not self.annotation_path.exists():
            logging.warning(f"标注文件不存在: {self.annotation_path}，使用默认标注")
            return self._create_default_annotations()

        try:
            with open(self.annotation_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            logging.info(f"成功加载标注文件: {self.annotation_path}")
            return annotations
        except json.JSONDecodeError as e:
            logging.error(f"标注文件格式错误: {e}")
            return self._create_default_annotations()

    def _create_default_annotations(self) -> dict:
        """创建默认标注结构"""
        return {
            "version": "1.0",
            "annotation_config": {
                "default_weight": 1.0,
                "weight_range": [0.1, 10.0]
            },
            "annotations": {}
        }

    def _validate_annotations(self):
        """验证标注文件格式"""
        if "annotation_config" not in self.annotations:
            logging.warning("标注文件缺少 annotation_config，使用默认配置")
            self.annotations["annotation_config"] = {
                "default_weight": 1.0,
                "weight_range": [0.1, 10.0]
            }

        config = self.annotations["annotation_config"]
        min_weight, max_weight = config.get("weight_range", [0.1, 10.0])

        # 验证所有权重在有效范围内
        for ep_key, ep_data in self.annotations.get("annotations", {}).items():
            for frame_key, frame_data in ep_data.get("frames", {}).items():
                weight = frame_data.get("weight", config.get("default_weight", 1.0))
                if weight < min_weight or weight > max_weight:
                    logging.warning(
                        f"{ep_key}/{frame_key}: 权重 {weight} 超出范围 "
                        f"[{min_weight}, {max_weight}]，将被裁剪"
                    )
                    frame_data["weight"] = max(min_weight, min(max_weight, weight))

    def get_frame_weight(self, episode_idx: int, frame_idx: int) -> float:
        """
        获取指定 frame 的采样权重

        Args:
            episode_idx: Episode 索引
            frame_idx: Frame 索引（相对于 episode 起始位置）

        Returns:
            该 frame 的采样权重
        """
        default_weight = self.annotations.get("annotation_config", {}).get("default_weight", 1.0)

        episode_key = f"episode_{episode_idx}"
        if episode_key not in self.annotations.get("annotations", {}):
            return default_weight

        frames = self.annotations["annotations"][episode_key].get("frames", {})

        # 查找匹配的 frame 范围或单帧
        for key, value in frames.items():
            if "-" in key:
                # 范围标注（如 "0-49"）
                start, end = map(int, key.split("-"))
                if start <= frame_idx <= end:
                    return value.get("weight", default_weight)
            elif key.isdigit() and int(key) == frame_idx:
                # 单帧标注
                return value.get("weight", default_weight)

        return default_weight

    def get_episode_weights(self, episode_idx: int, episode_length: int) -> list[float]:
        """
        获取整个 episode 的权重数组

        Args:
            episode_idx: Episode 索引
            episode_length: Episode 长度（帧数）

        Returns:
            长度为 episode_length 的权重列表
        """
        return [self.get_frame_weight(episode_idx, i) for i in range(episode_length)]

    def save_annotations(self, output_path: Optional[Path] = None):
        """
        保存标注文件

        Args:
            output_path: 输出路径，如果为 None 则覆盖原文件
        """
        save_path = output_path if output_path is not None else self.annotation_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)

        logging.info(f"标注文件已保存到: {save_path}")

    def get_statistics(self) -> dict:
        """
        获取标注统计信息

        Returns:
            包含统计信息的字典
        """
        total_episodes = len(self.annotations.get("annotations", {}))
        total_annotated_frames = 0
        weights = []

        for ep_data in self.annotations.get("annotations", {}).values():
            for frame_key, frame_data in ep_data.get("frames", {}).items():
                weight = frame_data.get("weight", 1.0)
                weights.append(weight)

                if "-" in frame_key:
                    start, end = map(int, frame_key.split("-"))
                    total_annotated_frames += (end - start + 1)
                else:
                    total_annotated_frames += 1

        stats = {
            "total_episodes": total_episodes,
            "total_annotated_frames": total_annotated_frames,
            "default_weight": self.annotations.get("annotation_config", {}).get("default_weight", 1.0),
        }

        if weights:
            import numpy as np
            stats.update({
                "weight_min": float(np.min(weights)),
                "weight_max": float(np.max(weights)),
                "weight_mean": float(np.mean(weights)),
                "weight_std": float(np.std(weights)),
            })

        return stats
