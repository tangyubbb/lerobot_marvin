#!/usr/bin/env python3
"""
自定义标签加载器

用于在训练时加载和使用数据集的自定义标签字段（如关键帧标记）。
支持在数据采样时过滤、加权或标注数据。

用法：
    from lerobot.datasets.label_loader import LabelLoader

    # 初始化标签加载器
    label_loader = LabelLoader(dataset_root="/path/to/dataset")

    # 加载关键帧标签
    label_loader.load_label("is_keyframe")

    # 在数据加载时获取标签
    label_value = label_loader.get_label(index, "is_keyframe")
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class LabelLoader:
    """加载和管理数据集的自定义标签"""

    def __init__(self, dataset_root: str | Path):
        """
        初始化标签加载器

        Args:
            dataset_root: 数据集根目录路径
        """
        self.dataset_root = Path(dataset_root)
        self.labels_dir = self.dataset_root / "labels"

        # 存储已加载的标签数据 {label_name: DataFrame}
        self.loaded_labels: dict[str, pd.DataFrame] = {}

        # 存储标签元数据 {label_name: metadata_dict}
        self.label_metadata: dict[str, dict] = {}

        # 检查labels目录是否存在
        if not self.labels_dir.exists():
            logger.warning(f"标签目录不存在: {self.labels_dir}")
            logger.info("可以使用 keyframe_label/add_frame_labels.py 创建标签文件")

    def load_label(self, label_name: str, format: str = "parquet") -> bool:
        """
        加载指定的标签文件

        Args:
            label_name: 标签字段名称（如 "is_keyframe"）
            format: 文件格式 ('parquet', 'csv', 'json')

        Returns:
            是否成功加载
        """
        if not self.labels_dir.exists():
            logger.warning(f"标签目录不存在: {self.labels_dir}")
            return False

        # 构造标签文件路径
        if format == "parquet":
            label_file = self.labels_dir / f"{label_name}.parquet"
        elif format == "csv":
            label_file = self.labels_dir / f"{label_name}.csv"
        elif format == "json":
            label_file = self.labels_dir / f"{label_name}.json"
        else:
            raise ValueError(f"不支持的格式: {format}")

        if not label_file.exists():
            logger.warning(f"标签文件不存在: {label_file}")
            return False

        try:
            # 读取标签数据
            if format == "parquet":
                labels_df = pd.read_parquet(label_file)
            elif format == "csv":
                labels_df = pd.read_csv(label_file)
            elif format == "json":
                labels_df = pd.read_json(label_file, orient='records')

            # 验证必需的列
            required_cols = ['index', label_name]
            missing_cols = [col for col in required_cols if col not in labels_df.columns]
            if missing_cols:
                logger.error(f"标签文件缺少必需的列: {missing_cols}")
                return False

            # 设置index为索引以加速查找
            labels_df = labels_df.set_index('index')

            self.loaded_labels[label_name] = labels_df
            logger.info(f"成功加载标签: {label_name} ({len(labels_df)} 条记录)")

            # 加载元数据（如果存在）
            metadata_file = self.labels_dir / f"{label_name}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.label_metadata[label_name] = json.load(f)
                    logger.info(f"加载标签元数据: {label_name}")

            return True

        except Exception as e:
            logger.error(f"加载标签失败: {label_name}, 错误: {e}")
            return False

    def get_label(self, index: int, label_name: str, default: Any = None) -> Any:
        """
        获取指定帧的标签值

        Args:
            index: 全局帧索引
            label_name: 标签字段名称
            default: 如果标签不存在时的默认值

        Returns:
            标签值，如果未找到则返回default
        """
        if label_name not in self.loaded_labels:
            return default

        labels_df = self.loaded_labels[label_name]

        try:
            if index in labels_df.index:
                return labels_df.loc[index, label_name]
            else:
                return default
        except Exception as e:
            logger.warning(f"获取标签失败 (index={index}, label={label_name}): {e}")
            return default

    def get_labels_batch(self, indices: list[int], label_name: str, default: Any = None) -> list[Any]:
        """
        批量获取多个帧的标签值

        Args:
            indices: 全局帧索引列表
            label_name: 标签字段名称
            default: 如果标签不存在时的默认值

        Returns:
            标签值列表
        """
        if label_name not in self.loaded_labels:
            return [default] * len(indices)

        labels_df = self.loaded_labels[label_name]

        results = []
        for idx in indices:
            try:
                if idx in labels_df.index:
                    results.append(labels_df.loc[idx, label_name])
                else:
                    results.append(default)
            except Exception:
                results.append(default)

        return results

    def has_label(self, label_name: str) -> bool:
        """
        检查是否已加载指定标签

        Args:
            label_name: 标签字段名称

        Returns:
            是否已加载
        """
        return label_name in self.loaded_labels

    def get_metadata(self, label_name: str) -> dict | None:
        """
        获取标签的元数据

        Args:
            label_name: 标签字段名称

        Returns:
            元数据字典，如果不存在则返回None
        """
        return self.label_metadata.get(label_name)

    def list_available_labels(self) -> list[str]:
        """
        列出labels目录中所有可用的标签文件

        Returns:
            标签名称列表
        """
        if not self.labels_dir.exists():
            return []

        available_labels = []

        # 查找所有parquet文件
        for label_file in self.labels_dir.glob("*.parquet"):
            label_name = label_file.stem
            if not label_name.endswith("_metadata"):
                available_labels.append(label_name)

        return available_labels

    def get_keyframe_indices(self, label_name: str = "is_keyframe") -> list[int]:
        """
        获取所有标记为关键帧的索引

        Args:
            label_name: 关键帧标签字段名称

        Returns:
            关键帧索引列表
        """
        if label_name not in self.loaded_labels:
            logger.warning(f"标签 {label_name} 未加载")
            return []

        labels_df = self.loaded_labels[label_name]

        # 找出所有非零值的索引（假设0=非关键帧，非0=关键帧）
        keyframe_mask = labels_df[label_name] != 0
        keyframe_indices = labels_df[keyframe_mask].index.tolist()

        logger.info(f"找到 {len(keyframe_indices)} 个关键帧")
        return keyframe_indices

    def filter_keyframes(self, indices: list[int], label_name: str = "is_keyframe") -> list[int]:
        """
        从给定的索引列表中筛选出关键帧

        Args:
            indices: 待筛选的索引列表
            label_name: 关键帧标签字段名称

        Returns:
            关键帧索引列表
        """
        if label_name not in self.loaded_labels:
            logger.warning(f"标签 {label_name} 未加载，返回原始索引")
            return indices

        labels_df = self.loaded_labels[label_name]

        keyframe_indices = []
        for idx in indices:
            try:
                if idx in labels_df.index and labels_df.loc[idx, label_name] != 0:
                    keyframe_indices.append(idx)
            except Exception:
                continue

        return keyframe_indices

    def __repr__(self) -> str:
        return (
            f"LabelLoader(dataset_root={self.dataset_root}, "
            f"loaded_labels={list(self.loaded_labels.keys())})"
        )


def add_labels_to_batch(batch: dict, label_loader: LabelLoader, label_names: list[str]) -> dict:
    """
    将标签添加到训练batch中

    Args:
        batch: 训练数据batch（包含'index'字段）
        label_loader: 标签加载器实例
        label_names: 要添加的标签名称列表

    Returns:
        添加了标签字段的batch
    """
    if 'index' not in batch:
        logger.warning("Batch中没有'index'字段，无法添加标签")
        return batch

    # 获取batch中的索引
    indices = batch['index']

    # 如果是tensor，转换为列表
    if hasattr(indices, 'cpu'):
        indices = indices.cpu().numpy().tolist()
    elif hasattr(indices, 'tolist'):
        indices = indices.tolist()

    # 为每个标签添加对应的值
    for label_name in label_names:
        if label_loader.has_label(label_name):
            label_values = label_loader.get_labels_batch(indices, label_name, default=0)
            batch[label_name] = label_values

    return batch


# 示例用法
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试标签加载器")
    parser.add_argument("--dataset-root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--label-name", type=str, default="is_keyframe", help="标签名称")
    args = parser.parse_args()

    # 初始化标签加载器
    loader = LabelLoader(args.dataset_root)

    # 列出可用标签
    print(f"\n可用标签: {loader.list_available_labels()}")

    # 加载标签
    if loader.load_label(args.label_name):
        print(f"\n成功加载标签: {args.label_name}")

        # 获取元数据
        metadata = loader.get_metadata(args.label_name)
        if metadata:
            print(f"标签元数据: {metadata}")

        # 测试获取标签
        test_indices = [0, 10, 100, 1000]
        print(f"\n测试索引 {test_indices} 的标签值:")
        for idx in test_indices:
            value = loader.get_label(idx, args.label_name)
            print(f"  index {idx}: {value}")

        # 获取所有关键帧
        keyframes = loader.get_keyframe_indices(args.label_name)
        print(f"\n总关键帧数: {len(keyframes)}")
        if keyframes:
            print(f"前10个关键帧索引: {keyframes[:10]}")
