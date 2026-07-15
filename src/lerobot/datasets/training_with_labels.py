#!/usr/bin/env python3
"""
训练时使用自定义标签的示例和工具

展示如何在训练过程中：
1. 加载自定义标签（如关键帧标记）
2. 在数据采样时使用标签进行过滤或加权
3. 在训练batch中包含标签信息

用法示例：

# 1. 在训练脚本中集成标签加载
from lerobot.datasets.label_loader import LabelLoader
from lerobot.datasets.training_with_labels import (
    wrap_dataset_with_labels,
    create_keyframe_weighted_sampler
)

# 初始化标签加载器
label_loader = LabelLoader(dataset_root="/path/to/dataset")
label_loader.load_label("is_keyframe")

# 包装数据集以支持标签
dataset = wrap_dataset_with_labels(dataset, label_loader, ["is_keyframe"])

# 创建加权采样器（可选）
sampler = create_keyframe_weighted_sampler(dataset, label_loader, keyframe_weight=2.0)

# 创建DataLoader
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

# 训练循环
for batch in dataloader:
    # batch 现在包含 'is_keyframe' 字段
    print(f"关键帧数量: {batch['is_keyframe'].sum()}")
    loss = train_step(batch)
"""

import logging
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from lerobot.datasets.label_loader import LabelLoader

logger = logging.getLogger(__name__)


class LabelAwareDataset(Dataset):
    """
    包装原始数据集，在__getitem__时自动添加标签字段
    """

    def __init__(
        self,
        base_dataset: Dataset,
        label_loader: LabelLoader,
        label_names: list[str] = None,
    ):
        """
        Args:
            base_dataset: 原始数据集
            label_loader: 标签加载器
            label_names: 要包含的标签名称列表（None表示包含所有已加载的标签）
        """
        self.base_dataset = base_dataset
        self.label_loader = label_loader

        # 确定要使用的标签
        if label_names is None:
            self.label_names = list(label_loader.loaded_labels.keys())
        else:
            # 验证标签是否已加载
            self.label_names = []
            for name in label_names:
                if label_loader.has_label(name):
                    self.label_names.append(name)
                else:
                    logger.warning(f"标签 '{name}' 未加载，将被忽略")

        logger.info(f"LabelAwareDataset 将包含标签: {self.label_names}")

    def __len__(self):
        return len(self.base_dataset)

    def __getattr__(self, name):
        # Keep the wrapper compatible with LeRobotDataset attributes such as
        # meta, episodes, num_frames, num_episodes, and camera_keys.
        return getattr(self.base_dataset, name)

    def __getitem__(self, idx):
        # 获取原始数据
        item = self.base_dataset[idx]

        # 添加标签
        # 假设item是字典，包含'index'字段
        if isinstance(item, dict) and 'index' in item:
            frame_index = item['index']

            # 如果index是tensor，转换为整数
            if isinstance(frame_index, torch.Tensor):
                frame_index = frame_index.item()

            # 为每个标签添加值
            for label_name in self.label_names:
                label_value = self.label_loader.get_label(frame_index, label_name, default=0)
                item[label_name] = label_value

        return item


class KeyframeWeightedSampler(Sampler):
    """
    加权采样器：对关键帧赋予更高的采样权重
    """

    def __init__(
        self,
        dataset: Dataset,
        label_loader: LabelLoader,
        label_name: str = "is_keyframe",
        keyframe_weight: float = 2.0,
        num_samples: int = None,
    ):
        """
        Args:
            dataset: 数据集
            label_loader: 标签加载器
            label_name: 关键帧标签名称
            keyframe_weight: 关键帧的权重倍数（>1 表示更频繁采样）
            num_samples: 每个epoch采样的样本数（None表示使用数据集长度）
        """
        self.dataset = dataset
        self.label_loader = label_loader
        self.label_name = label_name
        self.keyframe_weight = keyframe_weight
        self.num_samples = num_samples if num_samples is not None else len(dataset)

        # 检查标签是否已加载
        if not label_loader.has_label(label_name):
            raise ValueError(f"标签 '{label_name}' 未加载")

        # 计算每个样本的权重
        self.weights = self._compute_weights()

        logger.info(
            f"KeyframeWeightedSampler: {len(self.weights)} 个样本, "
            f"关键帧权重={keyframe_weight}"
        )

    def _compute_weights(self) -> torch.Tensor:
        """计算每个样本的采样权重"""
        weights = []

        for idx in range(len(self.dataset)):
            # 获取该样本的标签值
            label_value = self.label_loader.get_label(idx, self.label_name, default=0)

            # 如果是关键帧，赋予更高权重
            if label_value != 0:
                weight = self.keyframe_weight
            else:
                weight = 1.0

            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float)

    def __iter__(self):
        # 根据权重进行采样
        indices = torch.multinomial(
            self.weights,
            num_samples=self.num_samples,
            replacement=True
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


class KeyframeOnlySampler(Sampler):
    """
    只采样关键帧的采样器
    """

    def __init__(
        self,
        dataset: Dataset,
        label_loader: LabelLoader,
        label_name: str = "is_keyframe",
        shuffle: bool = True,
    ):
        """
        Args:
            dataset: 数据集
            label_loader: 标签加载器
            label_name: 关键帧标签名称
            shuffle: 是否随机打乱关键帧顺序
        """
        self.dataset = dataset
        self.label_loader = label_loader
        self.label_name = label_name
        self.shuffle = shuffle

        # 获取所有关键帧索引
        self.keyframe_indices = label_loader.get_keyframe_indices(label_name)

        if not self.keyframe_indices:
            logger.warning("没有找到任何关键帧！")

        logger.info(f"KeyframeOnlySampler: {len(self.keyframe_indices)} 个关键帧")

    def __iter__(self):
        indices = self.keyframe_indices.copy()

        if self.shuffle:
            import random
            random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return len(self.keyframe_indices)



class KeyframeRepeatSampler(Sampler):
    """Repeat keyframes and nearby frames according to fixed per-epoch totals.

    Total samples per epoch:
    - distance 0 from an ``is_keyframe=1`` frame: 20 samples
    - distance 1-15: 10 samples
    - distance 16-30: 5 samples
    - all other frames: 1 sample

    When windows overlap, the highest total sample count wins.
    """

    def __init__(
        self,
        dataset: Dataset,
        label_name: str = "is_keyframe",
        repeat_count: int | None = None,
        base_sampler: Sampler | None = None,
        shuffle: bool = True,
        seed: int = 0,
        keyframe_total: int = 20,
        near_total: int = 10,
        far_total: int = 5,
        near_radius: int = 15,
        far_radius: int = 30,
    ):
        self.dataset = dataset
        self.label_name = label_name
        self.base_sampler = base_sampler
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.keyframe_total = keyframe_total
        self.near_total = near_total
        self.far_total = far_total
        self.near_radius = near_radius
        self.far_radius = far_radius

        label_loader = getattr(dataset, "label_loader", None)
        if label_loader is None or not label_loader.has_label(label_name):
            self.keyframe_indices = []
            self.sample_totals = {}
        else:
            if base_sampler is not None and hasattr(base_sampler, "indices"):
                valid_indices = set(int(idx) for idx in base_sampler.indices)
            else:
                valid_indices = set(range(len(dataset)))

            self.keyframe_indices = [
                int(idx)
                for idx in label_loader.get_keyframe_indices(label_name)
                if int(idx) in valid_indices
            ]
            self.sample_totals = self._build_sample_totals(valid_indices)

        logger.info(
            "KeyframeRepeatSampler: %d keyframes, %d repeated-window frames, "
            "totals=(keyframe=%d, near=%d, far=%d)",
            len(self.keyframe_indices),
            len(self.sample_totals),
            self.keyframe_total,
            self.near_total,
            self.far_total,
        )

    def _target_total_for_distance(self, distance: int) -> int:
        if distance == 0:
            return self.keyframe_total
        if distance <= self.near_radius:
            return self.near_total
        if distance <= self.far_radius:
            return self.far_total
        return 1

    def _build_sample_totals(self, valid_indices: set[int]) -> dict[int, int]:
        sample_totals: dict[int, int] = {}
        dataset_last_index = len(self.dataset) - 1
        for keyframe_idx in self.keyframe_indices:
            start = max(0, keyframe_idx - self.far_radius)
            end = min(dataset_last_index, keyframe_idx + self.far_radius)
            for idx in range(start, end + 1):
                if idx not in valid_indices:
                    continue
                target_total = self._target_total_for_distance(abs(idx - keyframe_idx))
                if target_total > sample_totals.get(idx, 1):
                    sample_totals[idx] = target_total
        return sample_totals

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1

        if self.base_sampler is None:
            base_indices = list(range(len(self.dataset)))
            if self.shuffle:
                order = torch.randperm(len(base_indices), generator=generator).tolist()
                base_indices = [base_indices[i] for i in order]
        else:
            base_indices = list(iter(self.base_sampler))

        extra_indices = []
        for idx, total_count in self.sample_totals.items():
            extra_indices.extend([idx] * max(0, total_count - 1))

        if extra_indices:
            order = torch.randperm(len(extra_indices), generator=generator).tolist()
            extra_indices = [extra_indices[i] for i in order]

        indices = base_indices + extra_indices
        if self.shuffle and extra_indices:
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in order]

        return iter(indices)

    def __len__(self):
        base_len = len(self.base_sampler) if self.base_sampler is not None else len(self.dataset)
        return base_len + sum(max(0, total_count - 1) for total_count in self.sample_totals.values())

    def state_dict(self) -> dict:
        state = {"epoch": self.epoch}
        if hasattr(self.base_sampler, "state_dict"):
            state["base_sampler"] = self.base_sampler.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.epoch = state.get("epoch", 0)
        if "base_sampler" in state and hasattr(self.base_sampler, "load_state_dict"):
            self.base_sampler.load_state_dict(state["base_sampler"])


def create_keyframe_repeat_sampler(
    dataset: Dataset,
    label_name: str = "is_keyframe",
    repeat_count: int | None = None,
    base_sampler: Sampler | None = None,
    shuffle: bool = True,
) -> KeyframeRepeatSampler | None:
    label_loader = getattr(dataset, "label_loader", None)
    if label_loader is None or not label_loader.has_label(label_name):
        return None

    sampler = KeyframeRepeatSampler(
        dataset=dataset,
        label_name=label_name,
        repeat_count=repeat_count,
        base_sampler=base_sampler,
        shuffle=shuffle,
    )
    if not sampler.sample_totals:
        return None
    return sampler


def wrap_dataset_with_labels(
    dataset: Dataset,
    label_loader: LabelLoader,
    label_names: list[str] = None,
) -> LabelAwareDataset:
    """
    包装数据集以支持标签

    Args:
        dataset: 原始数据集
        label_loader: 标签加载器
        label_names: 要包含的标签名称列表

    Returns:
        包装后的数据集
    """
    return LabelAwareDataset(dataset, label_loader, label_names)


def create_keyframe_weighted_sampler(
    dataset: Dataset,
    label_loader: LabelLoader,
    label_name: str = "is_keyframe",
    keyframe_weight: float = 2.0,
) -> KeyframeWeightedSampler:
    """
    创建关键帧加权采样器

    Args:
        dataset: 数据集
        label_loader: 标签加载器
        label_name: 关键帧标签名称
        keyframe_weight: 关键帧权重倍数

    Returns:
        加权采样器
    """
    return KeyframeWeightedSampler(
        dataset=dataset,
        label_loader=label_loader,
        label_name=label_name,
        keyframe_weight=keyframe_weight,
    )


def create_keyframe_only_sampler(
    dataset: Dataset,
    label_loader: LabelLoader,
    label_name: str = "is_keyframe",
    shuffle: bool = True,
) -> KeyframeOnlySampler:
    """
    创建只采样关键帧的采样器

    Args:
        dataset: 数据集
        label_loader: 标签加载器
        label_name: 关键帧标签名称
        shuffle: 是否随机打乱

    Returns:
        关键帧采样器
    """
    return KeyframeOnlySampler(
        dataset=dataset,
        label_loader=label_loader,
        label_name=label_name,
        shuffle=shuffle,
    )


# ==================== 使用示例 ====================

def example_training_with_labels():
    """
    示例：如何在训练中使用标签
    """
    from torch.utils.data import DataLoader

    # 假设已经有一个数据集
    # dataset = make_dataset(cfg)

    # 1. 初始化标签加载器
    dataset_root = "/path/to/dataset"
    label_loader = LabelLoader(dataset_root)

    # 2. 加载关键帧标签
    label_loader.load_label("is_keyframe")

    # 3. 选择采样策略

    # 策略A: 普通采样，但batch中包含标签信息
    # labeled_dataset = wrap_dataset_with_labels(dataset, label_loader, ["is_keyframe"])
    # dataloader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)

    # 策略B: 加权采样，关键帧被更频繁采样
    # labeled_dataset = wrap_dataset_with_labels(dataset, label_loader, ["is_keyframe"])
    # sampler = create_keyframe_weighted_sampler(labeled_dataset, label_loader, keyframe_weight=2.0)
    # dataloader = DataLoader(labeled_dataset, batch_size=32, sampler=sampler)

    # 策略C: 只训练关键帧
    # labeled_dataset = wrap_dataset_with_labels(dataset, label_loader, ["is_keyframe"])
    # sampler = create_keyframe_only_sampler(labeled_dataset, label_loader)
    # dataloader = DataLoader(labeled_dataset, batch_size=32, sampler=sampler)

    # 4. 训练循环
    # for epoch in range(num_epochs):
    #     for batch in dataloader:
    #         # batch 现在包含 'is_keyframe' 字段
    #         images = batch['observation.images.camera0']
    #         actions = batch['action']
    #         is_keyframe = batch['is_keyframe']
    #
    #         # 可以根据is_keyframe调整损失权重
    #         loss = compute_loss(images, actions)
    #         if is_keyframe.sum() > 0:
    #             # 对关键帧给予更高的损失权重
    #             keyframe_mask = is_keyframe.bool()
    #             loss[keyframe_mask] = loss[keyframe_mask] * 2.0
    #
    #         loss.mean().backward()
    #         optimizer.step()

    print("示例代码已准备好，请参考注释中的用法")


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("训练中使用标签的三种策略:")
    print("=" * 60)
    print("\n策略A: 普通采样 + 标签信息")
    print("  - 正常随机采样所有数据")
    print("  - batch中包含标签字段供训练使用")
    print("  - 适用于：在损失函数中使用标签调整权重")

    print("\n策略B: 加权采样")
    print("  - 关键帧被更频繁地采样")
    print("  - 通过keyframe_weight参数控制权重倍数")
    print("  - 适用于：关键帧数量较少但很重要的场景")

    print("\n策略C: 只训练关键帧")
    print("  - 只使用标记为关键帧的数据")
    print("  - 大幅减少训练数据量")
    print("  - 适用于：数据量大且关键帧已充分标注")

    print("\n" + "=" * 60)
    print("集成到训练脚本的步骤:")
    print("=" * 60)
    print("""
1. 在训练脚本开头导入:
   from lerobot.datasets.label_loader import LabelLoader
   from lerobot.datasets.training_with_labels import wrap_dataset_with_labels

2. 在创建数据集后添加:
   label_loader = LabelLoader(cfg.dataset.root)
   label_loader.load_label("is_keyframe")
   dataset = wrap_dataset_with_labels(dataset, label_loader, ["is_keyframe"])

3. 在训练循环中使用:
   for batch in dataloader:
       is_keyframe = batch['is_keyframe']
       # 根据需要使用标签信息
    """)
