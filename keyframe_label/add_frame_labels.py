#!/usr/bin/env python3
"""
为 LeRobot 数据集的每一帧添加自定义字段

用途：
- 为每一帧添加新字段（如关键帧标记、重要性分数等）
- 保存为数据集的附属文件，不修改原始数据
- 支持后续读取和修改标注

用法：
    python keyframe_label/add_frame_labels.py --repo-id /home/marvin/hhw/peg_optical_module_0707   
    python add_frame_labels.py --root /path/to/dataset --field-name importance --default-value 0.0

示例：
    # 添加关键帧标记字段（默认值为0）
    python add_frame_labels.py --repo-id hukewei/eval_pro_act5 --field-name is_keyframe

    # 添加重要性分数字段（默认值为0.0）
    python add_frame_labels.py --root /path/to/dataset --field-name importance --default-value 0.0
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


class FrameLabelManager:
    """管理数据集帧标签的工具类"""

    def __init__(self, dataset_root: Path):
        """
        初始化标签管理器

        Args:
            dataset_root: 数据集根目录
        """
        self.dataset_root = Path(dataset_root)
        self.info_path = self.dataset_root / "meta" / "info.json"
        self.labels_dir = self.dataset_root / "labels"

        # 加载数据集信息
        with open(self.info_path, 'r') as f:
            self.info = json.load(f)

        self.total_episodes = self.info["total_episodes"]
        self.total_frames = self.info["total_frames"]
        self.fps = self.info["fps"]

        print(f"数据集信息:")
        print(f"  总 episodes: {self.total_episodes}")
        print(f"  总帧数: {self.total_frames}")
        print(f"  FPS: {self.fps}")

    def create_labels_dir(self):
        """创建标签目录"""
        self.labels_dir.mkdir(exist_ok=True)
        print(f"\n标签目录: {self.labels_dir}")

    def load_data_files(self) -> pd.DataFrame:
        """
        加载所有数据文件，获取完整的帧索引

        Returns:
            包含所有帧信息的 DataFrame
        """
        print("\n加载数据文件...")

        data_dir = self.dataset_root / "data"
        all_data = []

        # 读取所有 chunk 中的数据文件
        for chunk_dir in sorted(data_dir.glob("chunk-*")):
            for data_file in sorted(chunk_dir.glob("file-*.parquet")):
                df = pd.read_parquet(data_file)
                all_data.append(df)
                print(f"  读取: {data_file.name} ({len(df)} 帧)")

        if not all_data:
            raise ValueError("未找到数据文件")

        # 合并所有数据
        full_data = pd.concat(all_data, ignore_index=True)
        print(f"\n总共加载: {len(full_data)} 帧")

        return full_data

    def create_frame_labels(
        self,
        field_name: str,
        default_value: Any = 0,
        data_type: str = "int64"
    ) -> pd.DataFrame:
        """
        创建帧标签数据

        Args:
            field_name: 字段名称
            default_value: 默认值
            data_type: 数据类型 ('int64', 'float64', 'bool', 'str')

        Returns:
            包含标签的 DataFrame
        """
        print(f"\n创建帧标签:")
        print(f"  字段名: {field_name}")
        print(f"  默认值: {default_value}")
        print(f"  数据类型: {data_type}")

        # 加载数据获取帧信息
        full_data = self.load_data_files()

        # 创建标签 DataFrame
        labels_df = pd.DataFrame({
            'index': full_data['index'],
            'episode_index': full_data['episode_index'],
            'frame_index': full_data['frame_index'],
            'timestamp': full_data['timestamp'],
            field_name: np.full(len(full_data), default_value, dtype=data_type)
        })

        return labels_df

    def save_labels(
        self,
        labels_df: pd.DataFrame,
        field_name: str,
        format: str = "parquet"
    ):
        """
        保存标签到文件

        Args:
            labels_df: 标签数据
            field_name: 字段名称
            format: 文件格式 ('parquet', 'csv', 'json')
        """
        self.create_labels_dir()

        if format == "parquet":
            output_path = self.labels_dir / f"{field_name}.parquet"
            labels_df.to_parquet(output_path, index=False)
        elif format == "csv":
            output_path = self.labels_dir / f"{field_name}.csv"
            labels_df.to_csv(output_path, index=False)
        elif format == "json":
            output_path = self.labels_dir / f"{field_name}.json"
            labels_df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"不支持的格式: {format}")

        print(f"\n标签已保存: {output_path}")
        print(f"  文件大小: {output_path.stat().st_size / 1024:.2f} KB")

    def load_labels(self, field_name: str, format: str = "parquet") -> pd.DataFrame:
        """
        加载已有的标签

        Args:
            field_name: 字段名称
            format: 文件格式

        Returns:
            标签 DataFrame
        """
        if format == "parquet":
            label_path = self.labels_dir / f"{field_name}.parquet"
            if not label_path.exists():
                raise FileNotFoundError(f"标签文件不存在: {label_path}")
            return pd.read_parquet(label_path)
        elif format == "csv":
            label_path = self.labels_dir / f"{field_name}.csv"
            if not label_path.exists():
                raise FileNotFoundError(f"标签文件不存在: {label_path}")
            return pd.read_csv(label_path)
        elif format == "json":
            label_path = self.labels_dir / f"{field_name}.json"
            if not label_path.exists():
                raise FileNotFoundError(f"标签文件不存在: {label_path}")
            return pd.read_json(label_path, orient='records')
        else:
            raise ValueError(f"不支持的格式: {format}")

    def create_metadata(self, field_name: str, description: str, default_value: Any):
        """创建标签元数据文件"""
        metadata = {
            "field_name": field_name,
            "description": description,
            "default_value": default_value,
            "total_frames": self.total_frames,
            "created_at": pd.Timestamp.now().isoformat(),
            "dataset": {
                "total_episodes": self.total_episodes,
                "fps": self.fps
            }
        }

        metadata_path = self.labels_dir / f"{field_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"元数据已保存: {metadata_path}")

    def summary(self, labels_df: pd.DataFrame, field_name: str):
        """打印标签摘要信息"""
        print(f"\n{'='*60}")
        print(f"标签摘要: {field_name}")
        print(f"{'='*60}")
        print(f"总帧数: {len(labels_df)}")
        print(f"Episodes: {labels_df['episode_index'].nunique()}")
        print(f"\n按 Episode 统计:")

        for ep_idx in sorted(labels_df['episode_index'].unique()):
            ep_frames = len(labels_df[labels_df['episode_index'] == ep_idx])
            print(f"  Episode {ep_idx}: {ep_frames} 帧")

        print(f"\n字段 '{field_name}' 统计:")
        print(labels_df[field_name].describe())


def main():
    parser = argparse.ArgumentParser(
        description="为 LeRobot 数据集的每一帧添加自定义字段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 添加关键帧标记（整数，默认0）
  python add_frame_labels.py --repo-id hukewei/eval_pro_act5 --field-name is_keyframe

  # 添加重要性分数（浮点数，默认0.0）
  python add_frame_labels.py --repo-id hukewei/eval_pro_act5 \\
      --field-name importance --default-value 0.0 --data-type float64

  # 添加布尔标记
  python add_frame_labels.py --root /path/to/dataset \\
      --field-name needs_review --default-value False --data-type bool

  # 保存为不同格式
  python add_frame_labels.py --repo-id hukewei/eval_pro_act5 \\
      --field-name is_keyframe --format csv
        """
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        help="数据集ID (格式: username/dataset_name)"
    )

    parser.add_argument(
        "--root",
        type=str,
        help="数据集根目录路径（与 --repo-id 二选一）"
    )

    parser.add_argument(
        "--field-name",
        type=str,
        default="is_keyframe",
        help="要添加的字段名称（默认为 'is_keyframe'）"
    )

    parser.add_argument(
        "--default-value",
        type=str,
        default="0",
        help="默认值（默认为 '0'）"
    )

    parser.add_argument(
        "--data-type",
        type=str,
        default="int64",
        choices=["int64", "float64", "bool", "str"],
        help="数据类型（默认为 int64）"
    )

    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["parquet", "csv", "json"],
        help="保存格式（默认为 parquet）"
    )

    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="字段描述（可选）"
    )

    args = parser.parse_args()

    # 确定数据集路径
    if args.root:
        dataset_root = Path(args.root)
    elif args.repo_id:
        home = Path.home()
        dataset_root = home / ".cache" / "huggingface" / "lerobot" / args.repo_id
    else:
        parser.error("必须指定 --repo-id 或 --root")

    if not dataset_root.exists():
        print(f"错误: 数据集路径不存在: {dataset_root}")
        return

    # 转换默认值类型
    if args.data_type == "int64":
        default_value = int(args.default_value)
    elif args.data_type == "float64":
        default_value = float(args.default_value)
    elif args.data_type == "bool":
        default_value = args.default_value.lower() in ['true', '1', 'yes']
    else:  # str
        default_value = args.default_value

    try:
        # 创建标签管理器
        manager = FrameLabelManager(dataset_root)

        # 创建帧标签
        labels_df = manager.create_frame_labels(
            field_name=args.field_name,
            default_value=default_value,
            data_type=args.data_type
        )

        # 保存标签
        manager.save_labels(labels_df, args.field_name, format=args.format)

        # 创建元数据
        description = args.description or f"自动生成的 {args.field_name} 字段"
        manager.create_metadata(args.field_name, description, default_value)

        # 打印摘要
        manager.summary(labels_df, args.field_name)

        print(f"\n{'='*60}")
        print("✓ 完成！")
        print(f"{'='*60}")
        print(f"\n后续操作:")
        print(f"  1. 使用视频浏览器标注关键帧")
        print(f"  2. 修改 {dataset_root}/labels/{args.field_name}.{args.format}")
        print(f"  3. 在训练时读取标签文件")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
