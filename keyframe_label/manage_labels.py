#!/usr/bin/env python3
"""
读取和更新 LeRobot 数据集的帧标签

用途：
- 读取已创建的帧标签
- 更新特定帧的标签值
- 批量修改标签
- 导出统计信息

用法：
    # 读取标签
    python manage_labels.py --repo-id username/dataset_name --field-name is_keyframe --action show

    # 设置某一帧为关键帧
    python manage_labels.py --repo-id username/dataset_name --field-name is_keyframe \\
        --action update --episode 0 --frame 10 --value 1

    # 批量设置关键帧
    python manage_labels.py --repo-id username/dataset_name --field-name is_keyframe \\
        --action batch-update --keyframes "0:10,0:50,1:20,1:100"
"""

import argparse
import json
from pathlib import Path

import pandas as pd


class LabelEditor:
    """标签编辑器"""

    def __init__(self, dataset_root: Path, field_name: str, format: str = "parquet"):
        """
        初始化标签编辑器

        Args:
            dataset_root: 数据集根目录
            field_name: 字段名称
            format: 文件格式
        """
        self.dataset_root = Path(dataset_root)
        self.field_name = field_name
        self.format = format
        self.labels_dir = self.dataset_root / "labels"

        # 加载标签
        self.labels_df = self._load_labels()
        self.metadata = self._load_metadata()

        print(f"已加载标签: {field_name}")
        print(f"  总帧数: {len(self.labels_df)}")

    def _load_labels(self) -> pd.DataFrame:
        """加载标签文件"""
        if self.format == "parquet":
            label_path = self.labels_dir / f"{self.field_name}.parquet"
            return pd.read_parquet(label_path)
        elif self.format == "csv":
            label_path = self.labels_dir / f"{self.field_name}.csv"
            return pd.read_csv(label_path)
        elif self.format == "json":
            label_path = self.labels_dir / f"{self.field_name}.json"
            return pd.read_json(label_path, orient='records')
        else:
            raise ValueError(f"不支持的格式: {self.format}")

    def _load_metadata(self) -> dict:
        """加载元数据"""
        metadata_path = self.labels_dir / f"{self.field_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def save(self):
        """保存修改后的标签"""
        if self.format == "parquet":
            output_path = self.labels_dir / f"{self.field_name}.parquet"
            self.labels_df.to_parquet(output_path, index=False)
        elif self.format == "csv":
            output_path = self.labels_dir / f"{self.field_name}.csv"
            self.labels_df.to_csv(output_path, index=False)
        elif self.format == "json":
            output_path = self.labels_dir / f"{self.field_name}.json"
            self.labels_df.to_json(output_path, orient='records', indent=2)

        print(f"\n✓ 标签已保存: {output_path}")

    def show(self, episode: int | None = None):
        """显示标签信息"""
        print(f"\n{'='*60}")
        print(f"标签信息: {self.field_name}")
        print(f"{'='*60}")

        if episode is not None:
            # 显示特定 episode 的标签
            ep_data = self.labels_df[self.labels_df['episode_index'] == episode]
            if len(ep_data) == 0:
                print(f"Episode {episode} 不存在")
                return

            print(f"\nEpisode {episode}:")
            print(f"  总帧数: {len(ep_data)}")
            print(f"  {self.field_name} 统计:")
            print(ep_data[self.field_name].value_counts().to_string())

            # 显示非零值的帧
            non_zero = ep_data[ep_data[self.field_name] != 0]
            if len(non_zero) > 0:
                print(f"\n  非零值的帧:")
                for _, row in non_zero.iterrows():
                    print(f"    Frame {row['frame_index']}: {row[self.field_name]}")
        else:
            # 显示所有 episodes 的统计
            print(f"\n总帧数: {len(self.labels_df)}")
            print(f"Episodes: {self.labels_df['episode_index'].nunique()}")

            print(f"\n字段 '{self.field_name}' 全局统计:")
            print(self.labels_df[self.field_name].value_counts().to_string())

            print(f"\n按 Episode 统计:")
            for ep_idx in sorted(self.labels_df['episode_index'].unique()):
                ep_data = self.labels_df[self.labels_df['episode_index'] == ep_idx]
                non_zero_count = len(ep_data[ep_data[self.field_name] != 0])
                print(f"  Episode {ep_idx}: {len(ep_data)} 帧, {non_zero_count} 个非零值")

    def update_frame(self, episode: int, frame: int, value):
        """更新单个帧的标签"""
        mask = (self.labels_df['episode_index'] == episode) & \
               (self.labels_df['frame_index'] == frame)

        if mask.sum() == 0:
            print(f"错误: 未找到 Episode {episode}, Frame {frame}")
            return False

        self.labels_df.loc[mask, self.field_name] = value
        print(f"✓ 已更新 Episode {episode}, Frame {frame} -> {value}")
        return True

    def batch_update(self, keyframes: str):
        """
        批量更新标签

        Args:
            keyframes: 格式为 "ep:frame,ep:frame,..." 或 "ep:frame:value,..."
                      例如: "0:10,0:50,1:20" 或 "0:10:1,0:50:2,1:20:1"
        """
        updates = []

        for item in keyframes.split(','):
            parts = item.strip().split(':')
            if len(parts) == 2:
                ep, frame = int(parts[0]), int(parts[1])
                value = 1  # 默认值
            elif len(parts) == 3:
                ep, frame, value = int(parts[0]), int(parts[1]), parts[2]
                # 尝试转换为合适的类型
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # 保持字符串
            else:
                print(f"警告: 跳过无效格式: {item}")
                continue

            updates.append((ep, frame, value))

        print(f"\n批量更新 {len(updates)} 个帧:")
        success_count = 0

        for ep, frame, value in updates:
            if self.update_frame(ep, frame, value):
                success_count += 1

        print(f"\n完成: {success_count}/{len(updates)} 个帧已更新")

    def reset(self, episode: int | None = None):
        """重置标签为默认值"""
        default_value = self.metadata.get('default_value', 0)

        if episode is not None:
            mask = self.labels_df['episode_index'] == episode
            self.labels_df.loc[mask, self.field_name] = default_value
            print(f"✓ 已重置 Episode {episode} 的所有标签为 {default_value}")
        else:
            self.labels_df[self.field_name] = default_value
            print(f"✓ 已重置所有标签为 {default_value}")

    def export_keyframes(self, output_path: Path | None = None):
        """导出所有非零值的帧（关键帧）"""
        non_zero = self.labels_df[self.labels_df[self.field_name] != 0]

        if output_path is None:
            output_path = self.labels_dir / f"{self.field_name}_keyframes.json"

        keyframes = {}
        for _, row in non_zero.iterrows():
            ep = int(row['episode_index'])
            frame = int(row['frame_index'])
            value = row[self.field_name]

            if ep not in keyframes:
                keyframes[ep] = []

            keyframes[ep].append({
                'frame': frame,
                'timestamp': float(row['timestamp']),
                'value': value
            })

        with open(output_path, 'w') as f:
            json.dump(keyframes, f, indent=2)

        print(f"\n✓ 关键帧已导出: {output_path}")
        print(f"  总计: {len(non_zero)} 个关键帧")


def main():
    parser = argparse.ArgumentParser(
        description="读取和更新 LeRobot 数据集的帧标签",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 显示标签信息
  python manage_labels.py --repo-id hukewei/eval_pro_act5 --field-name is_keyframe --action show

  # 显示特定 episode 的标签
  python manage_labels.py --repo-id hukewei/eval_pro_act5 --field-name is_keyframe \\
      --action show --episode 0

  # 更新单个帧
  python manage_labels.py --repo-id hukewei/eval_pro_act5 --field-name is_keyframe \\
      --action update --episode 0 --frame 10 --value 1

  # 批量更新（设置关键帧）
  python manage_labels.py --repo-id hukewei/eval_pro_act5 --field-name is_keyframe \\
      --action batch-update --keyframes "0:10,0:50,0:100,1:20,1:80"

  # 批量更新（带不同的值）
  python manage_labels.py --repo-id hukewei/eval_pro_act5 --field-name importance \\
      --action batch-update --keyframes "0:10:0.8,0:50:1.0,1:20:0.6"

  # 重置所有标签
  python manage_labels.py --repo-id hukewei/eval_pro_act5 --field-name is_keyframe \\
      --action reset

  # 导出关键帧列表
  python manage_labels.py --repo-id hukewei/eval_pro_act5 --field-name is_keyframe \\
      --action export
        """
    )

    parser.add_argument("--repo-id", type=str, help="数据集ID")
    parser.add_argument("--root", type=str, help="数据集根目录路径")
    parser.add_argument("--field-name", type=str, required=True, help="字段名称")
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["parquet", "csv", "json"],
        help="文件格式"
    )
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["show", "update", "batch-update", "reset", "export"],
        help="操作类型"
    )
    parser.add_argument("--episode", type=int, help="Episode 索引")
    parser.add_argument("--frame", type=int, help="帧索引")
    parser.add_argument("--value", type=str, help="要设置的值")
    parser.add_argument("--keyframes", type=str, help="批量更新的关键帧列表")
    parser.add_argument("--output", type=str, help="导出文件路径")

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

    try:
        editor = LabelEditor(dataset_root, args.field_name, format=args.format)

        if args.action == "show":
            editor.show(episode=args.episode)

        elif args.action == "update":
            if args.episode is None or args.frame is None or args.value is None:
                parser.error("update 操作需要 --episode, --frame 和 --value 参数")

            # 转换值类型
            try:
                value = int(args.value)
            except ValueError:
                try:
                    value = float(args.value)
                except ValueError:
                    value = args.value

            editor.update_frame(args.episode, args.frame, value)
            editor.save()

        elif args.action == "batch-update":
            if args.keyframes is None:
                parser.error("batch-update 操作需要 --keyframes 参数")

            editor.batch_update(args.keyframes)
            editor.save()

        elif args.action == "reset":
            editor.reset(episode=args.episode)
            editor.save()

        elif args.action == "export":
            output_path = Path(args.output) if args.output else None
            editor.export_keyframes(output_path)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
