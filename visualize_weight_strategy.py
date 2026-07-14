#!/usr/bin/env python3
"""
可视化渐进式关键帧权重分配策略

生成示意图展示权重如何在关键帧周围分布
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_weight_diagram(output_path: Path = None):
    """创建权重分配示意图"""

    # 创建一个示例场景：100 帧，3 个关键帧
    total_frames = 100
    keyframe_indices = [25, 50, 75]

    # 构建权重
    weights = np.ones(total_frames, dtype=np.float32)

    for kf_idx in keyframe_indices:
        # 关键帧
        weights[kf_idx] = 20.0

        # 前后 16-30 帧：权重 5
        for offset in range(16, 31):
            if 0 <= kf_idx - offset < total_frames:
                weights[kf_idx - offset] = max(weights[kf_idx - offset], 5.0)
            if 0 <= kf_idx + offset < total_frames:
                weights[kf_idx + offset] = max(weights[kf_idx + offset], 5.0)

        # 前后 1-15 帧：权重 10
        for offset in range(1, 16):
            if 0 <= kf_idx - offset < total_frames:
                weights[kf_idx - offset] = max(weights[kf_idx - offset], 10.0)
            if 0 <= kf_idx + offset < total_frames:
                weights[kf_idx + offset] = max(weights[kf_idx + offset], 10.0)

    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # ============ 子图 1: 权重条形图 ============
    ax1 = axes[0]

    # 为不同权重分配颜色
    colors = []
    for w in weights:
        if w == 20:
            colors.append('#e74c3c')  # 红色 - 关键帧
        elif w == 10:
            colors.append('#f39c12')  # 橙色 - 近邻帧
        elif w == 5:
            colors.append('#f1c40f')  # 黄色 - 中等帧
        else:
            colors.append('#ecf0f1')  # 浅灰 - 普通帧

    ax1.bar(range(total_frames), weights, color=colors, width=1.0, edgecolor='none')

    # 标记关键帧
    for kf_idx in keyframe_indices:
        ax1.axvline(x=kf_idx, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.text(kf_idx, 21, f'KF{kf_idx}', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color='red')

    # 标注权重区域
    ax1.axhspan(0, 1, alpha=0.1, color='gray', label='普通帧 (权重=1)')
    ax1.axhspan(4, 6, alpha=0.1, color='yellow', label='中等帧 (权重=5)')
    ax1.axhspan(9, 11, alpha=0.1, color='orange', label='近邻帧 (权重=10)')
    ax1.axhspan(19, 21, alpha=0.1, color='red', label='关键帧 (权重=20)')

    ax1.set_xlabel('帧索引 (Frame Index)', fontsize=11)
    ax1.set_ylabel('采样权重 (Sampling Weight)', fontsize=11)
    ax1.set_title('渐进式关键帧权重分配 - 整体视图', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 22)
    ax1.set_xlim(-2, total_frames + 2)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # ============ 子图 2: 单个关键帧的详细视图 ============
    ax2 = axes[1]

    # 聚焦到第二个关键帧周围
    focus_kf = keyframe_indices[1]
    focus_range = 45
    focus_start = max(0, focus_kf - focus_range)
    focus_end = min(total_frames, focus_kf + focus_range)

    focus_weights = weights[focus_start:focus_end]
    focus_indices = range(focus_start, focus_end)
    focus_colors = colors[focus_start:focus_end]

    ax2.bar(focus_indices, focus_weights, color=focus_colors, width=1.0, edgecolor='none')

    # 标记区域边界
    ax2.axvline(x=focus_kf, color='red', linestyle='-', linewidth=2, label='关键帧')
    ax2.axvline(x=focus_kf - 15, color='orange', linestyle='--', alpha=0.7, linewidth=1,
                label='近邻边界 (±15)')
    ax2.axvline(x=focus_kf + 15, color='orange', linestyle='--', alpha=0.7, linewidth=1)
    ax2.axvline(x=focus_kf - 30, color='yellow', linestyle=':', alpha=0.7, linewidth=1,
                label='中等边界 (±30)')
    ax2.axvline(x=focus_kf + 30, color='yellow', linestyle=':', alpha=0.7, linewidth=1)

    # 添加文字标注
    ax2.text(focus_kf, 20.5, '关键帧\n权重=20', ha='center', va='bottom', fontsize=9,
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.8,
            edgecolor='none'), color='white')

    ax2.text(focus_kf - 8, 10.5, '近邻帧\n权重=10', ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.8, edgecolor='none'),
            color='white')

    ax2.text(focus_kf - 23, 5.5, '中等帧\n权重=5', ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#f1c40f', alpha=0.8, edgecolor='none'),
            color='white')

    ax2.text(focus_kf - 38, 1.5, '普通帧\n权重=1', ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#95a5a6', alpha=0.8, edgecolor='none'),
            color='white')

    ax2.set_xlabel('帧索引 (Frame Index)', fontsize=11)
    ax2.set_ylabel('采样权重 (Sampling Weight)', fontsize=11)
    ax2.set_title(f'单个关键帧周围的权重分布 (关键帧 #{focus_kf})', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 22)
    ax2.set_xlim(focus_start - 2, focus_end + 2)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # ============ 子图 3: 权重统计 ============
    ax3 = axes[2]

    # 统计各权重级别的帧数
    weight_counts = {
        20: np.sum(weights == 20),
        10: np.sum(weights == 10),
        5: np.sum(weights == 5),
        1: np.sum(weights == 1)
    }

    weight_labels = ['关键帧\n(权重=20)', '近邻帧\n(权重=10)', '中等帧\n(权重=5)', '普通帧\n(权重=1)']
    weight_values = [weight_counts[20], weight_counts[10], weight_counts[5], weight_counts[1]]
    weight_colors_bar = ['#e74c3c', '#f39c12', '#f1c40f', '#ecf0f1']

    bars = ax3.bar(weight_labels, weight_values, color=weight_colors_bar, edgecolor='black',
                   linewidth=1.5)

    # 添加数值标签
    for bar, value in zip(bars, weight_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(value)} 帧\n({value/total_frames*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax3.set_ylabel('帧数 (Frame Count)', fontsize=11)
    ax3.set_title('各权重级别的帧数统计', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, max(weight_values) * 1.2)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加平均权重信息
    avg_weight = np.mean(weights)
    ax3.text(0.98, 0.95, f'平均权重: {avg_weight:.2f}',
            transform=ax3.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 权重示意图已保存: {output_path}")
    else:
        plt.show()


def create_comparison_diagram(output_path: Path = None):
    """创建简单加权 vs 渐进式加权的对比图"""

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 示例场景
    total_frames = 80
    keyframe_idx = 40

    # ========== 简单加权策略 ==========
    simple_weights = np.ones(total_frames, dtype=np.float32)
    simple_weights[keyframe_idx] = 2.0

    ax1 = axes[0]
    colors1 = ['#e74c3c' if i == keyframe_idx else '#ecf0f1' for i in range(total_frames)]
    ax1.bar(range(total_frames), simple_weights, color=colors1, width=1.0, edgecolor='none')
    ax1.axvline(x=keyframe_idx, color='red', linestyle='--', linewidth=2)
    ax1.set_ylabel('权重', fontsize=11)
    ax1.set_title('策略 A: 简单加权 (关键帧=2x, 其他=1x)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 22)
    ax1.set_xlim(-2, total_frames + 2)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加注释
    ax1.text(keyframe_idx, 2.5, '关键帧 (2×)', ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7), color='white')

    # ========== 渐进式加权策略 ==========
    progressive_weights = np.ones(total_frames, dtype=np.float32)
    progressive_weights[keyframe_idx] = 20.0

    for offset in range(16, 31):
        if 0 <= keyframe_idx - offset < total_frames:
            progressive_weights[keyframe_idx - offset] = 5.0
        if 0 <= keyframe_idx + offset < total_frames:
            progressive_weights[keyframe_idx + offset] = 5.0

    for offset in range(1, 16):
        if 0 <= keyframe_idx - offset < total_frames:
            progressive_weights[keyframe_idx - offset] = 10.0
        if 0 <= keyframe_idx + offset < total_frames:
            progressive_weights[keyframe_idx + offset] = 10.0

    ax2 = axes[1]
    colors2 = []
    for w in progressive_weights:
        if w == 20:
            colors2.append('#e74c3c')
        elif w == 10:
            colors2.append('#f39c12')
        elif w == 5:
            colors2.append('#f1c40f')
        else:
            colors2.append('#ecf0f1')

    ax2.bar(range(total_frames), progressive_weights, color=colors2, width=1.0, edgecolor='none')
    ax2.axvline(x=keyframe_idx, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=keyframe_idx - 15, color='orange', linestyle=':', alpha=0.7)
    ax2.axvline(x=keyframe_idx + 15, color='orange', linestyle=':', alpha=0.7)
    ax2.axvline(x=keyframe_idx - 30, color='yellow', linestyle=':', alpha=0.7)
    ax2.axvline(x=keyframe_idx + 30, color='yellow', linestyle=':', alpha=0.7)

    ax2.set_xlabel('帧索引', fontsize=11)
    ax2.set_ylabel('权重', fontsize=11)
    ax2.set_title('策略 B: 渐进式加权 (关键帧=20x, 近邻=10x, 中等=5x, 其他=1x)',
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 22)
    ax2.set_xlim(-2, total_frames + 2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加注释
    ax2.text(keyframe_idx, 20.5, '关键帧\n(20×)', ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.8), color='white')
    ax2.text(keyframe_idx - 8, 10.5, '近邻\n(10×)', ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.8), color='white')
    ax2.text(keyframe_idx - 23, 5.5, '中等\n(5×)', ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#f1c40f', alpha=0.8), color='white')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 对比图已保存: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="可视化渐进式关键帧权重分配策略")
    parser.add_argument("--output-dir", type=str, default=".", help="输出目录")
    parser.add_argument("--show", action="store_true", help="显示图形而不是保存")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("生成权重分配示意图...")

    if args.show:
        create_weight_diagram()
        create_comparison_diagram()
    else:
        create_weight_diagram(output_dir / "progressive_weight_diagram.png")
        create_comparison_diagram(output_dir / "weight_strategy_comparison.png")
        print("\n✓ 所有图形已生成")


if __name__ == "__main__":
    main()
