#!/bin/bash

# 关键帧加权训练完整示例
# 数据集: /home/marvin/hhw/peg_optical_module_0707

echo "========================================"
echo "关键帧加权训练完整流程"
echo "========================================"

DATASET_PATH="/home/marvin/hhw/peg_optical_module_0707"

# 步骤 1: 创建关键帧标签文件
echo -e "\n[步骤 1/4] 创建关键帧标签文件..."
python keyframe_label/add_frame_labels.py \
    --repo-id "$DATASET_PATH" \
    --field-name is_keyframe \
    --default-value 0

if [ $? -eq 0 ]; then
    echo "✓ 标签文件创建成功"
else
    echo "✗ 标签文件创建失败"
    exit 1
fi

# 步骤 2: 使用视频浏览器标记关键帧
echo -e "\n[步骤 2/4] 打开视频浏览器标记关键帧..."
echo "提示: 使用以下快捷键操作"
echo "  k - 标记当前帧为关键帧"
echo "  u - 取消关键帧标记"
echo "  p/n - 切换上一个/下一个 episode"
echo "  空格 - 暂停/播放"
echo "  q - 退出并保存"
echo ""
echo "按回车键打开视频浏览器..."
read

python keyframe_label/video_browser.py \
    --repo-id "$DATASET_PATH"

# 步骤 3: 测试采样功能（可选）
echo -e "\n[步骤 3/4] 测试关键帧采样..."
echo "是否运行采样测试？(y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    python test_keyframe_sampling.py \
        --dataset-root "$DATASET_PATH" \
        --num-epochs 3
fi

# 步骤 4: 开始训练
echo -e "\n[步骤 4/4] 开始训练..."
echo "是否开始训练？(y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    lerobot-train \
        --dataset.repo_id=hhw/peg_optical_module_0707 \
        --dataset.root=/home/marvin/hhw/peg_optical_module_0707 \
        --policy.type=act \
        --output_dir=/home/marvin/hhw/peg_optical_module_0707/model \
        --job_name=peg_optical_module_0707_keyframe \
        --policy.device=cuda \
        --wandb.enable=false \
        --batch_size=2 \
        --policy.use_amp=true \
        --steps=200000
else
    echo "跳过训练"
    echo ""
    echo "如需手动训练，使用以下命令："
    echo "lerobot-train \\"
    echo "  --dataset.repo_id=hhw/peg_optical_module_0707 \\"
    echo "  --dataset.root=/home/marvin/hhw/peg_optical_module_0707 \\"
    echo "  --policy.type=act \\"
    echo "  --output_dir=/home/marvin/hhw/peg_optical_module_0707/model \\"
    echo "  --job_name=peg_optical_module_0707_keyframe \\"
    echo "  --policy.device=cuda \\"
    echo "  --wandb.enable=false \\"
    echo "  --batch_size=2 \\"
    echo "  --policy.use_amp=true \\"
    echo "  --steps=200000"
fi

echo ""
echo "========================================"
echo "流程完成"
echo "========================================"
