# 关键帧标注与训练指南

本指南介绍如何为 LeRobot 数据集标注关键帧，并在训练时使用关键帧加权采样。

## 概述

关键帧加权采样能让训练时更频繁地采样重要的帧，从而提高模型对关键动作的学习效果。

**工作原理：**
- `is_keyframe=0`: 普通帧，每个epoch平均采样1次
- `is_keyframe=1`: 关键帧，每个epoch平均采样2次

## 使用流程

### 1. 创建标签文件

首先为数据集创建 `is_keyframe` 标签文件（所有帧默认值为0）：

```bash
python keyframe_label/add_frame_labels.py \
    --repo-id /home/marvin/hhw/peg_optical_module_0707 \
    --field-name is_keyframe
```

这会创建：
- `labels/is_keyframe.parquet`: 标签数据文件
- `labels/is_keyframe_metadata.json`: 元数据文件

### 2. 标注关键帧

使用可视化工具浏览数据集并标注关键帧：

```bash
python keyframe_label/video_browser.py \
    --repo-id /home/marvin/hhw/peg_optical_module_0707
```

**操作说明：**
- 拖动滑块浏览帧
- 按 `k` 键标记当前帧为关键帧（`is_keyframe=1`）
- 按 `u` 键取消标记（`is_keyframe=0`）
- 按 `p`/`n` 键切换 episode
- 按 `q` 键退出并保存

**标注建议：**
- 标记任务的关键阶段（如抓取、放置、接触等）
- 标记动作发生变化的时刻
- 标记任务成功/失败的关键帧
- 一般标注5-10%的帧作为关键帧

### 3. 训练模型

训练脚本会自动检测 `labels/is_keyframe.parquet` 文件：

```bash
lerobot-train \
  --dataset.repo_id=hhw/peg_optical_module_0707 \
  --dataset.root=/home/marvin/hhw/peg_optical_module_0707 \
  --policy.type=act \
  --output_dir=/home/marvin/hhw/peg_optical_module_0707/model \
  --job_name=peg_optical_module_0707 \
  --policy.device=cuda \
  --wandb.enable=false \
  --batch_size=2 \
  --policy.use_amp=true \
  --steps=200000
```

**训练日志示例：**
```
检测到关键帧标签文件，加载中...
✓ 关键帧标签已加载: 12000 个关键帧
  普通帧权重: 1.0, 关键帧权重: 2.0
  关键帧占比: 5.21%
✓ 使用加权采样器（关键帧权重=2.0）
```

如果没有标签文件，训练会正常进行（所有帧权重相同）。

## 文件结构

```
dataset_root/
├── data/              # 原始数据文件（不修改）
├── videos/            # 视频文件（不修改）
├── meta/              # 元数据（不修改）
└── labels/            # 标签文件（新增）
    ├── is_keyframe.parquet           # 关键帧标签
    └── is_keyframe_metadata.json     # 元数据
```

## 工具脚本

### add_frame_labels.py
创建标签文件：
```bash
python keyframe_label/add_frame_labels.py \
    --repo-id /path/to/dataset \
    --field-name is_keyframe \
    --default-value 0
```

### video_browser.py
可视化浏览和标注：
```bash
python keyframe_label/video_browser.py \
    --repo-id /path/to/dataset \
    [--camera camera_name]
```

### test_keyframe_training.py
测试标签加载和采样：
```bash
python test_keyframe_training.py
```

## 采样权重详解

### 权重计算
- 普通帧：权重 = 1.0
- 关键帧：权重 = 2.0

### 采样概率
假设数据集有 230,378 帧，其中 12,000 个关键帧：

- 总权重 = 218,378 × 1.0 + 12,000 × 2.0 = 242,378
- 单个关键帧采样概率 = 2.0 / 242,378 = 0.00000825
- 单个普通帧采样概率 = 1.0 / 242,378 = 0.00000413
- 关键帧相对采样率 = 2.0x

### 每个 Epoch 采样次数
由于采样是无放回的，每个 epoch 会采样 230,378 次（总帧数）：

- 关键帧总采样次数 ≈ 23,812 次（约占10.3%）
- 普通帧总采样次数 ≈ 206,566 次（约占89.7%）
- 平均每个关键帧被采样 ≈ 1.98 次
- 平均每个普通帧被采样 ≈ 0.95 次

## 常见问题

### Q: 关键帧权重可以调整吗？
A: 可以。编辑 `src/lerobot/scripts/lerobot_train.py`，修改：
```python
keyframe_weights[idx] = 2.0  # 改为其他值，如 3.0、5.0 等
```

### Q: 可以只训练关键帧吗？
A: 可以。使用 `KeyframeOnlySampler`（需要修改训练脚本）。

### Q: 标签文件很大吗？
A: 不大。230,378 帧的标签文件约 1.6 MB。

### Q: 如何批量标注关键帧？
A: 可以编写脚本根据规则自动标注。例如：
```python
from lerobot.datasets.label_loader import LabelLoader
import pandas as pd

# 加载标签
label_loader = LabelLoader("/path/to/dataset")
df = pd.read_parquet("/path/to/dataset/labels/is_keyframe.parquet")

# 每隔30帧标记一个关键帧
df.loc[df.index % 30 == 0, 'is_keyframe'] = 1

# 保存
df.to_parquet("/path/to/dataset/labels/is_keyframe.parquet", index=False)
```

### Q: 训练时如何知道采样是否生效？
A: 查看训练日志中的信息：
```
✓ 关键帧标签已加载: 12000 个关键帧
✓ 使用加权采样器（关键帧权重=2.0）
```

## 实现细节

修改的文件：
1. `src/lerobot/datasets/__init__.py`: 导出 `WeightedEpisodeAwareSampler` 和 `LabelLoader`
2. `src/lerobot/scripts/lerobot_train.py`: 添加自动检测和加权采样逻辑
3. `src/lerobot/datasets/label_loader.py`: 标签加载工具
4. `src/lerobot/datasets/weighted_sampler.py`: 加权采样器

核心逻辑：
```python
# 1. 检测标签文件
label_path = Path(dataset_root) / "labels" / "is_keyframe.parquet"
if label_path.exists():
    # 2. 加载标签
    label_loader.load_label("is_keyframe")
    
    # 3. 计算权重
    keyframe_weights = np.ones(num_frames)
    for idx in keyframe_indices:
        keyframe_weights[idx] = 2.0
    
    # 4. 创建加权采样器
    sampler = WeightedEpisodeAwareSampler(
        ...,
        frame_weights=keyframe_weights,
        ...
    )
```

## 参考资料

- LeRobot 文档: https://github.com/huggingface/lerobot
- ACT 论文: 强调关键帧对模仿学习的重要性
- 加权采样: PyTorch WeightedRandomSampler

## 联系方式

如有问题，请查看 `keyframe_label/` 目录中的示例代码。
