# LeRobot 仓库更新方案 - 保留 Marvin 适配

## 当前状态

**你的版本**: 0.5.2 (自定义)  
**官方最新**: 0.5.1 release

**你的自定义内容**:
- Marvin 机器人适配: `src/lerobot/robots/marvin/`
- Marvin 遥操作: `src/lerobot/teleoperators/marvin_leader/`
- 数据格式改动: 力反馈支持
- 训练脚本和配置
- 数据转换脚本

---

## 🎯 更新目标

获取官方最新代码（包括 lerobot-rollout），同时保留你的 Marvin 适配。

---

## 📋 推荐方案：创建新分支 + 选择性合并

### 步骤 1: 备份当前工作

```bash
cd /home/marvin/hhw/lerobot_marvin

# 1. 提交当前所有更改
git add .
git commit -m "Backup: 保存当前 Marvin 适配和数据转换功能"

# 2. 创建备份分支
git branch backup-marvin-v0.5.2
git push origin backup-marvin-v0.5.2
```

### 步骤 2: 添加官方上游仓库

```bash
# 添加官方仓库（如果还没有）
git remote add upstream https://github.com/huggingface/lerobot.git

# 获取官方最新代码
git fetch upstream --tags
```

### 步骤 3: 创建更新分支

```bash
# 创建新分支用于更新
git checkout -b update-to-latest

# 查看官方最新的提交
git log upstream/main --oneline | head -20
```

### 步骤 4: 合并官方代码

**方案 A: 保守合并（推荐）**

```bash
# 合并官方 main 分支到你的分支
git merge upstream/main --no-commit --no-ff

# 查看冲突
git status
```

**方案 B: 重置后应用补丁（更彻底）**

```bash
# 仅用于参考，先不执行
# git reset --hard upstream/main
# git cherry-pick <your-marvin-commits>
```

### 步骤 5: 解决冲突

预计会有冲突的文件：
- `pyproject.toml` - 依赖和版本
- `src/lerobot/scripts/` - 可能的脚本更新
- `src/lerobot/datasets/` - 数据集相关
- `src/lerobot/processor/` - 特征处理

**处理策略**:
1. **保留你的 Marvin 代码**（robot/teleoperator）
2. **接受官方更新**（rollout 等新功能）
3. **手动合并**（有冲突的配置文件）

### 步骤 6: 重新安装

```bash
# 在 lerobot_marvin 环境中重新安装
conda activate lerobot_marvin

# 卸载旧版本
pip uninstall lerobot -y

# 安装新版本（开发模式）
pip install -e .
```

---

## 🔧 详细操作指南

### 1. 备份关键文件

在开始前，先备份你的关键定制文件：

```bash
# 创建备份目录
mkdir -p ~/lerobot_marvin_backup

# 备份 Marvin 相关代码
cp -r src/lerobot/robots/marvin ~/lerobot_marvin_backup/
cp -r src/lerobot/teleoperators/marvin_leader ~/lerobot_marvin_backup/

# 备份自定义脚本
cp -r scripts ~/lerobot_marvin_backup/

# 备份配置
cp pyproject.toml ~/lerobot_marvin_backup/

echo "✓ 备份完成: ~/lerobot_marvin_backup"
```

### 2. 识别你的关键修改

```bash
# 查看你相对于某个基准的修改
git log --oneline --graph --decorate | head -20

# 查看修改的文件
git diff --name-only upstream/main..HEAD

# 查看具体修改
git diff upstream/main..HEAD -- pyproject.toml
git diff upstream/main..HEAD -- src/lerobot/datasets/
```

### 3. 执行合并

```bash
cd /home/marvin/hhw/lerobot_marvin

# 确保在 main 分支
git checkout main

# 创建更新分支
git checkout -b update-with-rollout

# 获取官方最新代码
git fetch upstream

# 合并（会显示冲突）
git merge upstream/main
```

### 4. 解决冲突

#### pyproject.toml 冲突
```bash
# 手动编辑，保留：
# - 你的依赖 (pandas, pyarrow)
# - 官方的新依赖
# - 官方的新命令 (lerobot-rollout)
```

#### 其他冲突
```bash
# 查看冲突文件
git status | grep "both modified"

# 对每个冲突文件
git checkout --ours <file>   # 保留你的版本
git checkout --theirs <file> # 使用官方版本
# 或手动编辑文件解决冲突
```

### 5. 验证关键功能

```bash
# 检查 Marvin 机器人是否存在
ls -la src/lerobot/robots/marvin/

# 检查 rollout 是否存在
ls -la src/lerobot/rollout/ 2>/dev/null || echo "需要从官方获取"

# 检查命令定义
grep "lerobot-" pyproject.toml
```

### 6. 测试安装

```bash
conda activate lerobot_marvin

# 重新安装
pip install -e .

# 验证版本
python -c "import lerobot; print(lerobot.__version__)"

# 验证命令
which lerobot-record
which lerobot-rollout

# 测试导入 Marvin
python -c "from lerobot.robots.marvin import Marvin; print('✓ Marvin OK')"
```

---

## 🚨 风险评估

### 低风险（应该能平滑合并）
- ✅ Marvin 机器人代码（独立目录）
- ✅ Marvin 遥操作代码（独立目录）
- ✅ 你的训练脚本（在 scripts/ 下）

### 中风险（可能需要调整）
- ⚠️ `pyproject.toml` - 依赖和版本号
- ⚠️ 数据处理流程 - 如果官方改动了 feature_utils
- ⚠️ 数据集格式 - 如果官方改动了 LeRobotDataset

### 高风险（可能需要重写）
- 🔴 如果你修改了核心 API
- 🔴 如果官方重构了架构

---

## 📝 合并检查清单

更新后需要验证的功能：

- [ ] Marvin 机器人可以正常导入
- [ ] Marvin 遥操作可以正常导入  
- [ ] lerobot-record 命令存在
- [ ] lerobot-rollout 命令存在
- [ ] 数据转换脚本正常工作
- [ ] 力反馈特征处理正确
- [ ] 可以加载旧的数据集
- [ ] 可以训练模型

---

## 🎯 预期结果

更新完成后你会得到：

```
lerobot_marvin/ (最新版本)
├── src/lerobot/
│   ├── robots/
│   │   ├── marvin/           ← 你的代码（保留）
│   │   └── ...               ← 官方代码（更新）
│   ├── teleoperators/
│   │   ├── marvin_leader/    ← 你的代码（保留）
│   │   └── ...               ← 官方代码（更新）
│   ├── rollout/              ← 官方新功能（获得）
│   │   ├── strategies/
│   │   └── inference/
│   ├── scripts/
│   │   ├── lerobot_record.py    ← 官方更新
│   │   ├── lerobot_rollout.py   ← 官方新增
│   │   └── ...
│   └── ...
├── scripts/
│   ├── convert_force_feedback_pure_offline.py  ← 你的脚本（保留）
│   └── ...
└── pyproject.toml            ← 合并后的版本
```

---

## 🔄 回滚方案

如果更新出现问题，可以快速回滚：

```bash
# 回到备份分支
git checkout backup-marvin-v0.5.2

# 或者强制重置
git reset --hard backup-marvin-v0.5.2

# 重新安装
conda activate lerobot_marvin
pip install -e .
```

---

## 💡 替代方案：并行安装

如果担心破坏现有环境，可以：

```bash
# 克隆官方最新版本到新目录
cd /home/marvin/hhw
git clone https://github.com/huggingface/lerobot.git lerobot_official_latest

# 复制你的 Marvin 代码
cp -r lerobot_marvin/src/lerobot/robots/marvin \
      lerobot_official_latest/src/lerobot/robots/

cp -r lerobot_marvin/src/lerobot/teleoperators/marvin_leader \
      lerobot_official_latest/src/lerobot/teleoperators/

# 创建新环境
conda create -n lerobot_latest python=3.12
conda activate lerobot_latest
cd lerobot_official_latest
pip install -e .

# 测试新环境
python -c "from lerobot.robots.marvin import Marvin; print('OK')"
```

这样你可以：
- 保留 `lerobot_marvin` 环境（稳定版本）
- 测试 `lerobot_latest` 环境（最新功能）
- 确认无误后再迁移

---

## 📞 需要帮助？

执行任何步骤前，我可以帮你：
1. 先执行一次模拟合并，看看会有什么冲突
2. 生成详细的冲突解决脚本
3. 创建自动化的备份和恢复脚本

你想先执行哪个方案？我可以为你生成具体的命令。
