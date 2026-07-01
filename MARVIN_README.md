# Marvin机器人LeRobot集成完整指南

> **Marvin双臂7自由度协作机器人的LeRobot集成文档**  
> 版本：v1.1 | 更新时间：2026-07-01

---

## 目录

1. [机器人规格](#机器人规格)
2. [快速开始](#快速开始)
3. [遥操作](#遥操作)
4. [数据采集](#数据采集)
5. [模型训练](#模型训练)
6. [模型推理](#模型推理)
7. [数据格式](#数据格式)
8. [技术架构](#技术架构)
9. [常见问题](#常见问题)

---

## 机器人规格

### 硬件配置

| 项目 | 规格 |
|------|------|
| **自由度** | 双臂，每臂7个关节 |
| **夹爪** | DM4310电机（可选），485/CAN通信 |
| **控制器** | Marvin SDK通过UDP通信 |
| **默认IP** | 192.168.1.190 |

### 关节配置

**每个臂有7个关节**：
```python
_JOINT_NAMES = [
    "joint_1",  # 基座旋转
    "joint_2",  # 肩部俯仰
    "joint_3",  # 肩部侧摆
    "joint_4",  # 肘部俯仰
    "joint_5",  # 腕部旋转
    "joint_6",  # 腕部俯仰
    "joint_7"   # 末端旋转
]
```

### 数据范围

| 数据类型 | 单位 | 范围 | 说明 |
|---------|------|------|------|
| **关节位置** | 度（°） | 约±180° | 具体范围取决于关节 |
| **关节速度** | 度/秒（°/s） | - | 实时反馈 |
| **关节力矩** | Nm | - | 实时反馈 |
| **夹爪位置** | 弧度（rad） | 可配置 | 默认：-0.5 到 +0.5 rad |

---

## 快速开始

### 1. 验证连接

```bash
ping 192.168.1.190
```

### 2. 测试遥操作（无夹爪）

```bash
lerobot-teleoperate \
    --robot.type=marvin \
    --robot.ip=192.168.1.190 \
    --robot.use_arm=AB \
    --teleop.type=marvin_leader \
    --teleop.ip=192.168.1.190 \
    --teleop.use_arm=AB \
    --display_data=true
```

**预期行为**：
- 两臂移动到初始位置 `[0, 0, 0, -90, 0, 0, 0]`（约3秒）
- A臂（左臂）可自由拖动
- B臂（右臂）精确跟随A臂

### 3. 测试夹爪控制

```bash
lerobot-teleoperate \
    --robot.type=marvin \
    --robot.ip=192.168.1.190 \
    --robot.use_arm=AB \
    --robot.use_gripper=true \
    --robot.left_gripper_id=0x01 \
    --robot.left_gripper_master_id=0x11 \
    --robot.right_gripper_id=0x02 \
    --robot.right_gripper_master_id=0x12 \
    --teleop.type=marvin_leader \
    --teleop.ip=192.168.1.190 \
    --teleop.use_arm=AB \
    --display_data=true
```

**预期行为**：
- 机械臂跟随（同上）
- 左夹爪可手动开合（低刚度）
- 右夹爪自动跟随左夹爪（高刚度）

---

## 遥操作

### CYR遥操作模式

Marvin使用**SDK内置的CYR（Cyber遥操作）模式**：

```
┌─────────────────────────────────┐
│     物理机器人（192.168.1.190）    │
├─────────────────────────────────┤
│ A臂（左臂）：Leader              │
│  - 低刚度，可拖动                 │
│  - 用户手动示教                   │
├─────────────────────────────────┤
│ B臂（右臂）：Follower            │
│  - 高刚度，精确跟随                │
│  - SDK自动控制                    │
└─────────────────────────────────┘
```

### 连接流程

```python
# 1. Leader连接（配置CYR模式）
leader.connect()
  ├─> 连接SDK
  ├─> 配置扭矩模式 + 关节阻抗
  ├─> 开启拖动模式
  ├─> 移动到初始位置 [0,0,0,-90,0,0,0]
  ├─> 设置CYR阻抗参数 K=[400,400,400,400,300,300,300], D=[10,10,10,10,10,10,10]
  └─> 开启CYR模式（A臂可拖动，B臂跟随）

# 2. Follower连接（复用SDK连接）
robot.connect()
  ├─> 检测到SDK已连接
  ├─> 初始化夹爪（如果启用）
  └─> 连接摄像头
```

### 控制逻辑

#### 机械臂控制（SDK硬件级）
```
用户拖动A臂 → SDK控制器检测 → B臂自动跟随
（无需Python代码干预，延迟<5ms）
```

#### 夹爪控制（Python软件级，60Hz）
```python
# 在 Robot.send_action() 中
m1_pos = motor_left.getPosition()    # 读取左夹爪当前位置
m2_target = m1_pos + 0.1             # 右夹爪目标 = 左夹爪 + 偏移

# 左夹爪：低刚度，可拖动
gripper.controlMIT(motor_left, K=0.15, D=0.15, target=m1_pos, ...)

# 右夹爪：高刚度，跟随
gripper.controlMIT(motor_right, K=8.0, D=0.20, target=m2_target, ...)
```

---

## 数据采集

### 基础命令

```bash
lerobot-record \
    --robot.type=marvin \
    --robot.ip=192.168.1.190 \
    --robot.use_arm=AB \
    --robot.use_gripper=true \
    --robot.cameras="{front_cam: {type: opencv, index_or_path: /dev/cam_front, width: 640, height: 480, fps: 30}}" \
    --teleop.type=marvin_leader \
    --teleop.ip=192.168.1.190 \
    --teleop.use_arm=AB \
    --dataset.repo_id=hukewei/marvin_demo \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=6 \
    --dataset.push_to_hub=false \
    --dataset.single_task="pick and place the yellow cube" \
    --dataset.streaming_encoding=false
```

**⚠️ 性能优化建议**：
- `--dataset.streaming_encoding=false` — **强烈推荐**！禁用实时视频编码以避免 CPU 过载导致 EtherCAT 通信失败
- 降低相机分辨率（如 `320x240`）可进一步减轻 CPU 负载
- 录制完成后可用 `ffmpeg` 手动编码视频

### 启用力反馈数据采集

```bash
lerobot-record \
    --robot.type=marvin \
    --robot.ip=192.168.1.190 \
    --robot.use_arm=AB \
    --robot.use_gripper=true \
    --robot.use_force_feedback=true \
    --robot.force_feedback_types='["joint_vel", "joint_torque", "cart_force"]' \
    --robot.cameras="{front_cam: {type: opencv, index_or_path: /dev/cam_front, width: 640, height: 480, fps: 30}}" \
    --teleop.type=marvin_leader \
    --teleop.ip=192.168.1.190 \
    --teleop.use_arm=AB \
    --dataset.repo_id=hukewei/marvin_demo_force \
    --dataset.num_episodes=10 \
    --dataset.streaming_encoding=false
```

**力反馈类型说明**：
- `joint_vel` — 关节速度（deg/s）
- `joint_torque` — 关节传感器力矩（Nm）
- `joint_force` — 关节空间外力（Nm）
- `cart_force` — 末端笛卡尔空间力（Fx, Fy, Fz, Mx, My, Mz）

### 数据采集循环（30Hz）

```python
while recording:
    # 1. 读取机器人状态（Follower B臂）
    obs = robot.get_observation()
    # obs = {
    #   "joint_1.pos": float,   # B臂关节1位置（度）
    #   "joint_2.pos": float,
    #   ...
    #   "joint_7.pos": float,
    #   "gripper.pos": float,   # B臂夹爪位置（弧度）
    #   
    #   # 如果启用 use_force_feedback=true 和对应的 force_feedback_types:
    #   "joint_1.vel": float,      # 关节速度
    #   "joint_1.torque": float,   # 关节力矩
    #   "joint_1.force": float,    # 关节外力
    #   "cart_force.fx": float,    # 末端笛卡尔力
    #   "cart_force.fy": float,
    #   "cart_force.fz": float,
    #   "cart_force.mx": float,
    #   "cart_force.my": float,
    #   "cart_force.mz": float,
    #   
    #   "observation.images.front_cam": np.array(shape=(480,640,3)),
    # }
    
    # 2. 读取leader动作（人类示教的A臂位置）
    action = teleop.get_action()
    # action = {
    #   "joint_1.pos": float,   # A臂关节1位置
    #   ..., 
    #   "joint_7.pos": float,
    #   "gripper.pos": float,   # A臂夹爪位置
    # }
    
    # 3. 执行动作（Follower）
    robot.send_action(action)
    # - 机械臂：CYR模式自动跟随，不发送命令
    # - 夹爪：从硬件实时读取并控制
    
    # 4. 保存数据
    dataset.save(obs, action)
    # 记录：(B臂当前状态, A臂当前位置) 配对
```

### 采集到的数据说明

**重要**：Marvin 只记录 **B 臂（Follower）** 的状态和动作，A 臂仅用于示教。

**语义**：
- `observation`: 当前时刻 B 臂的状态（执行臂的位置）
- `action`: 当前时刻 A 臂的位置（示教目标，也是 B 臂应该到达的位置）

**实际上**：由于 CYR 模式，B 臂实时跟随 A 臂，所以 `observation` 和 `action` 非常接近（微小延迟约 5ms）。

---

## 模型训练

### 训练命令（ACT策略）

```bash
lerobot-train \
    --dataset.repo_id=username/marvin_demo \
    --policy.type=act \
    --output_dir=outputs/train/act_marvin \
    --job_name=act_marvin_demo \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.repo_id=username/act_marvin_policy \
    --batch_size=8 \
    --policy.use_amp=true
```

### 模型输入/输出

**输入（Observation）**：
```python
{
    "observation.state": [8个float],  # B臂关节+夹爪位置（不包含力反馈）
    "observation.images.front_cam": [H, W, 3],
}
```

**如果启用力反馈** (`use_force_feedback=true`)：
```python
{
    "observation.state": [8 + N个float],  # 位置 + 力反馈数据
    # N 取决于 force_feedback_types:
    # - joint_vel: +7
    # - joint_torque: +7
    # - joint_force: +7
    # - cart_force: +6
    "observation.images.front_cam": [H, W, 3],
}
```

**输出（Action）**：
```python
{
    "action": [8个float]  # B臂关节+夹爪目标位置
}
```

---

## 模型推理

### 推理命令

```bash
lerobot-record \
    --robot.type=marvin \
    --robot.ip=192.168.1.190 \
    --robot.use_arm=AB \
    --robot.use_gripper=true \
    --robot.cameras="{ \
        wrist_cam: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, \
        top_cam: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30} \
    }" \
    --policy.path=outputs/train/act_marvin/checkpoints/last/pretrained_model \
    --dataset.repo_id=username/marvin_eval \
    --dataset.num_episodes=10 \
    --dataset.push_to_hub=false
```


**注意**：推理时不需要`--teleop`参数！

### 推理循环（30Hz）

```python
while evaluating:
    # 1. 读取当前状态（B臂）
    obs = robot.get_observation()
    # obs = {
    #   "joint_1.pos": float,  # B臂位置
    #   ...
    #   "joint_7.pos": float,
    #   "gripper.pos": float,
    #   "observation.images.front_cam": np.array(...),
    # }
    
    # 2. 模型预测下一步动作
    predicted_action = policy(obs)
    # predicted_action = {
    #   "joint_1.pos": float,
    #   ...
    #   "joint_7.pos": float,
    #   "gripper.pos": float,
    # }
    
    # 3. 执行预测动作
    robot.send_action(predicted_action)
    # - 机械臂：发送位置命令（position模式或impedance模式，非CYR）
    # - 夹爪：发送目标位置（MIT模式，高刚度）
    
    # 4. 记录评估结果
    dataset.save(obs, predicted_action)
```

### 控制接口：Robot.send_action()

这是**唯一**执行动作的接口，包含模式切换逻辑：

```python
def send_action(self, action: dict) -> dict:
    """
    根据模式执行动作：
    - 遥操作模式（already_connected=True）：夹爪从硬件读取实时控制
    - 推理模式（already_connected=False）：夹爪+机械臂从action读取
    """
    
    _, _, already_connected = get_sdk(self.config.ip)
    
    # ========== 夹爪控制 ==========
    if self.config.use_gripper:
        if already_connected:
            # 遥操作模式：实时读取硬件
            m1_pos = motor_left.getPosition()
            m2_target = m1_pos + 0.1
            gripper.controlMIT(motor_left, 0.15, 0.15, m1_pos, ...)
            gripper.controlMIT(motor_right, 8.0, 0.20, m2_target, ...)
        else:
            # 推理模式：从action读取（只控制B臂）
            gripper_target = action["gripper.pos"]
            gripper.controlMIT(motor_right, 8.0, 0.20, gripper_target, ...)
    
    # ========== 机械臂控制 ==========
    if not already_connected:
        # 推理模式：发送位置命令到B臂
        joints = [action[f"joint_{i+1}.pos"] for i in range(7)]
        sdk.set_joint_cmd_pose(arm='B', joints=joints)
        sdk.send_cmd()
    # 遥操作模式：不发送命令（CYR自动处理）
    
    return action
```

**关键点**：
- **遥操作时**：`already_connected=True`，机械臂由CYR控制，夹爪实时读取
- **推理时**：`already_connected=False`，机械臂和夹爪都从`action`参数读取
- **只控制 B 臂**：A 臂仅在遥操作时用作示教设备

---

## 数据格式

### Observation（观测状态）

**默认配置**（`use_force_feedback=false`）：
```python
observation = {
    # ========== B臂位置（仅位置） ==========
    "joint_1.pos": float,     # 度（°）
    "joint_2.pos": float,
    "joint_3.pos": float,
    "joint_4.pos": float,
    "joint_5.pos": float,
    "joint_6.pos": float,
    "joint_7.pos": float,
    "gripper.pos": float,     # 弧度（rad）
    
    # ========== 摄像头图像 ==========
    "observation.images.front_cam": np.ndarray,  # shape=(H, W, 3), dtype=uint8
}
```

**启用力反馈**（`use_force_feedback=true` + `force_feedback_types=[...]`）：
```python
observation = {
    # ========== B臂位置（始终包含） ==========
    "joint_1.pos": float,
    ...
    "joint_7.pos": float,
    
    # ========== 力反馈数据（可选） ==========
    # 如果 "joint_vel" in force_feedback_types:
    "joint_1.vel": float,     # 度/秒（°/s）
    ...
    "joint_7.vel": float,
    
    # 如果 "joint_torque" in force_feedback_types:
    "joint_1.torque": float,  # Nm
    ...
    "joint_7.torque": float,
    
    # 如果 "joint_force" in force_feedback_types:
    "joint_1.force": float,   # Nm（关节空间外力）
    ...
    "joint_7.force": float,
    
    # 如果 "cart_force" in force_feedback_types:
    "cart_force.fx": float,   # N（末端笛卡尔力）
    "cart_force.fy": float,
    "cart_force.fz": float,
    "cart_force.mx": float,   # Nm（末端力矩）
    "cart_force.my": float,
    "cart_force.mz": float,
    
    "gripper.pos": float,
    
    # ========== 摄像头图像 ==========
    "observation.images.front_cam": np.ndarray,
}
```

**总计**：
- **基础**：7关节位置 + 1夹爪 = 8个值
- **+ joint_vel**：+7 = 15个值
- **+ joint_torque**：+7 = 22个值
- **+ joint_force**：+7 = 29个值
- **+ cart_force**：+6 = 35个值
- **摄像头**：N个图像

**重要**：只记录 **B 臂（Follower）** 数据，A 臂仅用于遥操作示教。

### Action（动作命令）

```python
action = {
    "joint_1.pos": float,   # 度（°）
    "joint_2.pos": float,
    "joint_3.pos": float,
    "joint_4.pos": float,
    "joint_5.pos": float,
    "joint_6.pos": float,
    "joint_7.pos": float,
    "gripper.pos": float,   # 弧度（rad）
}
```

**总计**：
- **B臂**：7关节 + 1夹爪 = 8个值
- **只包含位置**，不包含速度和力矩

**注意**：
- 在**遥操作模式**下，`action` 是 A 臂的当前位置（人类示教）
- 在**推理模式**下，`action` 是模型预测的 B 臂目标位置

### observation.state 的顺序

在 LeRobot 的数据集中，`observation.state` 是一个一维数组：

**默认配置**（`use_force_feedback=false`）：
```python
observation.state = np.array([
    joint_1.pos,
    joint_2.pos,
    joint_3.pos,
    joint_4.pos,
    joint_5.pos,
    joint_6.pos,
    joint_7.pos,
    gripper.pos,
])  # shape=(8,)
```

**启用所有力反馈**（`force_feedback_types=["joint_vel", "joint_torque", "joint_force", "cart_force"]`）：
```python
observation.state = np.array([
    joint_1.pos,
    joint_2.pos,
    ...,
    joint_7.pos,
    joint_1.vel,
    ...,
    joint_7.vel,
    joint_1.torque,
    ...,
    joint_7.torque,
    joint_1.force,
    ...,
    joint_7.force,
    cart_force.fx,
    cart_force.fy,
    cart_force.fz,
    cart_force.mx,
    cart_force.my,
    cart_force.mz,
    gripper.pos,
])  # shape=(35,)
```

**注意**：只包含 B 臂数据，不包含 A 臂。

---

## 技术架构

### SDK子进程隔离

```
┌─────────────────────────────────────┐
│       主进程（LeRobot + PyTorch）     │
│                                     │
│  ┌──────────┐      ┌──────────┐   │
│  │  Robot   │      │  Leader  │   │
│  └────┬─────┘      └────┬─────┘   │
│       │                 │          │
│       └────────┬────────┘          │
│                │                   │
│         multiprocessing.Pipe       │
│                │                   │
└────────────────┼───────────────────┘
                 │
┌────────────────┼───────────────────┐
│                ▼                   │
│      SDK子进程（Marvin SDK）        │
│                                     │
│  ┌─────────────────────────────┐  │
│  │  libMarvinSDK.so (C++)      │  │
│  │  - UDP通信                   │  │
│  │  - CYR控制                   │  │
│  │  - 485/CAN通信               │  │
│  └─────────────────────────────┘  │
└─────────────────────────────────────┘
                 │
                 │ UDP
                 ▼
┌─────────────────────────────────────┐
│    Marvin机器人控制器                │
│    IP: 192.168.1.190                │
└─────────────────────────────────────┘
```

**为什么需要子进程隔离？**
- Marvin SDK的C++库与PyTorch的C++库冲突
- 子进程中不加载PyTorch，避免符号冲突

### 连接池管理

```python
# sdk_pool.py
_pool = {
    "192.168.1.190": {
        "process": subprocess.Popen(...),  # SDK子进程
        "pipe": multiprocessing.Pipe(),     # 通信管道
        "lock": threading.Lock(),           # 串行访问
        "refcount": 2,                      # Leader + Robot
    }
}
```

**共享连接**：
- Leader和Robot连接同一IP时，共享SDK连接
- `refcount`跟踪引用计数
- 最后一个断开时才真正关闭SDK

### 数据流

```
遥操作数据流：
┌──────────┐
│ 用户拖动  │
│   A臂    │
└────┬─────┘
     │
     ▼
┌──────────────────────┐
│  SDK控制器（硬件）    │
│  - A臂：可拖动        │
│  - B臂：自动跟随      │
└──┬──────────────┬────┘
   │              │
   │ SDK.subscribe()
   │              │
   ▼              ▼
┌─────────┐  ┌─────────┐
│ Leader  │  │  Robot  │
│get_action│ │get_obs  │
└────┬────┘  └────┬────┘
     │            │
     └─────┬──────┘
           ▼
    ┌──────────────┐
    │   Dataset    │
    │ save(obs, a) │
    └──────────────┘

推理数据流：
┌──────────┐
│  Robot   │
│ get_obs  │
└────┬─────┘
     │
     ▼
┌──────────┐
│  Policy  │
│ predict  │
└────┬─────┘
     │
     ▼
┌──────────┐
│  Robot   │
│send_action│
│          │
│ - 机械臂位置控制 │
│ - 夹爪MIT控制   │
└──────────┘
```

---

## 常见问题

### Q1: B臂不跟随A臂？

**检查**：
```bash
# 日志中应该看到：
INFO: Set CYR KD params: 1
INFO: Enabled CYR teleoperation mode: 1
INFO: After CYR on: armA state=3, armB state=3
```

**如果返回0**：CYR模式开启失败，检查机器人状态和连接。

### Q2: 夹爪无法控制？

**检查**：
1. 夹爪CAN ID是否正确（默认：0x01, 0x02）
2. 夹爪是否已上电
3. 485通信是否正常

**调试**：运行原始SDK demo验证：
```bash
cd src/lerobot/Marvin_sdk/DEMO_PYTHON
python joint_drag_gripper.py
```

### Q3: 数据集中action没有夹爪？

**修复**：确保使用最新代码，已修复此问题。

**验证**：
```python
from lerobot.robots.marvin import MarvinRobot, MarvinRobotConfig
from lerobot.teleoperators.marvin_leader import MarvinLeader, MarvinLeaderConfig

robot = MarvinRobot(MarvinRobotConfig(use_gripper=True))
leader = MarvinLeader(MarvinLeaderConfig())

print(robot.action_features)   # 应包含 'gripper.pos'
print(leader.action_features)  # 应包含 'gripper.pos'
```

### Q4: 推理时机器人不动？

**检查**：
1. 模型输出的action范围是否正确（度 vs 弧度）
2. 是否在CYR模式下推理（应该关闭CYR）
3. 查看日志中的`already_connected`状态

### Q5: 485调试信息太多？

```
[Marvin SDK]: Set 485 of B arm: channel =1
```

这是SDK底层C++库打印的，无法在Python层关闭。不影响功能，可以忽略。

### Q6: 帧率低于30Hz？

**优化**：
1. 减少摄像头数量或降低分辨率
2. 关闭`--display_data`
3. 确保网络延迟<5ms
4. 检查CPU/GPU负载

### Q7: B臂失控/EtherCAT通信失败？⚠️

**症状**：
- B臂突然不受控制，位置无法追踪
- 日志中出现：
  ```
  [ERRO] Arm1: Domain WC changed to 21
  [ERRO] Arm1: Domain WC changed to 18
  ...
  [ERRO] Arm1: Domain WC changed to 0
  [WARN] Arm1: Domain state changed to 0
  ```
- 或者：
  ```
  [WARN] rt_task_wait_period error: -110, error count: 395
  ```

**根本原因**：
- CPU 实时任务超时（-110 = ETIMEDOUT）
- EtherCAT 通信失败（Domain Working Counter 从 24 降到 0）
- 通常由视频实时编码 + 相机捕获 + 数据写入同时运行导致 CPU 过载

**解决方案**：

1. **禁用实时视频编码**（最有效）：
   ```bash
   --dataset.streaming_encoding=false
   ```
   录制完成后手动编码：
   ```bash
   cd data/your_dataset/videos
   ffmpeg -i episode_0.raw -c:v libx264 episode_0.mp4
   ```

2. **降低相机分辨率**：
   ```bash
   --robot.cameras="{cam: {type: opencv, width: 320, height: 240, fps: 30}}"
   ```

3. **减少对齐频率**：
   - 每个 episode 只对齐一次
   - 避免频繁按键触发重新对齐

4. **检查网线质量**：
   ```bash
   ethtool -S eth0 | grep -E "error|drop|crc"
   ```
   如果有 CRC 错误，更换网线

5. **提高实时任务优先级**（控制器端）：
   ```bash
   chrt -f 99 -p $(pidof robot_controller)
   ```

### Q8: 如何启用力反馈数据采集？

**命令行参数**：
```bash
--robot.use_force_feedback=true \
--robot.force_feedback_types='["joint_vel", "joint_torque", "cart_force"]'
```

**验证**：
```python
from lerobot.datasets import LeRobotDataset
dataset = LeRobotDataset("your_username/your_dataset")
print(dataset.meta.observation_features)
# 应包含 joint_1.vel, joint_1.torque, cart_force.fx 等字段
```

**注意**：
- 力反馈数据会增加数据集大小（约 4× 原始大小）
- 默认配置只记录位置，保持向后兼容

---

## 配置参数参考

### Robot配置

```bash
--robot.type=marvin
--robot.ip=192.168.1.190
--robot.use_arm=AB                    # A, B, 或 AB
--robot.control_mode=impedance        # "position" 或 "impedance"
--robot.vel_ratio=10                  # 速度限制百分比 (1-100)
--robot.acc_ratio=10                  # 加速度限制百分比 (1-100)

# 阻抗控制参数（仅当 control_mode="impedance" 时有效）
--robot.impedance_k='[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]'  # 刚度 (Nm/deg)
--robot.impedance_d='[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]'  # 阻尼 (Nm/(deg/s))

# 夹爪配置
--robot.use_gripper=true
--robot.left_gripper_id=0x01
--robot.left_gripper_master_id=0x11
--robot.right_gripper_id=0x02
--robot.right_gripper_master_id=0x12
--robot.gripper_open_pos=-0.5         # 弧度
--robot.gripper_close_pos=0.5         # 弧度

# 工具重力补偿参数
--robot.enable_gravity_compensation=true
--robot.tool_mass_a=0.7               # A臂工具质量 (kg)
--robot.tool_mass_b=1.1               # B臂工具质量 (kg)
--robot.tool_com_a='[0.0, 0.0, 50.0]'  # A臂工具质心 (mm, X/Y/Z)
--robot.tool_com_b='[0.0, 0.0, 50.0]'  # B臂工具质心 (mm, X/Y/Z)

# 力反馈配置
--robot.use_force_feedback=true       # 启用力反馈数据采集
--robot.force_feedback_types='["joint_vel", "joint_torque", "cart_force", "joint_force"]'
# 可选类型：
# - "joint_vel": 关节速度 (deg/s)
# - "joint_torque": 关节传感器力矩 (Nm)
# - "joint_force": 关节空间外力 (Nm)
# - "cart_force": 末端笛卡尔空间力 (Fx, Fy, Fz, Mx, My, Mz)

# 其他
--robot.disable_torque_on_disconnect=true  # 断开时是否下电
```

### Teleop配置

```bash
--teleop.type=marvin_leader
--teleop.ip=192.168.1.190
--teleop.use_arm=AB
--teleop.align_position="[0,0,0,-90,0,0,0]"  # 可选，自定义初始位置
```

### 摄像头配置

```bash
--robot.cameras="{ \
    wrist_cam: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, \
    top_cam: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30} \
}"
```

---

## 相关文件

- **机器人实现**：`src/lerobot/robots/marvin/marvin_robot.py`
- **遥操作实现**：`src/lerobot/teleoperators/marvin_leader/marvin_leader.py`
- **SDK子进程**：`src/lerobot/robots/marvin/sdk_worker.py`
- **连接池**：`src/lerobot/robots/marvin/sdk_pool.py`
- **配置类**：`src/lerobot/robots/marvin/config_marvin.py`

---

## 总结

Marvin集成已完全支持LeRobot的标准工作流：

| 功能 | 状态 |
|------|------|
| ✅ 双臂遥操作（CYR模式） | 完成 |
| ✅ 夹爪控制（MIT模式） | 完成 |
| ✅ 数据采集 | 完成 |
| ✅ 模型训练 | 完成 |
| ✅ 模型推理 | 完成 |
| ✅ SDK子进程隔离 | 完成 |
| ✅ 连接池管理 | 完成 |
| ✅ 力反馈数据采集 | 完成 |
| ✅ 工具重力补偿 | 完成 |

**开始使用**：参考[快速开始](#快速开始)章节！

如有问题，请提issue或查看详细日志。🚀

---

## 补充说明

### 数据记录逻辑

**遥操作模式（数据采集）**：
- A臂（Leader，用户拖动） → Action（示教动作）
- B臂（Follower，跟随执行） → Observation（机器人状态）

**数据集记录**：
- `observation.state` = B臂位置（8个值，默认配置）
- `action` = A臂位置（8个值）

**推理模式**：
- Policy输出 → Action（预测动作）
- B臂执行 → Observation（机器人状态）

**数据流**：
- `observation.state` = B臂当前位置
- `policy(obs)` → `action`（预测的B臂目标）
- `robot.send_action(action)` → 控制B臂移动

### 重要提示

1. **只记录 B 臂数据** — A 臂仅用于遥操作示教，不记录到数据集
2. **默认只记录位置** — 速度、力矩等需要启用 `use_force_feedback=true`
3. **禁用实时编码** — 录制时务必使用 `--dataset.streaming_encoding=false` 避免 CPU 过载
4. **控制频率 30Hz** — 数据采集和推理都运行在 30Hz（`dataset.fps=30`）
