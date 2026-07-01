# Marvin机器人LeRobot集成完整指南

> **Marvin双臂7自由度协作机器人的LeRobot集成文档**  
> 版本：v1.0 | 更新时间：2026-06-04

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

### 命令

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
    --dataset.single_task="pick and place the yellow cube"
```

### 数据采集循环（60Hz）

```python
while recording:
    # 1. 读取机器人状态（Follower）
    obs = robot.get_observation()
    # obs = {
    #   "observation.state": [
    #     left_joint_1.pos, ..., left_joint_7.pos, left_gripper.pos,
    #     right_joint_1.pos, ..., right_joint_7.pos, right_gripper.pos
    #   ],  # 16个值（7+1）×2
    #   "observation.images.wrist_cam": np.array(shape=(480,640,3)),
    #   "observation.images.top_cam": np.array(shape=(480,640,3)),
    # }
    
    # 2. 读取leader动作（人类示教）
    action = teleop.get_action()
    # action = {
    #   "left_joint_1.pos": ..., 
    #   ..., 
    #   "left_gripper.pos": ...,  # A臂当前位置
    #   "right_joint_1.pos": ..., 
    #   ..., 
    #   "right_gripper.pos": ...  # B臂当前位置
    # }
    
    # 3. 执行动作（Follower）
    robot.send_action(action)
    # - 机械臂：CYR模式自动跟随，不发送命令
    # - 夹爪：从硬件实时读取并控制
    
    # 4. 保存数据
    dataset.save(obs, action)
    # 记录：(当前状态, 当前位置) 配对
```

### 采集到的数据说明

**语义**：
- `observation`: 当前时刻的状态（B臂的位置）
- `action`: 当前时刻的目标（A臂的位置，也是B臂应该到达的位置）

**实际上**：由于CYR模式，B臂实时跟随A臂，所以`observation`和`action`非常接近（微小延迟）。

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
    "observation.state": [16个float],  # 双臂关节+夹爪位置
    "observation.images.wrist_cam": [H, W, 3],
    "observation.images.top_cam": [H, W, 3],
}
```

**输出（Action）**：
```python
{
    "action": [16个float]  # 双臂关节+夹爪目标位置
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

### 推理循环（60Hz）

```python
while evaluating:
    # 1. 读取当前状态
    obs = robot.get_observation()
    # obs = {
    #   "observation.state": [16个float],
    #   "observation.images.wrist_cam": np.array(...),
    #   "observation.images.top_cam": np.array(...),
    # }
    
    # 2. 模型预测下一步动作
    predicted_action = policy(obs)
    # predicted_action = {
    #   "left_joint_1.pos": ...,
    #   ...
    #   "left_gripper.pos": ...,
    #   "right_joint_1.pos": ...,
    #   ...
    #   "right_gripper.pos": ...,
    # }
    
    # 3. 执行预测动作
    robot.send_action(predicted_action)
    # - 机械臂：发送位置命令（position模式，非CYR）
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
            # 推理模式：从action读取
            left_target = action["left_gripper.pos"]
            right_target = action["right_gripper.pos"]
            gripper.controlMIT(motor_left, 8.0, 0.20, left_target, ...)
            gripper.controlMIT(motor_right, 8.0, 0.20, right_target, ...)
    
    # ========== 机械臂控制 ==========
    if not already_connected:
        # 推理模式：发送位置命令
        for arm in ['A', 'B']:
            joints = [action[f"{prefix}joint_{i}.pos"] for i in range(1, 8)]
            sdk.set_joint_cmd_pose(arm=arm, joints=joints)
        sdk.send_cmd()
    # 遥操作模式：不发送命令（CYR自动处理）
    
    return action
```

**关键点**：
- **遥操作时**：`already_connected=True`，机械臂由CYR控制，夹爪实时读取
- **推理时**：`already_connected=False`，机械臂和夹爪都从`action`参数读取

---

## 数据格式

### Observation（观测状态）

```python
observation = {
    # ========== 机械臂状态 ==========
    "left_joint_1.pos": float,     # 度（°）
    "left_joint_1.vel": float,     # 度/秒（°/s）
    "left_joint_1.torque": float,  # Nm
    "left_joint_2.pos": float,
    ...
    "left_joint_7.torque": float,
    
    "left_gripper.pos": float,     # 弧度（rad）
    
    "right_joint_1.pos": float,
    ...
    "right_joint_7.torque": float,
    
    "right_gripper.pos": float,
    
    # ========== 摄像头图像 ==========
    "observation.images.wrist_cam": np.ndarray,  # shape=(H, W, 3), dtype=uint8
    "observation.images.top_cam": np.ndarray,
}
```

**总计**：
- **左臂**：7关节×3（pos/vel/torque）+ 1夹爪 = 22个值
- **右臂**：7关节×3 + 1夹爪 = 22个值
- **摄像头**：N个图像

### Action（动作命令）

```python
action = {
    "left_joint_1.pos": float,   # 度（°）
    "left_joint_2.pos": float,
    ...
    "left_joint_7.pos": float,
    "left_gripper.pos": float,   # 弧度（rad）
    
    "right_joint_1.pos": float,
    ...
    "right_joint_7.pos": float,
    "right_gripper.pos": float,
}
```

**总计**：
- **左臂**：7关节 + 1夹爪 = 8个值
- **右臂**：7关节 + 1夹爪 = 8个值
- **总共**：16个float

### observation.state的顺序

在LeRobot的数据集中，`observation.state`是一个一维数组：

```python
observation.state = np.array([
    left_joint_1.pos,
    left_joint_2.pos,
    left_joint_3.pos,
    left_joint_4.pos,
    left_joint_5.pos,
    left_joint_6.pos,
    left_joint_7.pos,
    left_gripper.pos,
    right_joint_1.pos,
    right_joint_2.pos,
    right_joint_3.pos,
    right_joint_4.pos,
    right_joint_5.pos,
    right_joint_6.pos,
    right_joint_7.pos,
    right_gripper.pos,
])  # shape=(16,)
```

**注意**：只包含位置，不包含速度和力矩。

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

print(robot.action_features)   # 应包含 'left_gripper.pos'
print(leader.action_features)  # 应包含 'left_gripper.pos'
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

### Q6: 帧率低于60Hz？

**优化**：
1. 减少摄像头数量或降低分辨率
2. 关闭`--display_data`
3. 确保网络延迟<5ms
4. 检查CPU/GPU负载

---

## 配置参数参考

### Robot配置

```bash
--robot.type=marvin
--robot.ip=192.168.1.190
--robot.use_arm=AB                    # A, B, 或 AB
--robot.use_gripper=true
--robot.left_gripper_id=0x01
--robot.left_gripper_master_id=0x11
--robot.right_gripper_id=0x02
--robot.right_gripper_master_id=0x12
--robot.gripper_open_pos=-0.5         # 弧度
--robot.gripper_close_pos=0.5         # 弧度
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

**开始使用**：参考[快速开始](#快速开始)章节！

如有问题，请提issue或查看详细日志。🚀


遥操作模式（数据采集）

A臂（Leader，用户拖动） → Action（示教动作）
B臂（Follower，跟随执行） → Observation（机器人状态）

数据集记录：
- observation.state = B臂位置（8个值）
- action = A臂位置（8个值）
推理模式

Policy输出 → Action（预测动作）
B臂执行 → Observation（机器人状态）

数据流：
- observation.state = B臂当前位置
- policy(obs) → action（预测的B臂目标）
- robot.send_action(action) → 控制B臂移动
