# DROID PPO Pi0.5 训练指南

## 目录结构

```
RLinf/
├── run_droid_ppo_pi05.sh                          # 一键启动脚本
├── models/Pi05-DROID-JointPos/                    # Pi0.5 预训练权重
├── logs/                                          # 训练日志和 checkpoint
├── examples/embodiment/config/
│   ├── droid_ppo_pi05.yaml                        # 训练总配置
│   └── env/droid_task.yaml                        # 环境默认配置
├── rlinf/envs/isaaclab/
│   ├── __init__.py                                # 环境注册表
│   ├── isaaclab_env.py                            # RLinf 基类（已有，未修改）
│   ├── venv.py                                    # 子进程通信（已有，未修改）
│   └── tasks/
│       ├── droid_env_cfg.py                       # IsaacLab 环境定义（核心）
│       ├── droid_task.py                          # RLinf wrapper
│       ├── init_franka.py                         # Franka 机器人配置
│       └── objaverse_targets_16.json              # 16 个物体池数据
└── rlinf/envs/wrappers/
    └── record_video.py                            # 视频录制 wrapper（已有，未修改）
```

---

## 前置检查

### 1. 模型权重
```bash
ls RLinf/models/Pi05-DROID-JointPos/
# 应包含: config.json  model.safetensors  assets/
```

### 2. 场景文件
```bash
ls /home/ruijieli0814/nerako/reasoning/fine-tune/env_file/scene/2026-03-25_21-44-26/scene.usdc
```

### 3. 物体资产
```bash
ls /home/ruijieli0814/nerako/reasoning/fine-tune/env_file/assets/ | wc -l
# 应有 16 个物体目录
```

### 4. GPU 确认
```bash
nvidia-smi
# 需要 GPU 4, 5, 6, 7 可用
# env: GPU4 | rollout: GPU5 | actor: GPU6+7
```

---

## 启动训练

```bash
cd /home/ruijieli0814/nerako/reasoning/fine-tune/RLinf
bash run_droid_ppo_pi05.sh
```

脚本自动完成：
- 激活 `venv_isaaclab` 虚拟环境
- 加载 Isaac Sim 环境变量（`isaac_sim/setup_conda_env.sh`）
- 设置 Vulkan、NCCL、Ray 等参数
- 启动训练，日志保存到 `logs/{timestamp}-droid_ppo_pi05/train.log`

---

## 训练配置

### GPU 分配

| 组件 | GPU | 说明 |
|------|-----|------|
| env | GPU 4 | IsaacLab 仿真（8 train + 4 eval 环境） |
| rollout | GPU 5 | Pi0.5 推理，生成 action |
| actor | GPU 6, 7 | PPO 训练，FSDP no_shard |

### 关键超参

| 参数 | 值 | 说明 |
|------|-----|------|
| train envs | 8 | 并行训练环境数 |
| eval envs | 4 | 并行评估环境数 |
| max_episode_steps | 150 | 每个 episode 最多步数 |
| num_action_chunks | 15 | 模型每次推理输出 15 步 action |
| chunk steps / rollout | 10 | `150 / 15 = 10` 次模型推理 |
| rollout_epoch | 1 | 每次收集 1 轮 rollout |
| update_epoch | 2 | 每轮 rollout 后 PPO 更新 2 遍 |
| global_batch_size | 16 | PPO 更新的 batch 大小 |
| max_epochs | 4 | 总训练轮数 |

### 一轮训练流程

```
每个大 epoch:
  1. 收集 rollout
     8 env × 150 步 = 1200 条 transition
     （10 次 chunk_step，每次模型推理 → 执行 15 步）

  2. PPO 更新 × 2 遍
     global_batch_size = 16

  3. eval（4 env × 150 步）

  4. 保存 checkpoint（每 2 个 epoch）
  5. 保存训练/eval 视频
```

---

## 输出文件

```
logs/{timestamp}-droid_ppo_pi05/
├── train.log                    # 完整训练日志
├── tensorboard/                 # TensorBoard 日志
├── video/
│   ├── train/seed_0/            # 每个 epoch 一个 .mp4（外部相机视角，8 env 拼图）
│   └── eval/seed_0/             # eval 视频
└── checkpoints/                 # 模型 checkpoint（每 2 epoch 保存）
```

### 查看 TensorBoard
```bash
tensorboard --logdir logs/{timestamp}-droid_ppo_pi05/tensorboard
```

---

## 修改配置

### 修改 GPU 编号
编辑 `examples/embodiment/config/droid_ppo_pi05.yaml`：
```yaml
cluster:
  component_placement:
    env: 0        # 改为你的 GPU 编号
    rollout:
      placement: 1
    actor: 2-3
```

### 修改场景文件路径
编辑 `examples/embodiment/config/env/droid_task.yaml`：
```yaml
init_params:
  scene_usdc_path: "/your/path/to/scene.usdc"
```

### 修改环境数量
编辑 `examples/embodiment/config/droid_ppo_pi05.yaml`：
```yaml
env:
  train:
    total_num_envs: 8   # 训练环境数
  eval:
    total_num_envs: 4   # 评估环境数
```

---

## 任务说明

### 场景
- Franka 机器人 + Robotiq 2F-85 夹爪
- 桌面场景（由 `scene.usdc` 加载）
- 每个 episode 从 16 个 Objaverse 物体中随机选 **1 个 target + 2 个 distractor** 放置到桌面

### 观测
| Key | Shape | 说明 |
|-----|-------|------|
| `main_images` | [N, 256, 256, 3] | 外部相机 RGB |
| `wrist_images` | [N, 256, 256, 3] | 腕部相机 RGB |
| `states` | [N, 8] | 7D 关节角 + 1D 夹爪开度 |
| `task_descriptions` | list[str] | 如 `"pick up the apple"` |

### 动作
- 8D：7D 关节位置 + 1D 夹爪（>0.5 = 合拢）

### 奖励
- 夹爪接触到桌面物体：+1.0（per step）
- 超时：episode 结束，无额外惩罚

---

## 文件详解

### 调用链总览

```
run_droid_ppo_pi05.sh
  → train_embodied_agent.py + Hydra(droid_ppo_pi05.yaml)
    → env_worker 查 __init__.py 注册表 → 创建 IsaaclabDroidEnv (droid_task.py)
      → IsaaclabBaseEnv.__init__ (isaaclab_env.py)
        → SubProcIsaacLabEnv 启动子进程 (venv.py)
          → 子进程: _make_env_function 闭包执行
            → import droid_env_cfg（触发 gym.register）
            → gym.make("DROID-Franka-JointPos-Visuomotor-v0")
              → IsaacLab ManagerBasedRLEnv(EnvCfg)
                → SceneCfg: 加载 scene.usdc + Franka + 16 个 RigidObject + 相机 + 传感器
                → startup 事件: 藏旧物体
                → reset 事件: 随机选物体放桌面
          → obs/reward 通过 mp.Queue 传回主进程
      → _wrap_obs: 转成 {main_images, wrist_images, states, task_descriptions}
    → rollout_worker: Pi0.5 推理生成 action chunks
    → actor_worker: PPO 更新模型权重
```

---

### 文件 1: `droid_env_cfg.py` — IsaacLab 环境定义（核心，651 行）

**只在子进程中运行**（依赖 `pxr`、`isaaclab` 等 IsaacLab 专属包，主进程没有）。
定义了 IsaacLab `ManagerBasedRLEnv` 的全部配置：场景、动作、观测、奖励、事件。

#### 模块级变量（L34-52）

| 变量 | 说明 |
|------|------|
| `SCENE_USDC_PATH` | 场景 USDC 文件的默认路径（可被 yaml 覆盖） |
| `ASSET_POOL_JSON` | 指向 `objaverse_targets_16.json` |
| `ASSET_POOL` | `{uid: info_dict}` 字典，16 个物体的完整元信息 |
| `ASSET_UIDS` | 16 个 uid 的有序列表，保证 index 稳定 |
| `ASSET_INFOS` | 16 个 info_dict 的有序列表，与 ASSET_UIDS 对应 |
| `NUM_POOL` | 常量 16 |

#### 函数

##### `_load_asset_pool()` (L42-45)
- **用途**：读取 `objaverse_targets_16.json`，返回 `{uid: info_dict}`
- **调用时机**：模块导入时执行一次

##### `_get_stage_from_isaaclab()` (L55-69)
- **用途**：获取当前 USD Stage 对象
- **逻辑**：先试 `sim_utils.get_current_stage()`，再试 `SimulationContext.instance().stage`
- **被调用**：`spawn_add_scene_as_sublayer()`、`startup_make_assets_interactive()`

##### `_print_assets_like_paths(stage, max_print=30)` (L72-88)
- **用途**：调试用，遍历 stage 打印包含 "Assets" 的 prim 路径（最多 30 条）
- **被调用**：`spawn_add_scene_as_sublayer()` 加载场景后打印

##### `spawn_add_scene_as_sublayer(*args, **kwargs)` (L91-137)
- **用途**：IsaacLab spawn 回调函数，把 scene.usdc 作为 USD sublayer 添加到 stage
- **流程**：
  1. 从参数中提取 `UsdFileCfg`，获取 `usd_path`
  2. 验证文件存在
  3. 获取 stage，切换 cwd 到 usdc 所在目录（USD 相对路径需要）
  4. 将 usd_path 添加到 `root_layer.subLayerPaths`
  5. 删除 USDC 中冲突的 `/physicsScene`、`/Replicator` prim
  6. 打印 Assets 路径用于调试
  7. 创建占位 Xform prim `/World/_SceneSublayer` 并返回
- **被调用**：`SceneCfg.scene_sublayer_loader` 的 `spawn.func`

##### `_apply_collision_to_mesh_descendants(root_prim)` (L140-149)
- **用途**：递归遍历 prim 下所有 Mesh 子节点，添加 `CollisionAPI` + `MeshCollisionAPI`（convexHull 近似）
- **被调用**：`_ensure_dynamic_rigidbody()`

##### `_ensure_dynamic_rigidbody(xform_prim, mass)` (L152-185)
- **用途**：确保一个 prim 是动态刚体——添加 `RigidBodyAPI`、设置非 kinematic、添加 `MassAPI`、递归添加碰撞
- **被调用**：目前作为工具函数保留（旧版 startup 事件用过，现在 pool 物体由 `RigidObjectCfg` 自动带物理）

##### `_find_assets_root(stage)` (L188-216)
- **用途**：在 stage 中查找 `/World/Assets` 根节点（兼容不同场景结构）
- **逻辑**：先直接查 `/World/Assets`，失败则遍历 stage 找含有 `target/distractor/container` 子节点的 `Assets` prim
- **被调用**：目前作为工具函数保留

##### `startup_make_assets_interactive(env, env_ids, mass=0.2)` (L219-259)
- **用途**：startup 事件回调，把 scene.usdc 中自带的旧物体藏到地下
- **流程**：
  1. 获取 stage
  2. 遍历 `/World/envs/env_0`, `/World/envs/env_1`, ... 收集所有 env namespace
  3. 对每个 `{ns}/Assets/target/`、`{ns}/Assets/distractor/`、`{ns}/Assets/container/` 下的子物体
  4. 清除 xform ops，添加 `TranslateOp(0, 0, -100)` 移到地下
- **执行时机**：仿真启动时执行一次（`EventCfg.startup_assets_physics`，mode="startup"）
- **为什么需要**：scene.usdc 里已有旧版 target/distractor 物体，会与新 pool 的 16 个 RigidObject 冲突

#### 场景配置类

##### `SceneCfg(InteractiveSceneCfg)` (L262-320)

定义仿真场景中所有实体：

| 属性 | 类型 | 说明 |
|------|------|------|
| `scene_sublayer_loader` | `AssetBaseCfg` | 调用 `spawn_add_scene_as_sublayer` 加载 scene.usdc（桌子、背景） |
| `sphere_light` | `AssetBaseCfg` | 球形灯光，intensity=5000，位于 (0, -0.6, 0.7) |
| `robot` | `ArticulationCfg` | Franka 机器人（= `Set_Franka`，从 init_franka.py 导入） |
| `external_cam` | `TiledCameraCfg` | 外部固定相机，256x256 RGB，prim_path=`{ENV_REGEX_NS}/external_cam` |
| `wrist_cam` | `TiledCameraCfg` | 腕部相机，挂在 Robotiq 夹爪 base_link 下，256x256 RGB |
| `gripper_target_contact_sensor` | `ContactSensorCfg` | 接触传感器，检测夹爪 base_link 与 pool 物体 `obj_.*` 的接触力 |

##### 16 个 RigidObjectCfg 动态添加 (L323-353)

模块级 for 循环，遍历 `ASSET_UIDS` 和 `ASSET_INFOS`：
- 对每个物体创建 `RigidObjectCfg`：
  - `prim_path = "{ENV_REGEX_NS}/pool/obj_{idx:02d}"`（每个 env 独立副本）
  - `spawn = UsdFileCfg(usd_path, scale, mass_props, rigid_props, collision_props)`
  - `init_state.pos = (0, 0, -5)`（初始藏在地下）
- `setattr(SceneCfg, f"obj_{idx:02d}", cfg)` 动态添加到 SceneCfg
- 循环后 `del` 清理临时变量

#### 动作类

##### `BinaryJointPositionZeroToOneAction(BinaryJointPositionAction)` (L356-371)
- **用途**：自定义二值夹爪动作
- **`process_actions(actions)`**：
  - 输入 action > 0.5 时 → `close_command`（finger_joint = pi/4，合拢）
  - 输入 action <= 0.5 时 → `open_command`（finger_joint = 0，张开）
  - 可选 clip 限制范围

##### `BinaryJointPositionZeroToOneActionCfg` (L374-376)
- 配置类，`class_type = BinaryJointPositionZeroToOneAction`

##### `ActionCfg` (L379-393)
| 属性 | 类型 | 说明 |
|------|------|------|
| `body` | `JointPositionActionCfg` | 7D，控制 panda_joint1~7，直接设目标关节角 |
| `finger_joint` | `BinaryJointPositionZeroToOneActionCfg` | 1D，二值夹爪控制 |

总 action_dim = 8。

#### 观测函数

##### `arm_joint_pos(env, asset_cfg)` (L396-413)
- **输入**：env（ManagerBasedRLEnv）
- **输出**：`[N, 7]` — 7 个手臂关节的当前角度
- **逻辑**：从 `robot.data.joint_pos` 中按名称索引 panda_joint1~7

##### `gripper_pos(env, asset_cfg)` (L416-426)
- **输入**：env
- **输出**：`[N, 1]` — 夹爪开度，归一化到 [0, 1]
- **逻辑**：读取 `finger_joint` 角度，除以 `pi/4`（最大开角）

##### `target_contact(env, sensor_cfg, threshold=0.5)` (L429-459)
- **输入**：env、传感器配置、力阈值
- **输出**：`[N]` — 每个 env 是否有接触（0.0 或 1.0）
- **逻辑**：
  1. 优先用 `force_matrix_w [N, B, M, 3]`（B=sensor bodies, M=filtered targets）
  2. 计算力向量模 → 任意 body x 任意 target 超过阈值 → 1.0
  3. fallback: 用 `net_forces_w [N, B, 3]`（不区分目标）
  4. 最终 fallback: 返回全 0
- **双重用途**：既作为 obs term（传入 obs dict），也作为 reward function

##### `active_target_idx(env)` (L462-473)
- **输入**：env
- **输出**：`[N, 1]` — 每个 env 当前 target 物体在 pool 中的 index（float）
- **逻辑**：读取 `env._current_target_indices`（由 `reset_randomize_objects` 写入）
- **用途**：传到主进程后，`_wrap_obs` 用这个 index 查 JSON 得到 caption，生成 task description

#### 观测配置

##### `ObservationCfg` (L476-508)

内部类 `PolicyCfg(ObsGroup)` 包含 6 个 obs term：

| Term | 函数 | 输出 shape | 附加处理 |
|------|------|-----------|---------|
| `arm_joint_pos` | `arm_joint_pos()` | [N, 7] | 无 |
| `gripper_pos` | `gripper_pos()` | [N, 1] | 高斯噪声 std=0.05，clip(0,1) |
| `external_cam` | `mdp.observations.image` | [N, 256, 256, 3] | 无 normalize |
| `wrist_cam` | `mdp.observations.image` | [N, 256, 256, 3] | 无 normalize |
| `target_contact` | `target_contact()` | [N] | 无 |
| `active_target_idx` | `active_target_idx()` | [N, 1] | 无 |

`__post_init__`：
- `enable_corruption = False`（不加 obs 噪声到整体）
- `concatenate_terms = False`（**关键**：obs 返回 dict 而非拼接 tensor，wrapper 按 key 取值）

#### 事件函数

##### `reset_randomize_objects(env, env_ids, ...)` (L511-572)
- **用途**：每次 reset 时随机化桌面物体
- **参数**：`num_target=1, num_distractor=2, table_x_range, table_y_range, table_z=0.45`
- **流程（对每个需要 reset 的 env_id）**：
  1. 从 16 个物体中 `random.sample` 选 3 个（1 target + 2 distractor）
  2. 对选中的 3 个物体：
     - 在桌面区域随机生成位置 `(x, y, 0.45)`
     - 加上 `env.scene.env_origins[eid]`（env 间的位移偏移）
     - `asset.write_root_pose_to_sim(pose, env_ids=eid_tensor)` 写位姿
     - `asset.write_root_velocity_to_sim(zeros)` 清速度
  3. 对其余 13 个物体：移到 `env_origin + (0, 0, -5)` 藏起来
  4. 存储 `env._current_target_indices[eid] = target_indices[0]`

#### 事件配置

##### `EventCfg` (L575-594)

| 事件 | 函数 | mode | 说明 |
|------|------|------|------|
| `reset_all` | `mdp.reset_scene_to_default` | reset | 所有物体回到 init_state（全部 z=-5） |
| `reset_randomize` | `reset_randomize_objects` | reset | 选 3 个物体放桌面（在 reset_all 之后执行） |
| `startup_assets_physics` | `startup_make_assets_interactive` | startup | 首次启动时藏旧物体（z=-100） |

执行顺序：
- **启动时**：`startup_assets_physics` 执行一次
- **每次 reset**：`reset_all` → `reset_randomize`（按声明顺序）

#### 奖励/终止/其他配置

##### `RewardsCfg` (L602-604)
- `target_contact`：weight=1.0，复用 `target_contact()` 函数，夹爪碰物体得 1 分

##### `TerminationsCfg` (L607-609)
- `time_out`：`mdp.time_out`，episode 超时截断（不算 termination）

##### `CommandsCfg` (L597-599) / `CurriculumCfg` (L612-614)
- 空配置（本任务不需要 command 或 curriculum）

#### 总配置

##### `EnvCfg(ManagerBasedRLEnvCfg)` (L617-641)

| 属性 | 值 | 说明 |
|------|-----|------|
| `scene` | `SceneCfg(num_envs=1, env_spacing=7.0)` | 默认 1 env（被 yaml 覆盖），env 间距 7m |
| `observations` | `ObservationCfg()` | 含 PolicyCfg |
| `actions` | `ActionCfg()` | 7D arm + 1D gripper |
| `rewards` | `RewardsCfg()` | contact reward |
| `terminations` | `TerminationsCfg()` | time_out |
| `events` | `EventCfg()` | startup + reset 事件 |

`__post_init__`：
- `episode_length_s = 18`（秒）
- `decimation = 8`（每 8 个 sim step 执行一次 policy step）
- `sim.dt = 1/120`（物理步长）
- `sim.render_interval = 8`
- `PhysX CCD = True`（连续碰撞检测）
- `gpu_*_capacity = 2^30`（1GB GPU buffer）
- `rerender_on_reset = True`

#### Gym 注册 (L645-650)

```python
gym.register(
    id="DROID-Franka-JointPos-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": EnvCfg},
)
```

---

### 文件 2: `init_franka.py` — Franka 机器人配置（58 行）

**只在子进程中运行**（被 `droid_env_cfg.py` import）。

#### `Set_Franka` (L14-57)

一个 `ArticulationCfg` 实例（不是类，是直接实例化的配置对象）：

| 属性 | 值 | 说明 |
|------|-----|------|
| `prim_path` | `"{ENV_REGEX_NS}/robot"` | 每个 env 独立的机器人 prim |
| `spawn` | `None` | **不 spawn 新的**——机器人已在 scene.usdc 中，IsaacLab 直接接管 |
| `init_state.pos` | `(0, 0, 0)` | 机器人位置（由场景决定） |
| `init_state.joint_pos` | 见下方 | 初始关节角度 |

初始关节角度（Franka 标准 home pose）：
- panda_joint1 = 0, joint2 = -pi/5, joint3 = 0, joint4 = -4pi/5, joint5 = 0, joint6 = 3pi/5, joint7 = 0
- finger_joint = 0（夹爪张开）
- outer/inner knuckle = 0

Actuator 配置：

| Actuator 组 | 关节 | effort | velocity | stiffness | damping |
|-------------|------|--------|----------|-----------|---------|
| `panda_shoulder` | joint1-4 | 87.0 | 2.175 | 400.0 | 80.0 |
| `panda_forearm` | joint5-7 | 12.0 | 2.61 | 400.0 | 80.0 |
| `gripper` | finger_joint | - | 1.0 | None | None |

---

### 文件 3: `objaverse_targets_16.json` — 物体池数据

16 个 Objaverse 物体的元信息，每个条目：

```json
{
  "uid": "19d1a8e4576e4f65877d6427a923fd41",
  "caption": "apple",                    // 用于生成 task description
  "category": "apple",
  "general_category": "food",
  "usd_path": "/path/to/instance.usd",   // USD 模型文件路径
  "scale_min": "0.09",                   // spawn 时 scale 范围
  "scale_max": "0.11",
  "mass_min": "0.18",                    // 物理质量范围
  "mass_max": "0.22",
  "height": "0.06",                      // 物体尺寸参考
  "can_grasp": "True"
}
```

包含的物体：apple, banana, lemon, orange, pear, avocado, croissant, donut, mug, beverage can, plastic bottle, soccer ball, basketball, camera, headphones, kettle。

---

### 文件 4: `droid_task.py` — RLinf Wrapper（125 行）

**在主进程中运行**。连接 RLinf 训练框架与 IsaacLab 仿真子进程。

#### 模块级变量 (L11-14)

```python
_POOL_JSON = Path(__file__).parent / "objaverse_targets_16.json"
_POOL = json.load(open(_POOL_JSON))
_ASSET_INFOS = list(_POOL.values())
```

**直接读 JSON**，不 import `droid_env_cfg`（因为主进程没有 `pxr`/`isaaclab`）。
用于 `_wrap_obs` 中根据 `active_target_idx` 查 caption。

#### `IsaaclabDroidEnv(IsaaclabBaseEnv)` (L17-124)

##### `__init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info)` (L25-39)
- 直接调用 `super().__init__()`
- 基类 `IsaaclabBaseEnv.__init__` 会调用 `self._make_env_function()` → `self._init_isaaclab_env()` → 启动子进程

##### `_make_env_function(self)` (L41-83)
- **返回**：一个闭包 `make_env_isaaclab`，由子进程执行
- **闭包内部流程**：
  1. 移除 `DISPLAY` 环境变量（headless 模式）
  2. 设置 Vulkan ICD 路径
  3. `AppLauncher(headless=True, enable_cameras=True, device="cuda:0")` 启动 IsaacLab
  4. `import droid_env_cfg`（触发 gym.register + 模块级 JSON 加载 + SceneCfg 构建）
  5. `from droid_env_cfg import EnvCfg, SCENE_USDC_PATH`
  6. 创建 `EnvCfg()`，用 yaml 配置覆盖：
     - `scene.num_envs` ← `self.cfg.init_params.num_envs`
     - `scene.external_cam.height/width` ← yaml `table_cam`
     - `scene.wrist_cam.height/width` ← yaml `wrist_cam`
     - `scene.scene_sublayer_loader.spawn.usd_path` ← yaml `scene_usdc_path`（可选覆盖）
  7. `gym.make("DROID-Franka-JointPos-Visuomotor-v0", cfg=EnvCfg).unwrapped`
  8. 返回 `(env, sim_app)`

##### `_wrap_obs(self, obs)` (L85-124)
- **输入**：IsaacLab 返回的 obs dict `{"policy": {"arm_joint_pos": ..., "wrist_cam": ..., ...}}`
- **输出**：Pi0.5 模型需要的 4-key dict
- **流程**：
  1. 读取 `obs["policy"]["active_target_idx"]` → `[N, 1]` float tensor
  2. 对每个 env，用 `int(idx)` 查 `_ASSET_INFOS[idx]["caption"]` → 生成 `"pick up the {caption}"`
  3. fallback：如果 `active_target_idx` 不存在，用 yaml 中的静态 `task_description`
  4. 拼接 states：`arm_joint_pos [N,7]` + `gripper_pos [N,1]` → `[N, 8]`
  5. 返回：

| Key | 值 | 说明 |
|-----|-----|------|
| `main_images` | `obs["policy"]["external_cam"]` | [N, 256, 256, 3] |
| `wrist_images` | `obs["policy"]["wrist_cam"]` | [N, 256, 256, 3] |
| `states` | `cat(arm_joint_pos, gripper_pos)` | [N, 8] |
| `task_descriptions` | `["pick up the apple", ...]` | 长度 N 的字符串列表 |

---

### 文件 5: `__init__.py` — 环境注册表（26 行）

```python
from .tasks.droid_task import IsaaclabDroidEnv

REGISTER_ISAACLAB_ENVS = {
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0": IsaaclabStackCubeEnv,
    "Isaac-Stack-Cube-Franka-JointPos-Visuomotor-Rewarded-v0": IsaaclabStackCubeJointPosEnv,
    "DROID-Franka-JointPos-Visuomotor-v0": IsaaclabDroidEnv,   # ← 新增
}
```

`env_worker` 根据 yaml 中 `init_params.id` 查这个字典来实例化对应的 env 类。

---

### 文件 6: `isaaclab_env.py` — 基类（已有，未修改，265 行）

`IsaaclabBaseEnv(gym.Env)` 提供：

| 方法 | 说明 |
|------|------|
| `__init__()` | 解析 cfg，设置 num_envs/seed/video_cfg，调用 `_init_isaaclab_env()` 启动子进程 |
| `_init_isaaclab_env()` | 调用子类的 `_make_env_function()` 获取闭包 → `SubProcIsaacLabEnv(fn)` 启动子进程 → `env.reset()` |
| `reset(seed, env_ids)` | 发送 reset 命令到子进程 → 收到 obs → `_wrap_obs()` → 重置 metrics |
| `step(actions)` | 发送 action 到子进程 → 收到 (obs, reward, term, trunc, info) → `_wrap_obs()` → 累加 elapsed_steps → 记录 metrics |
| `chunk_step(chunk_actions)` | 输入 `[N, chunk_size, action_dim]` → 循环 chunk_size 次调用 `step()` → 聚合 rewards/terminations |
| `_handle_auto_reset(dones)` | done 的 env 自动 reset，保存 final_observation |
| `_record_metrics()` | 累计 returns，标记 success_once |
| `_calc_step_reward()` | 支持 `use_rel_reward` 模式（reward diff） |
| `elapsed_steps` | property，返回每个 env 已走的步数 |

子类（`IsaaclabDroidEnv`）只需实现 `_make_env_function()` 和 `_wrap_obs()`。

---

### 文件 7: `venv.py` — 子进程通信（已有，未修改，119 行）

#### `_torch_worker(child_remote, parent_remote, env_fn_wrapper, action_queue, obs_queue, reset_idx_queue)` (L23-71)
- **运行在子进程中**
- 调用 `env_fn()` 创建 IsaacLab env + sim_app
- 进入主循环，监听 pipe 命令：
  - `"reset"` → 从 `reset_idx_queue` 取 `(env_ids, seed)` → `isaac_env.reset()` → 结果放入 `obs_queue`
  - `"step"` → 从 `action_queue` 取 action → `isaac_env.step(action)` → 结果放入 `obs_queue`
  - `"close"` → 关闭 env 和 sim_app
  - `"device"` → 通过 pipe 返回 device 字符串

#### `SubProcIsaacLabEnv` (L74-118)
- **运行在主进程中**
- `__init__(env_fn)` → `mp.Process(target=_torch_worker, start_method="spawn")` 启动子进程
- `reset(seed, env_ids)` → pipe 发 "reset" + queue 发参数 → queue 收结果
- `step(action)` → pipe 发 "step" + queue 发 action → queue 收 5-tuple
- `close()` → pipe 发 "close" → join + terminate
- `device()` → pipe 发 "device" → pipe 收返回值

通信机制：`Pipe`（命令信号）+ `Queue`（大数据，支持 CUDA tensor 序列化）。

---

### 文件 8: `droid_ppo_pi05.yaml` — 训练总配置（190 行）

通过 Hydra `defaults` 组合多个子配置：

```yaml
defaults:
  - env/droid_task@env.train       # 加载 droid_task.yaml 作为 env.train
  - env/droid_task@env.eval        # 同一文件作为 env.eval
  - model/pi0_5@actor.model        # Pi0.5 模型配置
  - training_backend/fsdp@actor.fsdp_config
```

然后在 `env:` 块中覆盖 train/eval 的具体参数（num_envs、max_steps 等）。

---

### 文件 9: `env/droid_task.yaml` — 环境默认配置（31 行）

提供 `IsaaclabDroidEnv` 构造所需的全部默认参数：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `env_type` | `isaaclab` | 告诉 env_worker 从 isaaclab 注册表查找 |
| `auto_reset` | `False` | 不自动 reset（由 RLinf 上层控制） |
| `use_rel_reward` | `True` | 使用 reward diff |
| `init_params.id` | `"DROID-Franka-JointPos-Visuomotor-v0"` | gym 注册 ID |
| `init_params.task_description` | `"pick up the target object"` | 静态 fallback |
| `init_params.scene_usdc_path` | 绝对路径 | 场景文件位置 |
| `init_params.table_cam / wrist_cam` | 256x256 | 相机分辨率 |

---

### 文件 10: `run_droid_ppo_pi05.sh` — 启动脚本（56 行）

```bash
# 1. 激活虚拟环境
source venv_isaaclab/bin/activate

# 2. 加载 Isaac Sim 环境变量
source isaac_sim/setup_conda_env.sh

# 3. 设置 PYTHONPATH（venv site-packages + RLinf 根目录）
export PYTHONPATH="${VENV_SITE}:${RLINF_DIR}:$PYTHONPATH"

# 4. 设置渲染（Vulkan）、通信（NCCL/GLOO）、Ray
export VK_ICD_FILENAMES=...
export NCCL_SOCKET_FAMILY=AF_INET
export RAY_ADDRESS="local"

# 5. 创建日志目录并启动训练
python train_embodied_agent.py \
  --config-path .../config/ \
  --config-name droid_ppo_pi05 \
  runner.logger.log_path="${LOG_DIR}"
```
