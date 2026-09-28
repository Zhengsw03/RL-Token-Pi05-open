# 配置项说明

[English](CONFIG.md) | **中文**

<details>
<summary>目录</summary>

- [配置文件怎么读](#配置文件怎么读)
- [路径与 checkpoint](#路径与-checkpoint)
- [任务、机械臂与串口](#任务机械臂与串口)
- [相机](#相机)
- [运行模式与执行节奏](#运行模式与执行节奏)
- [关键阶段与分类器](#关键阶段与分类器)
- [人工介入](#人工介入)
- [录制](#录制)
- [实时显示](#实时显示)
- [学习相关](#学习相关)
- [最小配置](#最小配置)

</details>

本文逐项说明 `scripts/infer_rlt_stage2_pi05.py` 的全部参数、默认值以及它改变什么。模板为
`configs/infer_pi05.json`,填好后在命令行传入:

```bash
python scripts/infer_rlt_stage2_pi05.py --config configs/infer_pi05.json
```

## 配置文件怎么读

- 键名就是去掉前缀 `--` 的参数名:`"control_fps"` 对应 `--control_fps`。文件是 JSON,允许写 `//` 注释。
- 没写的键保持默认值;写了脚本不认识的键会在启动时直接报错并列出可用名称,所以拼错不会被忽略。
- 命令行显式给出的参数会覆盖配置文件,便于临时做一次实验而不改动文件。
- 布尔参数有两类。单开关(`auto_critical`、`dry_run`、`save_debug_obs`、`inference_only`)只能打开,
  在文件里写 `true` 或加命令行 flag 均可;**没有** `--auto_critical false` 这种写法,所以
  `"dry_run": false` 只是把默认值又写了一遍。成对开关(`record_dataset`、`rtc_enabled`、
  `leader_mirror`)在文件里可写 `true`/`false`,命令行可用 `--no-...` 关闭。
- 有四个键存放的是 JSON 文本而不是普通值:`cameras`、`camera_map`、`max_relative_target`、
  `actor_max_relative_target`。

只有 `pi05_path` 与 `rlt_checkpoint` 是必填;其余都有可用默认值,与默认相同的键可以直接删掉,
不影响运行。

## 路径与 checkpoint

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `pi05_path` | 必填 | 微调后的 π0.5 目录。必须是产出 embedding、RLT checkpoint 与 actor 的同一个 VLA。 |
| `rlt_checkpoint` | 必填 | 步骤 3 得到的 RL Token 编码器,可以是 `best_checkpoint.pt` 或导出的 `pretrained_model/` 目录。 |
| `resume_from` | 无 | 要部署的第 2 阶段 actor,通常是 `final_checkpoint.pt`。也接受 `actor_checkpoint` 这个别名。 |
| `tokenizer_path` | 空 | 本地 PaliGemma tokenizer 目录;π0.5 目录里没带 tokenizer 时必须给(离线机器属常见情况)。 |
| `stats_path` | 无 | 覆盖归一化统计;留空表示用 π0.5 checkpoint 内保存的值。 |
| `output_dir` | `checkpoints/rlt_stage2_pi05` | 运行日志、关键帧与调试落盘的目录,建议指向本次实验的目录。 |
| `critical_classifier` | 无 | 步骤 4 得到的分类器,开启 `auto_critical` 时必填。 |
| `device` | `cuda` | VLA、RLT 编码器与 actor 使用的 torch 设备。 |
| `seed` | `42` | 采样、探索噪声以及可选复位过程的随机种子。 |

## 任务、机械臂与串口

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `task` | `pick up the red cube and place it in the box` | 交给 VLA 的任务文本,必须与微调时使用的一致。 |
| `follower_port` | 空 | 从臂串口;非 dry_run 时必填。 |
| `leader_port` | 空 | 主臂串口;非 dry_run 时必填。 |
| `follower_id` | `so101_follower` | 从臂在 LeRobot 标定库中的标定 id。 |
| `leader_id` | `so101_leader` | 主臂的标定 id。 |
| `max_relative_target` | 无 | 对所有动作来源生效的单步单关节角度上限(度)。可以是标量,也可以是 `{"wrist_roll": 4.0}` 这类 JSON;留空表示不做限幅。 |
| `observation_retries` | `3` | 相机/总线瞬时故障时的重试次数,重试期间不会下发任何指令。 |
| `observation_retry_delay_s` | `0.25` | 重试间隔(秒)。 |
| `reset_between_episodes` | 关 | 回合之间执行闭环复位,把从臂送回启动位姿。 |
| `reset_steps` | `90` | 复位过程长度。 |
| `reset_dt` | `0.08` | 复位指令间隔(秒)。 |
| `reset_tolerance_deg` | `5.0` | 允许的最大复位残差(度)。 |
| `reset_max_passes` | `2` | 为达到该残差允许的修正轮数。 |
| `manual_positioning` | 关 | 改用人工摆位:按 `p` 用主臂直接拖动从臂到起始位姿,摆位过程不写入任何数据。 |

## 相机

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `cameras` | 两台 OpenCV 相机的示例值 | 物理相机名到相机设置的 JSON:`type`、`index_or_path`、`width`、`height`、`fps`,可选 `fourcc`、`warmup_s`。设备号必须与物理相机对应,插拔后建议重新确认。 |
| `camera_map` | 空 | 物理相机名到 checkpoint 图像特征名的 JSON 映射,例如 `{"top": "observation.images.top"}`。必填,且必须覆盖 checkpoint 声明的全部图像特征。 |
| `camera_names` | `["top", "wrist"]` | 要打开的物理相机名,必须与 `cameras` 的键一致。 |
| `pad_missing_cameras` | 关 | 用全零图像补齐未映射的特征,使缺相机时也能启动。仅供测试,会明显降低策略表现。 |
| `camera_preflight_retries` | `6` | 启动时为每台相机尝试取新帧的次数。 |
| `camera_preflight_delay_s` | `0.5` | 上述尝试之间的间隔(秒)。 |

## 运行模式与执行节奏

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `dry_run` | 关 | 只校验路径、权重、相机与配套关系,不打开串口。首次上真机前先跑它。 |
| `vla_only` | 关 | 只跑冻结的 VLA,不执行 actor。用于确认相机画面与动作方向。 |
| `inference_only` | 关 | 纯部署模式:actor 从给定 checkpoint 运行,不更新权重、不记录回放、不弹奖励提示。 |
| `inference_score` | 开 | 在 `inference_only` 下每回合结束后问一次 y/n,用于累计成功率;答案不会进入学习。 |
| `control_fps` | `10.0` | 动作下发与帧记录的频率,应与采集遥操数据时的频率一致;随附配置里用的是 30。 |
| `steps_per_episode` | `300` | 单回合步数上限,策略提前完成则提前结束。 |
| `max_episodes` | `200` | 运行多少回合后停止。 |
| `actor_execution_steps` | `0` | actor 执行多少步后重新规划;`0` 表示用满整个 RL 块(10 步)。 |
| `actor_execution_scale` | `1.0` | actor 动作执行前的缩放系数,`0.5` 相当于半速,同一任务会占用更多控制步。 |
| `actor_execute_mean` | 关 | 执行 actor 的均值动作而不采样,消除采样抖动。 |
| `actor_noise_std` | `0.0` | 在上述均值上叠加的探索噪声,归一化动作空间。 |
| `actor_max_relative_target` | 无 | 只对 actor 动作生效的单步单关节角度上限(度),相对上一次下发位姿;VLA 与人工动作不受限。 |
| `actor_critical_delay_steps` | `15` | 关键阶段持续多少控制步之后 actor 才允许接管。 |
| `rtc_enabled` | 开 | π0.5 实时分块:用上一段动作引导新一段,消除重新规划之间的停顿。 |
| `rtc_execution_horizon` | `10` | 两次 π0.5 规划之间执行的动作数,必须与 RL 视野和 `n_action_steps_rl` 一致。 |
| `rtc_prefix_attention_schedule` | `EXP` | RTC 前缀的注意力调度,属于引导实现的内部细节。 |
| `rtc_max_guidance_weight` | `5.0` | RTC 引导权重的上限。 |
| `replan_every_window` | 开 | 每一个窗口都重新规划 VLA,使参考始终来自当前观测。 |

## 关键阶段与分类器

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `auto_critical` | 关 | 由分类器自动切换关键阶段,替代手动按 `c`;需要 `critical_classifier`。手动按 `c` 仍会在冷却时间内优先。 |
| `auto_critical_threshold_on` | `0.5` | 平滑后概率高于该值时进入关键阶段;越低越早,也更容易误触发。 |
| `auto_critical_threshold_off` | `0.35` | 平滑后概率低于该值时退出关键阶段。 |
| `auto_critical_smooth_steps` | `5` | 概率的滑动平均窗口(以块为单位);越小响应越快、抖动越大。 |
| `auto_critical_min_on_chunks` | `2` | 触发后至少保持的块数,避免概率回落造成开关抖动。 |
| `auto_critical_start_delay_steps` | `60` | 回合开头多少步内不允许自动进入,避免把接近阶段误判为关键阶段。 |
| `auto_critical_once` | 开 | 每回合只有一个关键段:自动退出后不会在同一回合内再次自动进入;手动 `c` 不受限。 |
| `auto_critical_override_cooldown` | `3.0` | 手动按 `c` 之后多少秒内屏蔽自动判定。 |
| `save_critical_frames` | 开 | 在关键阶段每次切换时保存相机帧,便于事后检查判定时刻。 |
| `critical_frames_dir` | 无 | 上述关键帧目录;留空表示 `<output_dir>/critical_frames`。 |
| `save_debug_obs` | 关 | 把每回合调试观测写到 `<output_dir>/debug_obs`:每次重新规划时的相机图,以及逐步的 state/action/source CSV。 |

## 人工介入

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `leader_mirror` | 关 | 让主臂持续跟随从臂,两条臂一起运动。关闭时主臂在被接管前保持不动。 |
| `leader_mirror_torque_limit` | `0.5` | 上述跟随的保持力矩(占最大值的比例);越低越容易被手扳动。 |
| `leader_hil_align_duration_s` | `5.0` | 接管前把上电的主臂平滑对齐到从臂所用的时间。 |
| `leader_hil_align_fps` | `50.0` | 对齐指令频率。 |
| `leader_hil_align_tolerance` | `10.0` | 对齐结束后允许的六关节误差(度)。 |
| `intervention_threshold` | `1.0` | 主臂单个采样的位移达到多少度即请求自动接管;操作者也可以直接按 `h` 请求接管。 |
| `intervention_trigger_frames` | `2` | 锁存该请求所需的连续运动采样数。 |
| `intervention_release_frames` | `100` | 最后一次运动之后主臂需保持静止多少个采样才交还策略;按 `s` 可立即交还。 |
| `leader_hil_min_dwell_s` | `2.0` | 最短直接跟随时间,以及运动结束后的静默保护时间。 |
| `intervention_reward_bonus` | `0.0` | 对发生介入的回合额外给的奖励;`0` 表示完全按结束时的判定给奖励。 |

## 录制

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `record_dataset` | 关 | 把 rollout 录成标准 LeRobot 数据集(`observation.images.*`、`observation.state`、`action`),episode 索引从 0 开始。 |
| `record_repo_id` | `local/so101_deploy_rollouts` | 数据集标识,只是名字,不会上传到 Hub。 |
| `record_root` | 空 | 数据集目录;留空表示 `<repo>/outputs/deploy_recordings/<record_repo_id 的最后一段>`。 |
| `record_task` | 空 | 写入数据集的任务文本;留空复用 `task`。 |
| `record_videos` | 开 | 把图像编码成视频,与遥操数据一致;关闭后改为存图像。 |
| `record_hil_steps` | 开 | 同时记录人工接管的步骤;关闭时含介入的回合会被丢弃,因为轨迹会出现缺口。 |
| `record_streaming_encoding` | 开 | 在后台线程编码视频,避免拖慢控制周期。 |
| `record_encoder_threads` | `2` | 后台编码线程数。 |

## 实时显示

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `display_data` | 开 | 在 Rerun 里显示实时相机画面与已执行动作。 |
| `display_compressed_images` | 关 | 发送前压缩图像,降低带宽但损失画质。 |
| `display_fps` | `10.0` | Rerun 的最大记录频率。 |
| `display_session_name` | `rlt_deploy` | Rerun 会话名,便于区分多次运行。 |

## 学习相关

这些属于在线 RL 训练循环,纯部署运行时不生效;列在这里是因为部署脚本接受与训练脚本同名的参数。

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `actor_lr` | `3e-4` | Actor 学习率。 |
| `critic_lr` | `3e-4` | Critic 学习率。 |
| `discount` | `0.99` | 回报折扣因子。 |
| `target_tau` | `0.005` | 目标 critic 的软更新系数。 |
| `online_bc_weight` | `10.0` | 学习器预热之后行为克隆正则项的权重。 |
| `delta_weight` | `0.0` | 平滑损失的权重,惩罚单步增量与参考增量之间的差异。 |
| `policy_fixed_std` | `0.05` | actor 损失中使用的固定策略标准差。 |
| `reference_dropout` | `0.5` | 训练 actor 时把参考块置零的概率。 |
| `actor_hidden_dim` | `256` | Actor 隐藏层宽度。 |
| `actor_num_layers` | `2` | Actor 层数,取 2 或 3。 |
| `critic_hidden_dim` | `256` | Critic 隐藏层宽度。 |
| `critic_num_layers` | `2` | Critic 层数,取 2 或 3。 |
| `actor_residual_scale` | `0.0` | 兼容性遗留开关,论文对齐配置要求为 0。 |

## 最小配置

除下列键之外,其余键都可以不写进配置文件:

```json
{
  "pi05_path": "/path/to/pi05/checkpoints/030000/pretrained_model",
  "rlt_checkpoint": "/path/to/rlt_stage1/best_checkpoint.pt",
  "resume_from": "/path/to/stage2/final_checkpoint.pt",
  "tokenizer_path": "/path/to/paligemma_tokenizer",
  "task": "Insert the bolt into the hole",
  "follower_port": "/dev/ttyACM0",
  "leader_port": "/dev/ttyACM1",
  "cameras": {
    "top":   {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"},
    "wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}
  },
  "camera_map": {"top": "observation.images.top", "wrist": "observation.images.wrist"},
  "output_dir": "outputs/deploy",
  "control_fps": 30.0,
  "actor_execution_steps": 10,
  "auto_critical": true,
  "critical_classifier": "/path/to/critical_classifier.pt",
  "record_dataset": true,
  "record_repo_id": "local/deploy_rollouts"
}
```

`python scripts/infer_rlt_stage2_pi05.py --help` 会把同一批参数按命令行形式打印出来,是查默认值或
确认键名的最快方式。
