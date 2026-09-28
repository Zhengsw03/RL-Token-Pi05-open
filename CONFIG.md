# Configuration reference

**English** | [中文](CONFIG.zh-CN.md)

<details>
<summary>Contents</summary>

- [How a config file is read](#how-a-config-file-is-read)
- [Paths and checkpoints](#paths-and-checkpoints)
- [Task, arms and serial ports](#task-arms-and-serial-ports)
- [Cameras](#cameras)
- [Run mode and execution cadence](#run-mode-and-execution-cadence)
- [Critical phase and classifier](#critical-phase-and-classifier)
- [Human intervention](#human-intervention)
- [Rollout recording](#rollout-recording)
- [Live display](#live-display)
- [Learning options](#learning-options)
- [Minimal configuration](#minimal-configuration)

</details>

Every option of `scripts/infer_rlt_stage2_pi05.py`, with its default value and what it changes in
a run. The template is `configs/infer_pi05.json`; a filled-in copy is passed on the command line:

```bash
python scripts/infer_rlt_stage2_pi05.py --config configs/infer_pi05.json
```

## How a config file is read

- A key is the flag name without the leading `--`: `"control_fps"` sets `--control_fps`. The file
  is JSON and may contain `//` comments.
- Keys that are absent keep their default; keys that are unknown abort the startup with the list
  of accepted names, so a typo is never ignored silently.
- Explicit command-line flags override the file, which allows a one-off experiment without editing
  it.
- Two styles of boolean exist. Single flags (`auto_critical`, `dry_run`, `save_debug_obs`,
  `inference_only`) can only be turned on, from the file or with the flag; there is no
  `--auto_critical false`, so a key like `"dry_run": false` is simply the default. Paired flags
  (`record_dataset`, `rtc_enabled`, `leader_mirror`) accept `true` or `false` in the file and
  `--no-...` on the command line.
- Four keys hold JSON text rather than plain values: `cameras`, `camera_map`,
  `max_relative_target` and `actor_max_relative_target`.

Only `pi05_path` and `rlt_checkpoint` are mandatory. Everything else has a working default, and
the values that follow the defaults can be deleted from a config without changing a run.

## Paths and checkpoints

| Key | Default | Meaning |
| --- | --- | --- |
| `pi05_path` | required | Fine-tuned π0.5 directory. Must be the same VLA that produced the embeddings, the RLT checkpoint and the actor. |
| `rlt_checkpoint` | required | RL Token encoder-decoder from Step 3, either `best_checkpoint.pt` or an exported `pretrained_model/` directory. |
| `resume_from` | none | Stage 2 actor to deploy, normally `final_checkpoint.pt`. Accepted as `actor_checkpoint` too. |
| `tokenizer_path` | empty | Local PaliGemma tokenizer directory; needed when the π0.5 directory does not carry one, which is the usual case offline. |
| `stats_path` | none | Override for the normalization statistics. Empty means the values stored in the π0.5 checkpoint are used. |
| `output_dir` | `checkpoints/rlt_stage2_pi05` | Directory for the run log, the critical-phase frames and the debug dumps. Point it at the experiment being run. |
| `critical_classifier` | none | Classifier checkpoint from Step 4, required when `auto_critical` is on. |
| `device` | `cuda` | Torch device for the VLA, the RLT encoder and the actor. |
| `seed` | `42` | Seed for sampling, exploration noise and the optional reset passes. |

## Task, arms and serial ports

| Key | Default | Meaning |
| --- | --- | --- |
| `task` | `pick up the red cube and place it in the box` | Task text handed to the VLA; it must match the text used during fine-tuning. |
| `follower_port` | empty | Follower serial port, required unless `dry_run` is on. |
| `leader_port` | empty | Leader serial port, required unless `dry_run` is on. |
| `follower_id` | `so101_follower` | Calibration id of the follower in the LeRobot calibration store. |
| `leader_id` | `so101_leader` | Calibration id of the leader. |
| `max_relative_target` | none | Per-step per-joint limit in degrees applied to every action source. A scalar or a JSON object such as `{"wrist_roll": 4.0}`; omitted disables clipping. |
| `observation_retries` | `3` | Retries for a transient camera or bus fault before the run aborts. No command is sent while retrying. |
| `observation_retry_delay_s` | `0.25` | Delay between those retries. |
| `reset_between_episodes` | off | Run the closed-loop reset that returns the follower to the startup pose between episodes. |
| `reset_steps` | `90` | Length of that reset pass. |
| `reset_dt` | `0.08` | Interval between reset commands, in seconds. |
| `reset_tolerance_deg` | `5.0` | Largest accepted residual reset error, in degrees. |
| `reset_max_passes` | `2` | Number of correction passes allowed to reach that tolerance. |
| `manual_positioning` | off | Let the operator position the follower by hand with the `p` key instead of the automatic reset. Positioning is never recorded. |

## Cameras

| Key | Default | Meaning |
| --- | --- | --- |
| `cameras` | template with two OpenCV cameras | JSON object of physical camera names to camera settings: `type`, `index_or_path`, `width`, `height`, `fps`, and optionally `fourcc` and `warmup_s`. The index has to match the physical camera, which is worth checking after any replug. |
| `camera_map` | empty | JSON object mapping each physical camera name to the checkpoint image feature it feeds, for example `{"top": "observation.images.top"}`. Required: it has to cover the features the checkpoint declares. |
| `camera_names` | `["top", "wrist"]` | Physical camera names to open. Must match the keys of `cameras`. |
| `pad_missing_cameras` | off | Fill unmapped image features with zero tensors so a run can start without a camera. Test only, it degrades the policy. |
| `camera_preflight_retries` | `6` | Fresh-frame attempts per camera at startup. |
| `camera_preflight_delay_s` | `0.5` | Delay between those attempts. |

## Run mode and execution cadence

| Key | Default | Meaning |
| --- | --- | --- |
| `dry_run` | off | Validate paths, weights, cameras and pairing, then stop without opening the serial ports. Use it before the first run on hardware. |
| `vla_only` | off | Frozen VLA only: the actor is never executed. Useful to check the camera images and the motion directions. |
| `inference_only` | off | Deployment without learning: the actor runs from the supplied checkpoint, no updates, no replay, no reward prompts. |
| `inference_score` | on | In `inference_only` mode, ask for y/n after each episode to accumulate a success rate. The answer is never fed back into learning. |
| `control_fps` | `10.0` | Rate at which actions are sent to the follower and frames are recorded. It should match the rate the teleoperation data was collected at, which is 30 in the provided configs. |
| `steps_per_episode` | `300` | Step limit for one episode; the episode ends early if the policy finishes first. |
| `max_episodes` | `200` | Number of episodes before the script stops. |
| `actor_execution_steps` | `0` | Steps the actor executes before the next replan. `0` uses the full RL chunk, that is 10 steps. |
| `actor_execution_scale` | `1.0` | Scale applied to actor actions before execution. `0.5` halves the speed and spreads the task over more steps. |
| `actor_execute_mean` | off | Execute the actor's mean action without sampling, which removes the sampling jitter. |
| `actor_noise_std` | `0.0` | Exploration noise added on top of that mean, in normalized action space. |
| `actor_max_relative_target` | none | Per-step per-joint limit in degrees applied only to actor actions, relative to the last commanded pose. VLA and human actions are not clamped. |
| `actor_critical_delay_steps` | `15` | Consecutive critical-phase control steps required before the actor may take over. |
| `rtc_enabled` | on | Pi0.5 real-time chunking: the previous chunk guides the new one, which removes the pause between replans. |
| `rtc_execution_horizon` | `10` | Actions executed between two Pi0.5 replans. It has to match the RL horizon and `n_action_steps_rl`. |
| `rtc_prefix_attention_schedule` | `EXP` | Attention schedule of the RTC prefix, an implementation detail of the guidance. |
| `rtc_max_guidance_weight` | `5.0` | Upper bound of the RTC guidance weight. |
| `replan_every_window` | on | Replan a fresh VLA chunk at every window so the reference always comes from the current observation. |

## Critical phase and classifier

| Key | Default | Meaning |
| --- | --- | --- |
| `auto_critical` | off | Let the classifier toggle the critical phase instead of the `c` key. Requires `critical_classifier`; a manual `c` press still overrides it for a cooldown. |
| `auto_critical_threshold_on` | `0.5` | Smoothed probability above which the phase turns on. Lower means earlier detection and occasional false starts. |
| `auto_critical_threshold_off` | `0.35` | Smoothed probability below which the phase turns off. |
| `auto_critical_smooth_steps` | `5` | Moving-average window in chunks. Smaller reacts faster and jitters more. |
| `auto_critical_min_on_chunks` | `2` | Minimum chunks the phase stays on once triggered, which debounces dips in the probability. |
| `auto_critical_start_delay_steps` | `60` | Steps at the beginning of an episode during which the phase cannot turn on, so the approach phase is never mistaken for the critical one. |
| `auto_critical_once` | on | One critical segment per episode: after an automatic off, the detector does not turn it on again. A manual `c` still works. |
| `auto_critical_override_cooldown` | `3.0` | Seconds after a manual `c` press during which the detector is muted. |
| `save_critical_frames` | on | Save camera frames at every critical-phase transition for review. |
| `critical_frames_dir` | none | Where those frames go. Empty means `<output_dir>/critical_frames`. |
| `save_debug_obs` | off | Save the per-episode debug observations under `<output_dir>/debug_obs`: camera images at each replan plus a per-step state, action and source CSV. |

## Human intervention

| Key | Default | Meaning |
| --- | --- | --- |
| `leader_mirror` | off | Mirror the follower pose onto the leader continuously, so both arms move together. Off means the leader stays still until it is taken over. |
| `leader_mirror_torque_limit` | `0.5` | Holding torque of that mirror, as a fraction of the maximum. Lower yields more easily when a human grabs the leader. |
| `leader_hil_align_duration_s` | `5.0` | Length of the smooth approach that aligns the powered leader to the follower before a takeover. |
| `leader_hil_align_fps` | `50.0` | Command rate of that alignment. |
| `leader_hil_align_tolerance` | `10.0` | Required six-joint alignment error after the approach, in degrees. |
| `intervention_threshold` | `1.0` | Leader movement per sample, in degrees, that requests an automatic takeover. The operator can also request the takeover with `h`. |
| `intervention_trigger_frames` | `2` | Consecutive moving leader samples needed to latch that request. |
| `intervention_release_frames` | `100` | Leader samples that must stay still after the last movement before the policy resumes. `s` resumes immediately instead. |
| `leader_hil_min_dwell_s` | `2.0` | Minimum time in direct follow and quiet period after motion before the policy resumes. |
| `intervention_reward_bonus` | `0.0` | Reward added for episodes that needed an intervention. `0` keeps the terminal reward as judged. |

## Rollout recording

| Key | Default | Meaning |
| --- | --- | --- |
| `record_dataset` | off | Record the rollouts as a standard LeRobot dataset with `observation.images.*`, `observation.state` and `action`, episode index starting at 0. |
| `record_repo_id` | `local/so101_deploy_rollouts` | Dataset identifier. It is only a name, nothing is uploaded. |
| `record_root` | empty | Dataset directory. Empty means `<repo>/outputs/deploy_recordings/<last segment of record_repo_id>`. |
| `record_task` | empty | Task text stored in the dataset; empty reuses `task`. |
| `record_videos` | on | Encode the images as video, matching teleoperation data. Turning it off stores images instead. |
| `record_hil_steps` | on | Also record the steps of a human takeover. With it off, an episode containing an intervention is discarded because its trajectory would have a gap. |
| `record_streaming_encoding` | on | Encode video in a background thread so the control period is not slowed down. |
| `record_encoder_threads` | `2` | Number of those encoding threads. |

## Live display

| Key | Default | Meaning |
| --- | --- | --- |
| `display_data` | on | Show the live camera images and the executed actions in Rerun. |
| `display_compressed_images` | off | Compress the images before sending them, which lowers the bandwidth at some quality cost. |
| `display_fps` | `10.0` | Maximum logging rate of the Rerun viewer. |
| `display_session_name` | `rlt_deploy` | Rerun session name, so several runs can be told apart. |

## Learning options

These belong to the online RL loop and have no effect in a pure deployment run; they are listed
here because the deployment script accepts the same names as the trainer.

| Key | Default | Meaning |
| --- | --- | --- |
| `actor_lr` | `3e-4` | Actor learning rate. |
| `critic_lr` | `3e-4` | Critic learning rate. |
| `discount` | `0.99` | Discount factor of the returns. |
| `target_tau` | `0.005` | Soft update rate of the target critic. |
| `online_bc_weight` | `10.0` | Weight of the behaviour-cloning regularization applied after the learner warmup. |
| `delta_weight` | `0.0` | Weight of the smoothness loss, which penalizes a per-step delta that differs from the reference delta. |
| `policy_fixed_std` | `0.05` | Fixed policy standard deviation used by the actor loss. |
| `reference_dropout` | `0.5` | Probability of zeroing the reference chunk while training the actor. |
| `actor_hidden_dim` | `256` | Hidden width of the actor. |
| `actor_num_layers` | `2` | Number of actor layers, 2 or 3. |
| `critic_hidden_dim` | `256` | Hidden width of the critic. |
| `critic_num_layers` | `2` | Number of critic layers, 2 or 3. |
| `actor_residual_scale` | `0.0` | Deprecated compatibility flag. The paper-aligned configuration requires 0. |

## Minimal configuration

Everything except these keys can stay out of a config file:

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

`python scripts/infer_rlt_stage2_pi05.py --help` prints the same options as flags, which is the
quickest way to check a default or the accepted spelling of a key.
