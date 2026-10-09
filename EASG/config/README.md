# Physics and Motion Configuration

To create a custom configuration, copy `default.json` to a new JSON file in this directory,
edit its values, and select it by filename. Run these commands from the project root:

```bash
./python.sh script/run.py --config default.json --task pick_place --headless
./python.sh script/replay.py --config default.json --resume-from output/previous_run --replay-only --headless
./python.sh script/construct_scene/init_scene.py --config default.json
./python.sh script/control_robot.py --config default.json --joints 0 -0.785 0 -2.356 0 1.571 0.785
```

`--config` defaults to `default.json` and accepts only a filename in this directory.
Configuration lookup does not depend on the current working directory.
All fields are required; missing or unknown fields and invalid values cause an error before simulation starts.
The configuration controls physics and motion parameters. Task selection, model selection,
display mode, and other options remain command-line arguments.

| Field | Units and meaning |
| --- | --- |
| `gripper_max_width_m` | Metres. Total opening between the fingers; must not exceed the current Franka asset's 0.08 m limit. |
| `max_translation_cm` | Centimetres. Maximum translation per VLM action. |
| `max_rotation_deg` | Degrees. Maximum absolute rotation angle per VLM action. |
| `table_height_m` | Metres. Tabletop height in world coordinates. The workmat, target pad, objects, robot base, initial hand pose, and camera heights adjust accordingly. |
| `workspace_x_m`, `workspace_y_m` | Metres. TCP world-coordinate bounds, specified as `[min, max]`; both endpoints are excluded. |
| `workspace_z_above_table_m` | Metres. TCP height bounds relative to the tabletop, specified as `[min, max]`; both endpoints are included. Add the tabletop height to obtain world Z coordinates. |
| `action_duration_s` | Seconds. Requested duration of each VLM pose action. |
| `arm_duration_s` | Seconds. Default requested duration for standalone joint or pose controller calls. |
| `gripper_duration_s` | Seconds. Gripper action duration; the grasp controller interpolates during the first 75% and holds during the final 25%. |
| `initial_pose_duration_s` | Seconds. Requested duration for reaching the initial hand pose during task preparation. |
| `action_settle_s` | Seconds. Settling time after an action. |
| `completion_settle_s` | Seconds. Settling time after the model declares completion or a replay-only run ends. |
| `force_backoff_duration_s` | Seconds. Requested retreat duration after the laptop task's excessive-contact-force guard is triggered. |

The default configuration preserves the original values. Joint and pose motions remain subject
to the existing velocity limit, so their actual execution time may be longer.
The controller no longer exposes `--duration`; use `arm_duration_s` instead.
An explicit `duration` in a Python API call or a JSON task can still override the duration
of an individual action. Absolute target coordinates are not automatically adjusted when the table height changes.

The same configuration is used for action limits, gripper control, observation normalization, and prompts.
The `[GRIPPER_MAX_CM]` placeholder in the default prompt is filled with the maximum opening at runtime.
Custom prompts should use this placeholder instead of hardcoding an opening width.
Task-specific lift and retreat distances, object geometry, and other controller parameters are outside
the scope of these configuration fields.

Each closed-loop run saves `physics_config.json` and includes the full configuration in `run_config.json`.
When replaying a run that contains a configuration snapshot, the selected configuration must match it.
For older runs without a snapshot, select a matching configuration yourself.
