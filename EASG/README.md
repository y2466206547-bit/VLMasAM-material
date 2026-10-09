# EASG

We provide code for zero-shot execution in simulation for “VLMs are Native Action Models: Unleashing the Manipulation Capabilities of VLMs with Explicit Action-Space Grounding.” Due to the gap between simulation and the real world, we do not guarantee that the success rates will match those reported in the paper or that all implementation details will be identical. The code is intended to demonstrate the core logic and implementation.

Two tasks are supported:

- `pick_place`: pick up the red cube, place it on the blue target area, release it, and retreat.
- `close_laptop`: close the laptop lid and withdraw the robot hand.

This guide covers setup, checks without model API calls, and online execution on Linux with Bash. A working pipeline does not guarantee that the model completes every task. Check the `success` field in `result.json` for the final outcome.

## 1. Requirements and Versions

This guide uses **Python 3.11 in a dedicated Conda environment, with Isaac Sim 5.1.0 installed through pip**. It does not rely on the Python interpreter bundled with a standalone Isaac Sim installation. Isaac Lab and ROS 2 are not required.

Versions and defaults:

| Component | Version or setting |
| --- | --- |
| Python | `3.11` (tested with `3.11.15`) |
| Conda environment name | `easg`; you may choose another name |
| Isaac Sim package version | `5.1.0.0`; use `==5.1.0` in the installation command |
| Default prompt | `script/prompt/prompt.txt` |
| Default physics configuration | `config/default.json` |

Install **Isaac Sim 5.1.0** to reproduce this setup, rather than installing an unpinned latest release. NVIDIA marks 5.1.0 as an unsupported release. It is pinned here for project compatibility; other versions have not been validated for this project. See the [official Python installation guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html).

Your system must meet the Isaac Sim requirements:

- Linux x86_64, preferably Ubuntu 22.04 or 24.04. Installation through pip requires GLIBC 2.35 or later.
- An NVIDIA GPU supporting RTX rendering and a compatible driver. `--headless` disables the window, not GPU rendering.
- The official 5.1 minimum specification lists 32 GB RAM, 16 GB VRAM, and 50 GB SSD storage. Allow additional space for project assets, caches, and recordings.
- GPUs without RT Cores, such as A100 and H100, are not supported. Select drivers according to the official requirements and compatibility checks.

See the [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html). Before installation, check:

```bash
nvidia-smi
ldd --version
```

## 2. Set Up Python and Install Isaac Sim

### Initial Installation

Install Conda, for example through Miniconda or Miniforge, then create a dedicated environment. Do not repeat the installation in an existing project environment:

```bash
conda create -n easg python=3.11 pip -y
conda activate easg

python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com
```

`all` installs the full package set; `extscache` installs cached extension dependencies. Access to the Python package indexes and NVIDIA download services is required. You do not need a separate desktop installation. See the [official installation command](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html).

Install the additional dependencies used directly by the project. These versions match the tested environment:

```bash
python -m pip install \
  numpy==1.26.0 Pillow==11.3.0 requests==2.32.3 \
  imageio==2.37.0 imageio-ffmpeg==0.6.0
```

Pillow handles images, requests handles model API calls, and imageio / imageio-ffmpeg write MP4 recordings. Install dependencies into the same Python environment used to run the project, not into the system Python.

After reading and accepting the [Omniverse License Agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html), set the following variables to avoid an interactive license prompt at startup:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
```

These settings apply only to the current terminal. Activate the environment and set them again in each new terminal. Check the interpreter, package version, and video encoder:

```bash
python -c 'import sys; from importlib.metadata import version; print(sys.executable); print(sys.version); print("Isaac Sim:", version("isaacsim"))'
python -c 'import numpy, PIL, requests, imageio, imageio_ffmpeg; print("Dependencies OK"); print(imageio_ffmpeg.get_ffmpeg_exe())'
```

### Subsequent Sessions

Install the environment only once. Before running the project in a new terminal, activate it and set the environment variables:

```bash
conda activate easg
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
```

All commands below use `python` from the activated environment. They do not depend on a developer-specific launcher or installation location.

## 3. Get the Project and Assets

Place the complete project folder in any working directory, then enter it:

```bash
cd EASG
```

Run all subsequent commands from the project root. Example paths are relative to that directory.

Main directories:

```text
EASG/
├── Asset/                    # Local USD assets, materials, and textures
├── config/
│   ├── default.json          # Physics and motion parameters
│   └── README.md             # Configuration reference
├── script/
│   ├── run.py                # Start a new closed-loop VLM task
│   ├── replay.py             # Replay actions, optionally resuming inference
│   ├── control_robot.py      # Joint, end-effector pose, and gripper control
│   ├── construct_scene/      # Scene construction
│   ├── prompt/prompt.txt     # Default prompt
│   └── test/                 # Local regression tests
└── output/                   # Run outputs; created automatically
```

Preserve the complete `Asset/` directory structure and its dependencies, not just the top-level USD files. The main asset entry points required by the closed-loop tasks include:

- `Asset/robot/franka/robot.usd`
- `Asset/environment/table0/instance.usd`
- `Asset/environment/envmap_lib/studio_small_03_1k.hdr`
- `Asset/objects/rigid/cube/object.usdz` (pick-and-place task)
- `Asset/objects/articulated/laptop/laptop9968/instance.usd` (laptop task)

The standalone scene initialization and control demos also load a mug, banana, and book. Installing Isaac Sim does not provide these project assets automatically. Obtain or copy the complete project asset collection when setting up another machine.

Assets are loaded from `Asset/` in the project root by default. If they are stored elsewhere, such as a sibling `easg-assets/` directory, use `--assets ../easg-assets` or set `EASG_ASSETS` to the asset directory.

## 4. Validate the Setup Without Model API Calls

### Regression Tests

```bash
python -m unittest discover -s script/test -p 'test_*.py' -v
```

These tests do not start the simulator or make real API requests, so they do not replace GPU and rendering checks. If `script/test/` is missing, obtain that directory separately before running the tests. A result reporting zero tests is not a successful validation.

### Create a Task Scene and Capture One Observation

```bash
python script/run.py --task pick_place --headless --observe-only \
  --output output/observe_pick_place
```

This command initializes the simulator, robot, and cameras, captures images, and generates a prompt. It does not call the VLM, require an API key, or execute model-generated actions. Scene preparation and initial robot motion still take place.

Inspect `external.png`, `wrist.png`, `wrist_tcp.png`, `observation.json`, and `prompt.txt` in `output/observe_pick_place/step_001/`. In observation-only mode, `success: false` does not indicate a setup failure: the task has not been executed.

To inspect the base tabletop scene separately:

```bash
python script/construct_scene/init_scene.py --headless --steps 120 --screenshot \
  --output output/scene_check
```

Inspect `preview.png`, `scene.json`, and `tabletop.usda` afterward. Omit `--headless` if a desktop display is available. The first launch may take time to load extensions and build rendering caches.

## 5. Configure the OpenRouter API Key

Choose either of the following methods.

Option 1: enter the key interactively in Bash to keep it out of your shell history:

```bash
read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY
printf '\n'
export OPENROUTER_API_KEY
```

Option 2: create a private JSON file outside the project with permissions set to `600`. It must contain a list with exactly one key:

```json
["YOUR_OPENROUTER_API_KEY"]
```

For example, store it at `../secrets/openrouter_keys.json` and add `--keys-file ../secrets/openrouter_keys.json` to the run command. When a file is specified, its key is used; otherwise, the client reads `OPENROUTER_API_KEY`. The project does not automatically load `.env` files. Never put real keys in source code, this README, or shared project files.

Authentication, permission, or quota errors (401 / 402 / 403) fail immediately. Network errors, rate limits, and certain temporary service errors are retried using the same key. Online execution sends camera images and prompts to the model service and incurs API charges.

## 6. Run the Full Closed Loop

### Pick and place cube

```bash
python script/run.py --task pick_place --headless \
  --output output/pick_place_run01
```

### Close laptop

```bash
python script/run.py --task close_laptop --headless \
  --output output/close_laptop_run01
```

Both tasks create their scenes through `run.py`; you do not need to run `init_scene.py` first. The laptop task uses the shared prompt, but this guide does not guarantee its success rate with that prompt.

The commands above use the code's current default model, `openai/gpt-6-astra`, with `xhigh` reasoning. Ensure your account can access the selected model. Use `--model` to select another available vision model and `python script/run.py --help` to list supported options. The model must support image input and the selected reasoning setting. For example, the following Qwen model has a model-specific default token budget in the client:

```bash
python script/run.py --task pick_place --headless \
  --model qwen/qwen3.8-27b --max-tokens 4096 --reasoning-effort xhigh \
  --chat-connection-mode continuous --max-task-steps 65 \
  --output output/pick_place_qwen_run01
```

Model availability depends on the provider and your account; the example does not guarantee task success. Full-history mode accumulates image context and typically consumes more tokens than the default single-turn mode.

Use a new `--output` directory for each run to avoid mixing or overwriting results. If omitted, `run.py` uses `output/vlmasam/<timestamp>/`.

### Common Options

| Option | Purpose |
| --- | --- |
| `--headless` | Run without a window; rendering remains enabled |
| `--gpu 0` | Select the rendering and physics GPU; default: 0 |
| `--config default.json` | Select a file in `config/`; accepts a filename only |
| `--prompt-template script/prompt/prompt.txt` | Override the prompt; this file is already the default |
| `--model MODEL_ID` | Select the model |
| `--max-task-steps 35` | Maximum closed-loop turns; default: 35, not simulation frames |
| `--chat-connection-mode separate` | Send only the current observation; default mode |
| `--chat-connection-mode continuous` | Send the full conversation history |
| `--chat-connection-mode first-recent --chat-recent-turns 2` | Retain the first turn and the two most recent turns of history |
| `--output PATH` | Set the run output directory |

See [config/README.md](config/README.md) for gripper opening, motion limits, workspace bounds, and timing parameters. `--steps` and `--screenshot` are only available for scene initialization and control demos, not for `run.py`.

## 7. Inspect Results and Replay Runs

Each online iteration captures images, renders the prompt, queries the VLM, parses and validates the action, executes it, and evaluates the physical task state before the next iteration.

Main outputs:

- `result.json`: final physical evaluation; success requires `success: true`.
- `execution.mp4`: run recording.
- `run_config.json` and `physics_config.json`: task settings and configuration records.
- `prompt_source.txt` and `prompt_template.txt`: prompt snapshots. A `system_prompt.txt` snapshot is also saved when a system policy is used.
- `api_usage.json`: recorded API usage.
- `step_XXX/`: images, observations, prompts, model responses, actions, and evaluations for each turn. Execution failures also produce `execution_error.txt`.

A model's completion claim is not proof of physical task success. If the step limit is reached without passing evaluation, the program reports `Task did not succeed`. Inspect the recording and per-step feedback.

Replay an existing run without model calls or an API key:

```bash
python script/replay.py --task pick_place --headless \
  --resume-from output/pick_place_run01 --replay-only \
  --output output/pick_place_replay01
```

Replay and then resume online inference, which requires an API key:

```bash
python script/replay.py --task pick_place --headless \
  --resume-from output/pick_place_run01 --max-task-steps 65 \
  --output output/pick_place_resume01
```

The task type and physics configuration must match the source run, and the output directory must be different. `--max-task-steps` includes replayed steps. Resuming re-executes recorded actions; it does not restore a simulator state snapshot. Explicitly select the original model, prompt, and conversation mode as needed rather than assuming all settings are restored automatically.

## 8. Troubleshooting

- **Missing `isaacsim` or dependencies:** run `python -c 'import sys; print(sys.executable)'` and ensure installation and execution use the same environment. Do not launch the project with the system Python.
- **pip cannot find the 5.1.0 package:** check that Python is 3.11, Linux GLIBC is at least 2.35, and the NVIDIA package index is accessible.
- **GPU, Vulkan, or renderer initialization fails:** check `nvidia-smi`, RTX support, and driver compatibility. Headless mode does not bypass these requirements.
- **Missing assets, materials, or textures:** check the complete `Asset/` directory, referenced dependencies, and the `--assets` path. Copying only the top-level `.usd` file is usually insufficient.
- **Missing prompt:** ensure `script/prompt/prompt.txt` exists or select a valid file with `--prompt-template`.
- **401 / 402 / 403:** check the key's validity, permissions, and balance. The client does not switch keys automatically.
- **Empty or incorrectly formatted model response:** inspect `response.txt`, `model_message.json`, the token budget, and model parameter support. Only the pipe-delimited action format is accepted, not JSON, explanations, or Markdown code blocks.
- **Video recording fails:** check `imageio` and `imageio-ffmpeg` in the same Python environment, along with output directory permissions and free disk space.

Follow this order: **verify versions and dependencies → capture observations without API calls → configure a key → run the online loop → inspect the physical result**. Passing unit tests or receiving a model completion claim does not establish task success.
