# Zero-Shot Demos

Each task folder contains:

```text
<task>/
├── execution_wrist_rgb.mp4  # Wrist-camera execution video (5 FPS)
├── images/                 # Per-step observations: RGB, depth, TCP annotations, and camera metadata
└── steps/                  # Per-step records: prompts, VLM responses, actions, and target poses; plus task metadata
```

Matching step numbers link observations to their corresponding decisions and actions. External-view images and task results are included where available.
