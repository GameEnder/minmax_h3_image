# MinMax H3 Still — Model Plugin for Wan2GP

A Wan2GP model plugin that generates still images with the MiniMax H3 video
models. Each generation runs one minimal H3 window and returns its middle
frame through Wan2GP's standard image pipeline, so results land in the image
gallery with the normal text-to-image interface.

## Models

| Model | Architecture | Source |
|---|---|---|
| MiniMax H3 33B FL2VA Text2Image | `minmax_h3_image` | Text prompt, optional start/end images |
| MiniMax H3 33B Ref2VA Image2Image | `minmax_h3_still_ref2va` | Text prompt plus reference images and reference/control videos |

Both appear under MinMax H3 in the Main Media Generator tab.

## Requirements

- Wan2GP 12.x (uses the `plugin_info.json` model-plugin discovery path)
- Checkpoints download automatically on first use from `DeepBeepMeep/MiniMax-H3`:
  - `MiniMax-H3-FL2VA_int8_convrot.safetensors` (FL2VA model)
  - `MiniMax-H3-Ref2VA_int8_convrot.safetensors` (Ref2VA model)
  - Qwen3-VL-32B text encoder (BF16 or INT8), video + audio VAEs
- The INT8 checkpoints keep VRAM/RAM usage low; a pruned variant can be
  substituted by changing the `URLs` in `defaults/*.json`

## Installation

Copy the `minmax_h3_image_plugin` folder into Wan2GP's `plugins/` directory
and restart Wan2GP. The plugin is discovered automatically; no other setup
is needed.

## How it works

- Inference delegates to Wan2GP's built-in H3 FL2VA/Ref2VA pipelines —
  no reimplementation of the model. The wrapper forces a minimal valid
  window (default 107 frames, floor 22), stock 24 fps, then returns the
  middle frame as a float image tensor.
- Frame count follows the incoming `frame_num` snapped to H3's 5+17k grid,
  so a `video_length` in a profile or settings tunes cost directly.
- Prompt-enhancer options mirror stock H3 (`An H3 Prompt from Text`,
  `An H3 Reference Prompt from Text + Ref. Images`), adapted to image-mode
  option keys.
- Live previews reuse Wan2GP's calibrated `minimax_h3` RGB factors.
- LoRAs resolve to the shared `minimax_h3` folder; the plugin registers
  both architectures with H3's AdaLN converter (`fl2va` / `ref2va`).

## Layout

```
minmax_h3_image_plugin/
├── plugin_info.json
├── README.md
├── __init__.py
├── defaults/
│   ├── minmax_h3_image.json
│   └── minmax_h3_still_ref2va.json
├── models/
│   └── minmax_h3_image/
│       ├── __init__.py
│       ├── minmax_h3_image_handler.py
│       └── minmax_h3_image_main.py
└── profiles/
    ├── minmax_h3_image/
    │   ├── Euler 30 Steps.json
    │   └── Turbo 8 Steps.json
    └── minmax_h3_still_ref2va/
        └── Euler 30 Steps.json
```

## License
Apache License
