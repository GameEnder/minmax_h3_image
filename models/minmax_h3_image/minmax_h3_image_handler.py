"""MinMax H3 Still WanGP model integration."""

import os

import torch
from shared.utils.hf import build_hf_url


REPO_ID = "DeepBeepMeep/MiniMax-H3"
MODEL_TYPE = "minmax_h3_image"
MODEL_TYPE_REF = "minmax_h3_still_ref2va"
TEXT_ENCODER_FOLDER = "Qwen3-VL-32B-Instruct"
TEXT_ENCODER_BF16 = "Qwen3-VL-32B-Instruct-layer50_bf16.safetensors"
TEXT_ENCODER_INT8 = "Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors"
VIDEO_VAE_FILE = "MiniMax-H3-video_vae_fp16.safetensors"
AUDIO_VAE_FILE = "MiniMax-H3-audio_vae_fp32.safetensors"


MINMAX_H3_IMAGE_INFOS = """
# MinMax H3 Still Plugin

This model is provided by a WanGP model plugin. It adds MinMax H3 Still to the
Main Media Generator Tab using the same handler keys as built-in models.

## Included Plugin Features

- Metadata-only plugin discovery through `plugin_info.json`.
- Plugin-provided defaults and profiles.
- A handler-declared text encoder download set.
- MMGP-managed transformer, text encoder, and VAE offloading.
"""


MINMAX_H3_IMAGE_PROMPT_INFOS = """
# MinMax H3 Still Prompt Notes

MinMax H3 Still responds best to concise, concrete image descriptions.
Put the subject first, then add style, composition, lighting, and quality cues.

## Useful Prompt Shape

```text
subject, setting, style or medium, camera/framing, lighting, color palette, detail level
```

## Negative Prompt

Use the negative prompt to remove common artifacts such as blur, distorted
anatomy, duplicate limbs, text, watermark, or low quality output.
"""


MINMAX_H3_IMAGE_PROMPT_ENHANCER = """
You are a MinMax H3 Still prompt writing assistant. Rewrite the user's idea into one image-generation prompt.

Output rules:
- Output only the final prompt text.
- Do not include explanations, markdown, bullet lists, or quotes around the prompt.
- Preserve the user's subject and intent.
- Add concrete visual details: medium, composition, lighting, color, mood, and texture.
- Keep the result compatible with MinMax H3 Still generation.
- Write one dense sentence or a short comma-separated prompt.
"""


class family_handler:
    @staticmethod
    def query_supported_types():
        return [MODEL_TYPE, MODEL_TYPE_REF]

    @staticmethod
    def query_family_infos():
        return {"minimax_h3": (190, "MinMax H3")}

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "minimax_h3"

    @staticmethod
    def query_model_def(base_model_type, model_def):
        reference_mode = base_model_type == MODEL_TYPE_REF
        result = {
            "image_outputs": True,
            "profile_type": "image",
            "guidance_max_phases": 1,
            "visible_phases": 0,
            "flow_shift": True,
            "no_negative_prompt": True,
            "frames_minimum": 22,
            "frames_steps": 17,
            "frames_offset": 5,
            "block_size": 32,
            "vae_block_size": 32,
            "first_block_cache": True,
            "first_block_cache_thresholds": [0.06, 0.08, 0.10, 0.12, 0.14],
            "skip_steps_multiplier_choices": [
                ("Low (0.06)", 0.06),
                ("Balanced (0.08, upstream default)", 0.08),
                ("High (0.10)", 0.10),
                ("Very High (0.12)", 0.12),
                ("Maximum (0.14)", 0.14),
            ],
            "sample_solvers": [("Euler", "euler"), ("RES Multistep", "res_multistep"), ("Ralston 2S (~2x slower)", "ralston_2s")],
            "preset_profiles_dir": [MODEL_TYPE],
            "profiles_dir": [],
            "infos": model_def.get("infos", MINMAX_H3_IMAGE_INFOS),
            "prompt_infos": model_def.get("prompt_infos", MINMAX_H3_IMAGE_PROMPT_INFOS),
            "prompt_enhancer_button_label": "Improve Prompt",
            "prompt_enhancer_def": {
                "selection": ["T", "TI"],
                "labels": {
                    "TP": "An H3 Prompt from Text",
                    "TIP": "An H3 Prompt from Text + {image_inputs}",
                },
                "default": "",
            },
            "text_prompt_enhancer_instructions": MINMAX_H3_IMAGE_PROMPT_ENHANCER,
            "image_prompt_enhancer_instructions": MINMAX_H3_IMAGE_PROMPT_ENHANCER,
            "text_prompt_enhancer_max_tokens": 512,
            "image_prompt_enhancer_max_tokens": 512,
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "text_encoder_URLs": [
                build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_BF16),
                build_hf_url(REPO_ID, TEXT_ENCODER_FOLDER, TEXT_ENCODER_INT8),
            ],
            "resolutions": [
                ("512x512 (1:1)", "512x512"),
                ("576x576 (1:1)", "576x576"),
                ("768x768 (1:1)", "768x768"),
                ("1024x1024 (1:1)", "1024x1024"),
                ("1088x1088 (1:1)", "1088x1088"),
                ("1536x1536 (1:1)", "1536x1536"),
                ("768x512 (3:2)", "768x512"),
                ("512x768 (2:3)", "512x768"),
                ("1536x1024 (3:2)", "1536x1024"),
                ("1024x1536 (2:3)", "1024x1536"),
                ("1632x1088 (3:2)", "1632x1088"),
                ("1088x1632 (2:3)", "1088x1632"),
                ("1024x768 (4:3)", "1024x768"),
                ("768x1024 (3:4)", "768x1024"),
                ("768x576 (4:3)", "768x576"),
                ("576x768 (3:4)", "576x768"),
                ("1440x1088 (4:3)", "1440x1088"),
                ("1088x1440 (3:4)", "1088x1440"),
                ("1280x768 (5:3)", "1280x768"),
                ("768x1280 (3:5)", "768x1280"),
                ("1344x768 (7:4)", "1344x768"),
                ("768x1344 (4:7)", "768x1344"),
                ("1024x576 (16:9)", "1024x576"),
                ("576x1024 (9:16)", "576x1024"),
                ("1920x1088 (16:9)", "1920x1088"),
                ("1088x1920 (9:16)", "1088x1920"),
                ("2048x1152 (16:9)", "2048x1152"),
                ("1152x2048 (9:16)", "1152x2048"),
                ("2560x1088 (21:9)", "2560x1088"),
            ],
            "no_background_removal": True,
        }
        if reference_mode:
            result.update({
                "preset_profiles_dir": [MODEL_TYPE_REF],
                "prompt_enhancer_def": {
                    "selection": ["T", "TI"],
                    "labels": {
                        "TP": "An H3 Reference Prompt from Text",
                        "TIP": "An H3 Reference Prompt from Text + {image_inputs}",
                    },
                    "default": "",
                },
                "image_ref_choices": {
                    "choices": [("Generate without Reference Images", ""),
                                ("Use Reference Images", "I"),
                                ("First Reference Image is the Main Subject / Landscape, defines Output Dimensions, and may be followed by other Reference Images", "KI")],
                    "letters_filter": "KI",
                    "default": "",
                    "label": "Reference Images",
                },
                "reference_image_enabled": True,
                "return_image_refs_tensor": False,
                "fit_into_canvas_image_refs": 0,
                "any_image_refs_relative_size": True,
                "image_refs_relative_size": {"min": 50, "max": 400, "step": 1},
                "guide_custom_choices": {
                    "choices": [("Generate without a Reference or Control Video", ""), ("Use One Reference Video", "V-U"),
                                ("Use Two Reference Videos", "V+-U"),
                                ("Transfer Depth Map From Control Video", "DV"),
                                ("Provide Generic Control Video", "GV")],
                    "letters_filter": "GPDEV+-U",
                    "default": "",
                    "label": "Reference / Control Video",
                },
                "preprocess_video_guide2": True,
                "reference_video_enabled": True,
                "reference_video_max_frames": 15 * 24,
                "reference_video_max_size": (768, 1344),
                "video_guide_label": "Reference / Control Video 1",
                "video_guide2_label": "Reference Video 2",
            })
        return result

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        # Main transformer stays in model.URLs. Everything else needed lives
        # here (mirrors the built-in H3 handler: "" = repo root files).
        return [
            {
                "repoId": REPO_ID,
                "sourceFolderList": ["", TEXT_ENCODER_FOLDER],
                "fileList": [
                    [
                        VIDEO_VAE_FILE,
                        AUDIO_VAE_FILE,
                    ],
                    [
                        "config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "preprocessor_config.json",
                        "vocab.json",
                    ],
                ],
            }
        ]

    @staticmethod
    def get_lora_dir(base_model_type):
        # New-style LoRA API: return a key, Wan2GP resolves it to
        # <lora_root>/<key> and creates the directory (resolve_lora_dir).
        # Shares the MinMax H3 LoRA folder with the built-in H3 models.
        return "minimax_h3"

    @staticmethod
    def set_cache_parameters(cache_type, base_model_type, model_def, inputs, skip_steps_cache):
        # Mirrors the built-in H3 handler: the First Block Cache threshold
        # is the user multiplier. Called by Wan2GP whenever a cache type is
        # active, so enabling first_block_cache requires this companion.
        if cache_type == "first_block":
            skip_steps_cache.threshold = float(skip_steps_cache.multiplier)
        elif cache_type != "spectrum":
            raise ValueError(f"MiniMax H3 Image does not support step-skipping type {cache_type!r}")

    @staticmethod
    def get_rgb_factors(base_model_type):
        # Live diffusion previews project latents to RGB via these factors.
        # Our latents ARE H3 latents (built-in FL2VA pipeline), so the stock
        # calibrated "minimax_h3" factors apply verbatim.
        from shared.RGB_factors import get_rgb_factors
        return get_rgb_factors("minimax_h3")

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        **kwargs,
    ):
        from .minmax_h3_image_main import model_factory

        pipe_processor = model_factory(
            model_filename=model_filename,
            model_type=model_type,
            base_model_type=base_model_type,
            model_def=model_def,
            text_encoder_filename=text_encoder_filename,
            quantizeTransformer=quantizeTransformer,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            save_quantized=save_quantized,
        )
        pipe = {
            "transformer": pipe_processor.transformer,
            "text_encoder": pipe_processor.text_encoder,
            "vae": pipe_processor.vae,
        }
        if pipe_processor.vision_encoder is not None:
            pipe["vision_encoder"] = pipe_processor.vision_encoder
        if pipe_processor.video_encoder is not None:
            pipe["video_encoder"] = pipe_processor.video_encoder
        if pipe_processor.audio_vae is not None:
            pipe["audio_vae"] = pipe_processor.audio_vae
        if pipe_processor.latent_upscaler is not None:
            pipe["latent_upscaler"] = pipe_processor.latent_upscaler
        return pipe_processor, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        # H3 is CFG-distilled: guidance 1.0 and shift 12.0 (same as the
        # built-in H3 handler). SD-era values (7.5 / ~5) overexpose the
        # image and amplify block artifacts.
        ui_defaults.update({
            "image_mode": 1,
            "resolution": "1024x1024",
            "num_inference_steps": 30,
            "guidance_scale": 1.0,
            "flow_shift": 12.0,
            "sample_solver": "euler",
            "skip_steps_cache_type": "first_block",
            "skip_steps_start_step_perc": 25,
            "skip_steps_multiplier": 0.08,
        })

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        # Image-only model: repair stale video state (e.g. image_mode carried
        # over from H3 video siblings) so the UI builds the text2image
        # interface instead of the video-length row (wgp.py image_mode logic).
        ui_defaults["image_mode"] = 1
