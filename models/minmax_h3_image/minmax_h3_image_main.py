"""MinMax H3 Still model factory for WanGP.

Uses the MiniMax H3 video model (FL2VA) to create stills: generates one
minimal H3 window and returns its first frame through Wan2GP's standard
image-output contract ``(C, F, H, W)`` float in ``[-1, 1]``.

All heavy lifting (omni transformer, Qwen3-VL text encoder, video/audio
VAEs) is delegated to Wan2GP's built-in H3 pipeline instead of being
reimplemented here.
"""

import torch


# Smallest window we will attempt (5 + 17 frames of H3 grid, ~0.9 s).
# Below H3's 4 s training range, so quality there is experimental --
# 107 (Wan2GP's H3 minimum) remains the default.
H3_MIN_FRAMES = 107
H3_FLOOR_FRAMES = 22


def _snap_frame_count(frame_num):
    """Snap a requested count to the H3 grid (5 + 17k), floored for safety."""
    try:
        requested = int(frame_num)
    except (TypeError, ValueError):
        return H3_MIN_FRAMES
    if requested < H3_FLOOR_FRAMES:
        return H3_FLOOR_FRAMES
    return 5 + ((requested - 5) // 17) * 17


class _H3ImageWrapper:
    """Presents a built-in H3 FL2VA pipeline as an image-output model."""

    def __init__(self, pipeline):
        self._pipeline = pipeline
        # Aliases mirroring the built-in H3 pipe mapping so the handler can
        # expose the same offload targets.
        self.transformer = pipeline.transformer
        self.unet = pipeline.transformer
        text_encoder = pipeline.text_encoder
        self.text_encoder = getattr(text_encoder, "language_model", text_encoder)
        self.vision_encoder = getattr(text_encoder, "visual", None)
        self.vae = getattr(pipeline, "video_decoder", getattr(pipeline, "vae", None))
        self.video_encoder = getattr(pipeline, "video_encoder", None)
        self.audio_vae = getattr(pipeline, "audio_vae", None)
        self.latent_upscaler = getattr(pipeline, "latent_upscaler", None)

    @property
    def _interrupt(self):
        return self._pipeline._interrupt

    @_interrupt.setter
    def _interrupt(self, value):
        self._pipeline._interrupt = value

    def generate(self, frame_num=None, **kwargs):
        """Run one minimal H3 window, return its middle frame as an image."""
        kwargs["frame_num"] = _snap_frame_count(
            H3_MIN_FRAMES if frame_num is None else frame_num
        )
        # Stock video runs at 24 fps; image mode arrives with fps=1, which
        # only stretches the (discarded) soundtrack — 107 s of audio instead
        # of ~4.5 s. Verified: fps touches audio/metadata only, never pixels.
        kwargs["fps"] = 24
        result = self._pipeline.generate(**kwargs)
        if result is None:
            return None
        video = result.get("x", None) if isinstance(result, dict) else result
        if video is None:
            return None
        mid = video.shape[1] // 2
        frame = video[:, mid:mid + 1].detach()
        if frame.dtype == torch.uint8:
            frame = frame.float().div(127.5).sub(1.0)
        else:
            frame = frame.float().clamp(-1.0, 1.0)
        return {"x": frame.cpu().contiguous()}


# Our FL2VA weights are stock full FL2VA (e.g. MiniMax-H3-FL2VA_int8_convrot)
# and our Ref2VA weights are stock full Ref2VA, so AdaLN LoRA conversion
# must treat each architecture as its matching family.
H3_ARCHITECTURES = {
    "minmax_h3_image": "fl2va",
    "minmax_h3_still_ref2va": "ref2va",
}
H3_MODEL_TYPES = {
    "minmax_h3_image": "minimax_h3_fl2va",
    "minmax_h3_still_ref2va": "minimax_h3_ref2va",
}


def _register_h3_image_architecture():
    """Teach H3's AdaLN LoRA converter our architecture name.

    The converter (models.minimax_h3.lora_affine) only knows built-in arch
    strings, so any LoRA (e.g. the Turbo profile) crashes with
    "Unsupported MiniMax H3 architecture". Registration runs here because
    model loading always precedes LoRA loading, and it only adds our own
    key -- built-in entries are never touched. Best-effort: if Wan2GP ever
    moves this map, the original error resurfaces instead of a new one.
    """
    try:
        from models.minimax_h3 import lora_affine
    except Exception:
        return
    architectures = getattr(lora_affine, "_ARCHITECTURES", None)
    if not isinstance(architectures, dict):
        return
    for arch, family in H3_ARCHITECTURES.items():
        architectures.setdefault(arch, family)


def model_factory(
    model_filename=None,
    model_type=None,
    base_model_type=None,
    model_def=None,
    text_encoder_filename=None,
    quantizeTransformer=False,
    dtype=torch.bfloat16,
    VAE_dtype=torch.float32,
    save_quantized=False,
    **kwargs,
):
    """Create an H3-backed still-image model processor (FL2VA or Ref2VA)."""
    from models.minimax_h3.minimax_h3_main import (
        AUDIO_VAE_FILE,
        VIDEO_VAE_FILE,
        model_factory as h3_model_factory,
    )

    _register_h3_image_architecture()

    if model_def is None:
        raise ValueError("MinMax H3 Still requires a model definition.")
    if model_filename is None:
        raise ValueError("No transformer checkpoint was provided for MinMax H3 Still.")
    if text_encoder_filename is None:
        raise ValueError("No Qwen3-VL text encoder checkpoint was provided for MinMax H3 Still.")

    reference_mode = base_model_type == "minmax_h3_still_ref2va"
    # NOTE: upstream dropped VAE_dtype from model_factory (VAE dtype is now
    # pinned internally), so it is accepted here for tolerance but no longer
    # forwarded. Keep the parameter so older callers don't break.
    pipeline = h3_model_factory(
        model_filename,
        text_encoder_filename,
        qkv_splitting=True,
        dtype=dtype,
        save_quantized=save_quantized,
        model_type=H3_MODEL_TYPES.get(base_model_type, "minimax_h3_fl2va"),
        reference_mode=reference_mode,
        video_vae_filename=VIDEO_VAE_FILE,
        audio_vae_filename=AUDIO_VAE_FILE,
    )
    return _H3ImageWrapper(pipeline)
