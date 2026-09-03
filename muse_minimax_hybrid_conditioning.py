"""MiniMax H3 hybrid ref2va + fl2va conditioning.

Keeps reference blocks and first/last keyframe pins in ONE conditioning payload.
The H3 core already supports both payload fields in PackedLayout.
"""
import math
import torchaudio

import nodes
import node_helpers
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_minimax_h3 import (
    _empty_av_latent, _resize, adapt_canvas,
    CANVAS_MULTIPLE, REF_IMAGE_SHORT_EDGE, FPS,
)


class MiniMaxH3HybridRefAndKeyframe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HybridRefAndKeyframe",
            display_name="MiniMax H3 Hybrid Cond (R2V + I2V)",
            category="model/conditioning/minimax",
            description=("Combines H3 reference blocks with optional first/last keyframe pins. "
                         "Use <Picture i>, <Video k>, and <Audio j> for reference media."),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="Frames at 24 fps; snaps to 17k+5."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video"), prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"), prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3)),
                io.Boolean.Input("also_ref_first_frame", default=False,
                                 tooltip="Also expose first_frame as the next <Picture N> reference."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @staticmethod
    def _encode_ref_audio(audio_vae, audio):
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        z = audio_vae.encode(waveform[:1].movedim(1, -1))
        return z, z.shape[-1]

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                ref_image_size="match", first_frame=None, last_frame=None,
                ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None, also_ref_first_frame=False):
        latent, frame_count = _empty_av_latent(width, height, length)
        keyframes = []
        keyframe_images = []

        if first_frame is not None:
            img = _resize(first_frame[:1], width, height, "disabled")
            keyframe_images.append(img)
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            img = _resize(last_frame[:1], width, height, "center")
            keyframe_images.append(img)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

        ref_items, ref_blocks = [], []
        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            scale = (min(1.0, math.sqrt((width * height) / (w * h)))
                     if ref_image_size == "match"
                     else min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h)))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16,
                               "latent_w": tw // 16, "latent": vae.encode(resized)})

        if also_ref_first_frame and first_frame is not None:
            # It is deliberately appended after user refs: it becomes the next Picture ordinal.
            ref_items.append({"type": "image", "data": keyframe_images[0]})
            ref_blocks.append({"kind": "image", "latent_h": height // 16,
                               "latent_w": width // 16, "latent": vae.encode(keyframe_images[0])})

        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            audio_latent, ref_audio_t = None, 0
            if soundtrack is not None:
                audio_latent, ref_audio_t = cls._encode_ref_audio(audio_vae, soundtrack)
                ref_items.append({"type": "audio"})
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            ref_items.append({"type": "video", "data": frames[sample_idx],
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            video_latent = vae.encode(frames)
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": video_latent.shape[2], "latent_h": ch // 16,
                               "latent_w": cw // 16, "ref_audio_t": ref_audio_t,
                               "latent": video_latent, "audio_latent": audio_latent})

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            audio_latent, ref_audio_t = cls._encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                               "audio_latent": audio_latent})

        if ref_items:
            tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        else:
            tokens = clip.tokenize(prompt, images=keyframe_images)
        cond = clip.encode_from_tokens_scheduled(tokens)

        values = {}
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            values["minimax_keyframes"] = keyframes
            values["minimax_frame_count"] = frame_count
        if ref_blocks:
            values["minimax_refs"] = ref_blocks
        if values:
            cond = node_helpers.conditioning_set_values(cond, values)
        return io.NodeOutput(cond, latent)


class MiniMaxH3HybridExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3HybridRefAndKeyframe]
