"""
Muse Minimax Refine V1.4 — finishes a Seed Hunt candidate scouted by
MuseMinimaxDirectorV1_2TwoStageBeta (the "TwoStage-Beta" package), instead of the
plain MuseMinimaxDirector V1.2 candidates V1.3 targets.

V1.3 (muse_minimax_refine_v1_3.py, kept unmodified alongside this file — additive,
not a replacement) upscales with plain comfy.utils.common_upscale() interpolation
and has no continuity option stronger than a pixel-VAE-reencoded carry. Both of
those were changed on the Beta Director itself after real testing:
  - The upscale step was swapped to MinimaxH3LatentUpscaler3D, a real trained
    latent-upscale network (Comfyui_Minimax_h3_latent_Upscaler) — confirmed via a
    real render to be a genuine improvement over naive interpolation.
  - A second continuity mechanism, raw_latent_carry_test, was added alongside the
    existing vae_reencode_carry_test (Auto-Chain): a real render + frame-dump
    comparison (chat, 2026-08-30) confirmed Auto-Chain only ever adds extra
    CONDITIONING the model is free to ignore — it does not reliably hold room/prop
    geometry across a chunk boundary. raw_latent_carry_test instead genuinely
    freezes part of the real sampled latent (via MiniMaxH3GeneratedAVMaskedContext,
    same node V1.3's own carry already uses, but fed the RAW previous-chunk latent
    instead of a lossy VAE re-encode of decoded pixels).

Refining a Beta-scouted candidate through V1.3 would silently use the OLD upscale
method and the weaker carry — not what actually generated the candidate, and not
what Director's own finishing pass would have used had two_stage_seed_hunt_latent_only
been off. This node exists so the finishing pass matches the pass that scouted it.

Auto-Chain (vae_reencode_carry_test) is deliberately NOT ported here: it comes from
a GPL-licensed file that only lives in the Beta package, kept isolated there on
purpose (see that package's own LICENSE/NOTICE/THIRD_PARTY_NOTICES.AUTOCHAIN.md).
Today's own finding is that mechanism is the weaker of the two anyway, so there is
no real loss — raw_latent_carry_test (default on here, since it's the one now
believed to actually work) is the only continuity mechanism this node offers beyond
V1.3's original always-on pixel-VAE carry, which stays as the fallback when it's off.

Everything else (scout-bundle reading, candidate selection, ref_images/ref_audio
handling, First/Last-Frame keyframe re-lock) is unchanged from V1.3 — that contract
(the _muse_scout_bundle format, _muse_model_used / _muse_keyframe_* embedding) is
shared by both Director packages, confirmed directly: the Beta Director's own
scout-bundle-saving code is identical in shape to V1.2's.
"""

import logging
import math
import os

import comfy.samplers
import comfy.utils
import folder_paths
import torch

from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo, CANVAS_MULTIPLE, REF_IMAGE_SHORT_EDGE, align_frame_count, _resize
from comfy_execution.graph import ExecutionBlocker
import node_helpers

log = logging.getLogger(__name__)


# ── Learned latent upscaler model registration ──────────────────────────────
# Duplicated from the Beta Director rather than imported from it — this package
# must keep working standalone (V1.3's own stated reason for duplicating its
# helpers applies just as much here), and importing across custom_nodes packages
# is fragile (load order, the other package not being installed at all).
_LATENT_UPSCALE_MODEL_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_MODEL_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_MODEL_FOLDER, os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_MODEL_FOLDER))


def _scan_latent_upscale_models():
    names = [
        name for name in folder_paths.get_filename_list(_LATENT_UPSCALE_MODEL_FOLDER)
        if name.lower().endswith((".safetensors", ".pt", ".pth", ".ckpt"))
    ]
    return names if names else [f"(place a model in ComfyUI/models/{_LATENT_UPSCALE_MODEL_FOLDER}/)"]


# ── Shared helpers — duplicated from muse_minimax_refine_v1_3.py verbatim ───

def _execute_comfy_node(node_class, **kwargs):
    """Invoke a ComfyUI node's main entrypoint, whether it is a comfy_api io.ComfyNode
    (classmethod 'execute') or a legacy node (instance method named by FUNCTION)."""
    if hasattr(node_class, "execute"):
        return node_class.execute(**kwargs)
    fn_name = getattr(node_class, "FUNCTION", None)
    instance = node_class()
    if fn_name and hasattr(instance, fn_name):
        return getattr(instance, fn_name)(**kwargs)
    raise RuntimeError(f"Could not determine how to execute node {node_class!r}")


def _unpack_node_result(out):
    """Normalise a node return (io.NodeOutput, tuple, list or dict) into a tuple of outputs."""
    if out is None:
        return ()
    for attr in ("result", "args", "values", "outputs"):
        if hasattr(out, attr):
            val = getattr(out, attr)
            if callable(val):
                try:
                    val = val()
                except Exception:
                    continue
            if isinstance(val, (tuple, list)):
                return tuple(val)
    if isinstance(out, (tuple, list)):
        return tuple(out)
    if isinstance(out, dict) and isinstance(out.get("result"), (tuple, list)):
        return tuple(out["result"])
    return (out,)


def _load_scout_chunk(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch builds.
        return torch.load(path, map_location="cpu")


def _rebuild_keyframe_conditioning(positive, vae, first_frame, last_frame, tgt_w_px, tgt_h_px, frame_count, guide_frames=None):
    """Mirrors MuseMinimaxDirector's own _rebuild_keyframe_conditioning_for_stage2 exactly
    (same resize conventions, same conditioning keys) — ported here rather than imported,
    same reasoning V1.3 already documents for its own copy of this function."""
    if first_frame is None and last_frame is None and not guide_frames:
        return positive
    if frame_count is None:
        log.warning("[MuseMinimaxRefineV14] Keyframe image(s) were provided but no frame_count came with "
                    "them — skipping the keyframe conditioning rebuild. Refining without the "
                    "First/Last-Frame lock.")
        return positive
    new_keyframes = []
    if first_frame is not None:
        img = _resize(first_frame[:1], tgt_w_px, tgt_h_px, "disabled")
        new_keyframes.append({"resolved_frame_index": 0, "latent": vae.encode(img)})
    for guide in (guide_frames or []):
        image = guide.get("image")
        if image is None:
            continue
        img = _resize(image[:1], tgt_w_px, tgt_h_px, "center")
        new_keyframes.append({"resolved_frame_index": int(guide["frame_idx"]), "latent": vae.encode(img)})

    # Deliberately do not re-encode the supplied last frame at Stage 2. Stage 1 has
    # already used it to shape the trajectory; imposing a fresh high-resolution copy
    # here caused the generated approach to snap back to the source image near the end.
    return node_helpers.conditioning_set_values(
        positive, {"minimax_keyframes": new_keyframes, "minimax_frame_count": frame_count})


def _latent_resize_ref_block(blk, tgt_w_latent, tgt_h_latent):
    """Legacy-fallback path for one visual minimax_refs block: no original source
    pixel is available to re-encode fresh, so just latent-space resize the EXISTING
    (Stage-1-resolution) latent up to the Stage-2 canvas and correct its latent_h/
    latent_w bookkeeping to match. Keeps the block's own metadata self-consistent
    and raises its token density, but — unlike a fresh re-encode — cannot recover
    any detail the Stage-1 encode never captured. Mirrors Director's own copy of
    this function exactly (see its docstring there for the full reasoning)."""
    old_samples = blk["latent"]["samples"] if isinstance(blk["latent"], dict) else blk["latent"]
    new_samples = comfy.utils.common_upscale(
        old_samples, int(tgt_w_latent), int(tgt_h_latent), "bicubic", "disabled")
    new_blk = dict(blk)
    new_blk["latent"] = {"samples": new_samples} if isinstance(blk["latent"], dict) else new_samples
    new_blk["latent_h"] = int(tgt_h_latent)
    new_blk["latent_w"] = int(tgt_w_latent)
    return new_blk


def _rebuild_refs_conditioning_for_stage2(positive, vae, tgt_w_latent, tgt_h_latent,
                                          ref_image_size="match", ref_images_dict=None,
                                          log_label=""):
    """Mirrors MuseMinimaxDirector's own _rebuild_refs_conditioning_for_stage2 exactly
    (same sizing rule, same fresh-reencode-with-legacy-fallback shape) — ported here
    rather than imported, same reasoning this file already documents for its own copy
    of _rebuild_keyframe_conditioning. Only image refs need handling here: refine's
    own MiniMaxH3ReferenceToVideo call above never wires ref_videos in at all (video-
    ref support doesn't exist in this node), so minimax_refs can only ever contain
    image and/or audio blocks by construction — audio blocks pass through unchanged.

    ref_images_dict here is the exact same dict that built `positive` a few lines
    above in _refine_one_chunk_beta — not a separately-persisted original, so the
    'legacy fallback' branch below is mostly defensive: it only fires if a block in
    positive's minimax_refs has no corresponding entry in ref_images_dict at all,
    which in practice means ref_images_dict changed shape between building positive
    and calling this function. Kept anyway, logged distinctly, rather than assuming
    it can never happen."""
    existing_refs = positive[0][1].get("minimax_refs") if positive else None
    if not existing_refs:
        return positive

    orig_images = list((ref_images_dict or {}).values())
    tgt_w_px, tgt_h_px = int(tgt_w_latent) * 16, int(tgt_h_latent) * 16
    img_cursor = 0
    fresh_count, fallback_count = 0, 0
    new_refs = []
    for blk in existing_refs:
        if blk.get("kind") != "image":
            new_refs.append(blk)
            continue
        src = orig_images[img_cursor] if img_cursor < len(orig_images) else None
        img_cursor += 1
        if src is None:
            new_refs.append(_latent_resize_ref_block(blk, tgt_w_latent, tgt_h_latent))
            fallback_count += 1
            continue
        h, w = src.shape[1], src.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((tgt_w_px * tgt_h_px) / (w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(src[:1], tw, th, "disabled")
        z = vae.encode(resized)
        new_blk = dict(blk)
        new_blk["latent_h"], new_blk["latent_w"], new_blk["latent"] = th // 16, tw // 16, z
        new_refs.append(new_blk)
        fresh_count += 1

    if fresh_count:
        log.info("[MuseMinimaxRefineV14] %s: freshly re-encoded %d Stage-2 visual "
                  "reference(s) from source pixels.", log_label, fresh_count)
    if fallback_count:
        log.info("[MuseMinimaxRefineV14] %s: scaled %d existing reference latent(s) "
                  "as legacy fallback (no original source pixels in scope).", log_label, fallback_count)

    return node_helpers.conditioning_set_values(positive, {"minimax_refs": new_refs})


def _rebuild_stage1_continuation(model, clip, vae, audio_vae, prompt, old_stage1_latent,
                                 first_frame, last_frame, frame_count, seed, steps,
                                 first_pass_steps, sampler_name, scheduler,
                                 carry_images, carry_audio, carry_length):
    """Re-run a continuation chunk's Stage 1 from the newly refined predecessor."""
    from nodes import NODE_CLASS_MAPPINGS
    BasicGuider = NODE_CLASS_MAPPINGS["BasicGuider"]
    BasicScheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]
    KSamplerSelect = NODE_CLASS_MAPPINGS["KSamplerSelect"]
    RandomNoise = NODE_CLASS_MAPPINGS["RandomNoise"]
    SamplerCustomAdvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]
    SplitSigmas = NODE_CLASS_MAPPINGS["SplitSigmas"]
    VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]
    VAEEncodeAudio = NODE_CLASS_MAPPINGS["VAEEncodeAudio"]
    MaskedContext = NODE_CLASS_MAPPINGS["MiniMaxH3GeneratedAVMaskedContext"]
    SeparateAV = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]

    old_video = _unpack_node_result(_execute_comfy_node(SeparateAV, av_latent=old_stage1_latent))[0]
    samples = old_video["samples"]
    width, height = int(samples.shape[-1]) * 16, int(samples.shape[-2]) * 16
    positive, latent = _unpack_node_result(_execute_comfy_node(
        MiniMaxH3ImageToVideo, clip=clip, vae=vae, prompt=prompt,
        width=width, height=height, length=int(frame_count),
        first_frame=first_frame, last_frame=last_frame))[:2]

    carry_n = align_frame_count(min(int(carry_length), int(carry_images.shape[0])))
    tail_pixels = _resize(carry_images[-carry_n:], width, height, "disabled")
    tail_video_latent = _unpack_node_result(_execute_comfy_node(VAEEncode, pixels=tail_pixels, vae=vae))[0]
    source_latent = {"samples": tail_video_latent["samples"]}
    if carry_audio is not None and carry_audio["waveform"].shape[-1] > 0:
        carry_sr = carry_audio["sample_rate"]
        carry_samples = min(int(round(carry_n / 24.0 * carry_sr)), carry_audio["waveform"].shape[-1])
        tail_waveform = carry_audio["waveform"][..., -carry_samples:]
        tail_audio_latent = _unpack_node_result(_execute_comfy_node(
            VAEEncodeAudio, audio={"waveform": tail_waveform, "sample_rate": carry_sr}, vae=audio_vae))[0]
        source_latent = {"samples": (tail_video_latent["samples"], tail_audio_latent["samples"])}
    latent = _unpack_node_result(_execute_comfy_node(
        MaskedContext, latent=latent, source_latent=source_latent,
        context_length=carry_n, audio_feather_ticks=8))[0]

    guider = _unpack_node_result(_execute_comfy_node(BasicGuider, model=model, conditioning=positive))[0]
    full_sigmas = _unpack_node_result(_execute_comfy_node(
        BasicScheduler, model=model, scheduler=scheduler, steps=steps, denoise=1.0))[0]
    split_step = max(1, min(int(first_pass_steps), int(steps) - 1))
    high_sigmas = _unpack_node_result(_execute_comfy_node(
        SplitSigmas, sigmas=full_sigmas, step=split_step))[0]
    sampler = _unpack_node_result(_execute_comfy_node(KSamplerSelect, sampler_name=sampler_name))[0]
    noise = _unpack_node_result(_execute_comfy_node(RandomNoise, noise_seed=int(seed)))[0]
    pass1_raw, pass1_denoised = _unpack_node_result(_execute_comfy_node(
        SamplerCustomAdvanced, noise=noise, guider=guider, sampler=sampler,
        sigmas=high_sigmas, latent_image=latent))[:2]
    raw_audio = _unpack_node_result(_execute_comfy_node(SeparateAV, av_latent=pass1_raw))[1]
    rebuilt = dict(pass1_denoised)
    rebuilt["_muse_two_stage_raw_audio"] = raw_audio
    rebuilt["_muse_seed_used"] = int(seed)
    rebuilt["_muse_first_pass_steps_used"] = int(split_step)
    log.info("[MuseMinimaxRefineV14] sequential Stage-1 rebuild: %d frames, seed=%d, steps=%d",
             int(frame_count), int(seed), int(split_step))
    return rebuilt

def _refine_one_chunk_beta(
    model, clip, vae, audio_vae, chunk_prompt, chunk_latent,
    ref_image_size, seed, steps, two_stage_first_pass_steps,
    sampler_name, scheduler, two_stage_latent_upscale_model, two_stage_target_megapixels,
    ref_images_dict, ref_audios_dict, first_frame, last_frame, frame_count, guide_frames,
    carry_images, carry_audio, carry_length,
    raw_latent_carry_test, carry_context_latent,
    log_label,
):
    """Same overall shape as V1.3's own _refine_one_chunk (upscale, priming pass,
    recombine, final DisableNoise pass, decode, trim) — two things changed to match
    what the Beta Director actually does: the upscale step, and the continuity carry.
    """
    from nodes import NODE_CLASS_MAPPINGS
    CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]
    BasicGuider = NODE_CLASS_MAPPINGS["BasicGuider"]
    KSamplerSelect = NODE_CLASS_MAPPINGS["KSamplerSelect"]
    BasicScheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]
    SamplerCustomAdvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]
    SplitSigmas = NODE_CLASS_MAPPINGS["SplitSigmas"]
    DisableNoise = NODE_CLASS_MAPPINGS["DisableNoise"]
    LTXVSeparateAVLatent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]
    LTXVConcatAVLatent = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]
    VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]
    VAEDecodeAudio = NODE_CLASS_MAPPINGS["VAEDecodeAudio"]
    VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]
    VAEEncodeAudio = NODE_CLASS_MAPPINGS["VAEEncodeAudio"]
    MiniMaxH3GeneratedAVMaskedContext = NODE_CLASS_MAPPINGS.get("MiniMaxH3GeneratedAVMaskedContext")
    MinimaxH3LatentUpscaler3D = NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")
    if MinimaxH3LatentUpscaler3D is None:
        raise RuntimeError("[MuseMinimaxRefineV14] 'MinimaxH3LatentUpscaler3D' isn't registered — install "
                            "Comfyui_Minimax_h3_latent_Upscaler into custom_nodes.")

    video_for_upscale, audio_carry = _unpack_node_result(_execute_comfy_node(
        LTXVSeparateAVLatent, av_latent=chunk_latent,
    ))[:2]

    # Same pass1_raw-vs-pass1_denoised note as V1.3 — the saved chunk latent only
    # ever carries pass1_denoised's audio half; pass1_raw's is stashed separately
    # by the Director when present, and is the correct source.
    if isinstance(chunk_latent, dict) and "_muse_two_stage_raw_audio" in chunk_latent:
        audio_carry = chunk_latent["_muse_two_stage_raw_audio"]

    video_samples = video_for_upscale["samples"]
    cur_h_latent, cur_w_latent = video_samples.shape[-2], video_samples.shape[-1]
    width, height = cur_w_latent * 16, cur_h_latent * 16

    log.info("[MuseMinimaxRefineV14] %s: source %dx%d, seed=%d, steps=%d (first-pass=%d), keyframes=%s",
              log_label, width, height, seed, steps, two_stage_first_pass_steps,
              "first+last" if (first_frame is not None and last_frame is not None)
              else "first" if first_frame is not None else "last" if last_frame is not None else "none")

    sampler = _unpack_node_result(_execute_comfy_node(KSamplerSelect, sampler_name=sampler_name))[0]

    if ref_images_dict or ref_audios_dict:
        positive = _unpack_node_result(_execute_comfy_node(
            MiniMaxH3ReferenceToVideo, clip=clip, vae=vae, audio_vae=audio_vae, prompt=chunk_prompt,
            width=width, height=height, length=video_samples.shape[2], ref_image_size=ref_image_size,
            ref_images=ref_images_dict, ref_audios=ref_audios_dict,
        ))[0]
    else:
        positive = _unpack_node_result(_execute_comfy_node(CLIPTextEncode, clip=clip, text=chunk_prompt))[0]

    # MinimaxH3LatentUpscaler3D — the actual swap. Same call shape as the Beta
    # Director's own two-stage upscale step (force_unload=True, fp16, CUDA), so the
    # upscale a refined candidate gets here matches what it would have gotten had
    # two_stage_seed_hunt_latent_only been off during scouting.
    upscaled_result = _unpack_node_result(_execute_comfy_node(
        MinimaxH3LatentUpscaler3D,
        latent={"samples": video_samples},
        model_name=two_stage_latent_upscale_model,
        mode={"mode": "megapixels", "megapixels": float(two_stage_target_megapixels)},
        align=CANVAS_MULTIPLE,
        enable_temporal_chunking=True,
        force_unload=True,
        device="cuda",
        precision="fp16",
    ))[0]
    upscaled_samples = upscaled_result["samples"]
    tgt_h, tgt_w = upscaled_samples.shape[-2], upscaled_samples.shape[-1]
    eff_x = tgt_w / cur_w_latent if cur_w_latent else 0.0
    eff_y = tgt_h / cur_h_latent if cur_h_latent else 0.0

    # Keyframe lock — resize/re-encode the ORIGINAL keyframe images fresh at THIS
    # pass's own (upscaled) resolution, matching Director's own Stage 2 fix.
    positive = _rebuild_keyframe_conditioning(positive, vae, first_frame, last_frame, tgt_w * 16, tgt_h * 16, frame_count, guide_frames)

    # minimax_refs (reference images) get the same Stage-2 resolution treatment —
    # see _rebuild_refs_conditioning_for_stage2's own docstring for why this
    # matters. ref_images_dict here is the same dict that built `positive` above,
    # so this is a fresh re-encode in the normal case, not a fallback.
    positive = _rebuild_refs_conditioning_for_stage2(
        positive, vae, tgt_w, tgt_h, ref_image_size, ref_images_dict, log_label=log_label,
    )

    guider = _unpack_node_result(_execute_comfy_node(BasicGuider, model=model, conditioning=positive))[0]
    full_sigmas = _unpack_node_result(_execute_comfy_node(
        BasicScheduler, model=model, scheduler=scheduler, steps=steps, denoise=1.0,
    ))[0]
    split_step = max(1, min(int(two_stage_first_pass_steps), steps - 1))
    _high_sigmas, low_sigmas = _unpack_node_result(_execute_comfy_node(
        SplitSigmas, sigmas=full_sigmas, step=split_step,
    ))[:2]

    upscaled_video = dict(video_for_upscale)
    upscaled_video["samples"] = upscaled_samples
    upscaled_video["noise_mask"] = torch.ones_like(upscaled_samples)
    log.info(
        "[MuseMinimaxRefineV14] %s upscale (MinimaxH3LatentUpscaler3D): latent %dx%d -> %dx%d "
        "(requested %.2f MP, effective %.3fx/%.3fx)",
        log_label, cur_w_latent, cur_h_latent, tgt_w, tgt_h, float(two_stage_target_megapixels), eff_x, eff_y,
    )

    noise1 = _unpack_node_result(_execute_comfy_node(
        NODE_CLASS_MAPPINGS["RandomNoise"], noise_seed=seed,
    ))[0]
    tiny_sigmas = _unpack_node_result(_execute_comfy_node(
        SplitSigmas, sigmas=low_sigmas, step=0,
    ))[0]
    video_primed = _unpack_node_result(_execute_comfy_node(
        SamplerCustomAdvanced, noise=noise1, guider=guider, sampler=sampler,
        sigmas=tiny_sigmas, latent_image=upscaled_video,
    ))[0]

    recombined = _unpack_node_result(_execute_comfy_node(
        LTXVConcatAVLatent, video_latent=video_primed, audio_latent=audio_carry,
    ))[0]

    carry_trim_frames = 0
    if raw_latent_carry_test and carry_context_latent is not None:
        # Genuinely freezes recombined's own opening latent using the PREVIOUS
        # refined chunk's own raw final sampled latent (carry_context_latent — this
        # refine pass's own equivalent of Director Beta's prev_chunk_final_context_latent),
        # no VAE decode/re-encode round trip. Same node V1.3's own carry already used,
        # fed the raw latent instead of a lossy pixel re-encode — see this node's own
        # module docstring for why this replaces (not layers onto) Auto-Chain.
        if MiniMaxH3GeneratedAVMaskedContext is None:
            log.warning("[MuseMinimaxRefineV14] %s: raw_latent_carry_test is on but "
                        "'MiniMaxH3GeneratedAVMaskedContext' isn't registered — no continuity carry applied. "
                        "Install ComfyUI-H3-Motion-Context-MultiRef into custom_nodes.", log_label)
        else:
            recombined, carry_trim_frames_out = _unpack_node_result(_execute_comfy_node(
                MiniMaxH3GeneratedAVMaskedContext,
                latent=recombined, source_latent={"samples": carry_context_latent["samples"]},
                context_length=int(carry_length), audio_feather_ticks=8,
            ))[:2]
            carry_trim_frames = int(carry_trim_frames_out)
            log.info("[MuseMinimaxRefineV14] %s raw-latent carry: requested %d frames, trim=%d frames",
                      log_label, int(carry_length), carry_trim_frames)
    elif carry_images is not None and carry_images.shape[0] > 0 and MiniMaxH3GeneratedAVMaskedContext is not None:
        # Fallback — V1.3's original always-on pixel-VAE carry, unchanged. Used
        # whenever raw_latent_carry_test is off, same "off behaves as before"
        # convention the Beta Director's own toggles follow.
        carry_n = align_frame_count(min(int(carry_length), int(carry_images.shape[0])))
        tail_pixels = carry_images[-carry_n:]
        tail_video_latent = _unpack_node_result(_execute_comfy_node(
            VAEEncode, pixels=tail_pixels, vae=vae,
        ))[0]
        source_latent = {"samples": tail_video_latent["samples"]}
        if carry_audio is not None and carry_audio["waveform"].shape[-1] > 0:
            carry_sr = carry_audio["sample_rate"]
            carry_samples = min(
                int(round(carry_n / 24.0 * carry_sr)), carry_audio["waveform"].shape[-1],
            )
            tail_waveform = carry_audio["waveform"][..., -carry_samples:]
            tail_audio_latent = _unpack_node_result(_execute_comfy_node(
                VAEEncodeAudio, audio={"waveform": tail_waveform, "sample_rate": carry_sr}, vae=audio_vae,
            ))[0]
            source_latent = {"samples": (tail_video_latent["samples"], tail_audio_latent["samples"])}
        recombined, carry_trim_frames_out = _unpack_node_result(_execute_comfy_node(
            MiniMaxH3GeneratedAVMaskedContext,
            latent=recombined, source_latent=source_latent,
            context_length=carry_n, audio_feather_ticks=8,
        ))[:2]
        carry_trim_frames = int(carry_trim_frames_out)
        log.info("[MuseMinimaxRefineV14] %s pixel-VAE carry (raw_latent_carry_test off): %d frames "
                  "re-encoded from the previous refined chunk, trim=%d frames",
                  log_label, carry_n, carry_trim_frames)
    elif carry_images is not None and MiniMaxH3GeneratedAVMaskedContext is None:
        log.warning("[MuseMinimaxRefineV14] %s: no continuity carry applied — the 'H3 Generated AV Masked "
                    "Context' custom node (ComfyUI-H3-Motion-Context-MultiRef) isn't installed. This chunk's "
                    "seam may not match the rest of the video.", log_label)

    noise2 = _unpack_node_result(_execute_comfy_node(DisableNoise))[0]
    sampled = _unpack_node_result(_execute_comfy_node(
        SamplerCustomAdvanced, noise=noise2, guider=guider, sampler=sampler,
        sigmas=low_sigmas, latent_image=recombined,
    ))[0]

    refined_images = _unpack_node_result(_execute_comfy_node(VAEDecode, samples=sampled, vae=vae))[0]
    refined_audio = _unpack_node_result(_execute_comfy_node(VAEDecodeAudio, samples=sampled, vae=audio_vae))[0]

    if carry_trim_frames > 0 and refined_images.shape[0] > carry_trim_frames:
        refined_images = refined_images[carry_trim_frames:]
        waveform = refined_audio["waveform"]
        audio_trim_samples = round(carry_trim_frames / 24.0 * refined_audio["sample_rate"])
        audio_trim_samples = min(audio_trim_samples, waveform.shape[-1] - 1)
        if audio_trim_samples > 0:
            waveform = waveform[..., audio_trim_samples:]
        refined_audio = {"waveform": waveform, "sample_rate": refined_audio["sample_rate"]}

    return refined_images, refined_audio, sampled


class MuseMinimaxRefineV14:
    """Finishes a Seed Hunt candidate scouted by MuseMinimaxDirectorV1_2TwoStageBeta
    (two_stage_seed_hunt_latent_only=True) at full resolution, using the SAME upscale
    method (MinimaxH3LatentUpscaler3D) and the SAME stronger continuity mechanism
    (raw_latent_carry_test) that node's own non-latent-only finishing pass uses —
    see this file's own module docstring for the full reasoning. model/first_frame/
    last_frame are all optional — auto-picked up from whatever the Beta Director
    embedded on the chosen candidate_N_latent, same convention V1.3 already uses.
    ref_audio_1/2/3 are NOT auto-embedded (unlike a single keyframe pair, up to
    three distinct audio clips can't ride cleanly inside one latent dict) — wire
    these explicitly from the Beta Director's own ref_audio_1_used/2_used/3_used
    outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Wire this from the Beta Director's compiled_prompt output. Ignored for a "
                               "multi-chunk scouting candidate — each chunk already carries its own saved "
                               "prompt, used instead."}),
                "candidate": ("INT", {"default": 0, "min": 0, "max": 4,
                    "tooltip": "Which of the four candidate slots to continue. Set by the button selector in "
                               "the node's UI. Defaults to 0 (none picked yet) — the node deliberately refuses "
                               "to run at 0."}),
                "ref_image_size": (["match", "max"], {"default": "match", "tooltip":
                    "Only used when ref_images is connected. 'match' scales references down to the output's "
                    "pixel area (faster). 'max' keeps up to a 2048px short edge for stronger identity "
                    "fidelity, but reference tokens ride every sampling step so it's several times slower."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip":
                    "For a real Seed Hunt candidate, this is only a fallback — the actual seed it was "
                    "scouted with travels embedded in the candidate latent itself (_muse_seed_used) and is "
                    "used automatically, so this widget's value is ignored whenever that's present. It only "
                    "matters if the embedded value is missing (an older/incompatible latent, or ComfyUI's "
                    "own temp folder was cleared) — in that case, set it to match the seed the chosen "
                    "candidate was actually generated with."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100, "tooltip":
                    "Must match the TOTAL steps the candidate's own Stage 1 was generated with — unlike "
                    "seed and First-Pass Steps, this one is NOT embedded in the candidate latent, so it "
                    "always needs to be set by hand."}),
                "two_stage_first_pass_steps": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1, "tooltip":
                    "For a real Seed Hunt candidate, this is only a fallback — the actual First-Pass Steps "
                    "value it was scouted with travels embedded in the candidate latent itself "
                    "(_muse_first_pass_steps_used) and is used automatically, so this widget's value is "
                    "ignored whenever that's present. It only matters if the embedded value is missing (an "
                    "older/incompatible latent, or ComfyUI's own temp folder was cleared) — in that case, "
                    "set it to match the First-Pass Steps the candidate's own Stage 1 actually used. Cap "
                    "raised from 6 to 50 (2026-09-02) to match the Director's own widget."}),
                "sampler_name": (list(comfy.samplers.KSampler.SAMPLERS), {"default": "euler"}),
                "scheduler": (list(dict.fromkeys(
                    ["simple", "normal", "beta", "sgm_uniform"]
                    + list(comfy.samplers.KSampler.SCHEDULERS)
                )), {"default": "beta"}),
                "two_stage_latent_upscale_model": (_scan_latent_upscale_models(), {"tooltip":
                    "Which trained latent-upscale checkpoint to use (from "
                    "ComfyUI/models/latent_upscale_models/) — same model family the Beta Director's own "
                    "two-stage upscale uses. Real learned network, not interpolation."}),
                "two_stage_target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 2.0, "step": 0.01, "tooltip":
                    "Target resolution for the upscale, in megapixels — matches the upscaler node's own "
                    "'megapixels' sizing mode (aspect ratio preserved, pixel-aligned to 32)."}),
                "raw_latent_carry_test": ("BOOLEAN", {"default": False, "tooltip":
                    "For multi-chunk candidates only. Genuinely freezes each continuation chunk's own opening "
                    "latent using the PREVIOUS refined chunk's raw final sampled latent (no VAE round trip) — "
                    "the same mechanism the Beta Director's own raw_latent_carry_test uses, confirmed via a "
                    "real render + frame dump to be the one that actually constrains room/prop geometry across "
                    "a chunk boundary. Off falls back to this node's original always-on pixel-VAE-reencoded "
                    "carry (weaker, but needs nothing extra installed beyond ComfyUI-H3-Motion-Context-MultiRef, "
                    "same as either mode)."}),
                "timeline_data": ("STRING", {"default": "{}", "multiline": False}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip":
                    "Leave unconnected to auto-use the model the Beta Director embedded on the chosen "
                    "candidate (_muse_model_used)."}),
                "candidate_1_latent": ("LATENT",),
                "candidate_2_latent": ("LATENT",),
                "candidate_3_latent": ("LATENT",),
                "candidate_4_latent": ("LATENT",),
                "ref_images": ("IMAGE", {"tooltip":
                    "The same reference photos the original candidate used — without these, fine detail "
                    "(exact props, skin, likeness) that was only ever anchored by them may drift."}),
                "first_frame": ("IMAGE", {"tooltip":
                    "Leave unconnected to auto-use the first-frame keyframe embedded on the chosen candidate "
                    "(First/Last Frame and Hybrid modes only)."}),
                "last_frame": ("IMAGE", {"tooltip": "Same as first_frame, for the last-frame keyframe."}),
                "ref_audio_1": ("AUDIO", {"tooltip":
                    "The same reference audio clip(s) that anchored voice in the original candidate — wire "
                    "from the Beta Director's ref_audio_1_used/2_used/3_used outputs."}),
                "ref_audio_2": ("AUDIO", {"tooltip": "Same as ref_audio_1, for Ref Audio 2."}),
                "ref_audio_3": ("AUDIO", {"tooltip": "Same as ref_audio_1, for Ref Audio 3."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "execute"
    CATEGORY = "Muse Collective"

    def execute(self, clip, vae, audio_vae, prompt, candidate,
                ref_image_size, seed, steps, two_stage_first_pass_steps,
                sampler_name, scheduler, two_stage_latent_upscale_model, two_stage_target_megapixels,
                raw_latent_carry_test, timeline_data,
                model=None, candidate_1_latent=None, candidate_2_latent=None,
                candidate_3_latent=None, candidate_4_latent=None,
                ref_images=None, first_frame=None, last_frame=None,
                ref_audio_1=None, ref_audio_2=None, ref_audio_3=None):
        candidates = {
            1: candidate_1_latent, 2: candidate_2_latent,
            3: candidate_3_latent, 4: candidate_4_latent,
        }
        if candidate == 0:
            log.warning("[MuseMinimaxRefineV14] No candidate selected (candidate=0) — click one of the four "
                        "buttons in the node's UI to pick which candidate to continue. Blocking, not running.")
            blocker = ExecutionBlocker(None)
            return (blocker, blocker)
        chosen_latent = candidates.get(candidate)
        not_generated = isinstance(chosen_latent, dict) and chosen_latent.get("_muse_candidate_not_generated")
        if chosen_latent is None or not_generated:
            log.warning("[MuseMinimaxRefineV14] Candidate slot %d has no latent connected — wire "
                        "candidate_%d_latent, or pick a filled slot. Blocking, not running.",
                        candidate, candidate)
            blocker = ExecutionBlocker(None)
            return (blocker, blocker)

        embedded = chosen_latent if isinstance(chosen_latent, dict) else {}
        resolved_model = embedded.get("_muse_model_used") or model
        if resolved_model is None:
            log.warning("[MuseMinimaxRefineV14] No model connected, and candidate %d has none embedded either "
                        "— wire the correct checkpoint into 'model' manually. Blocking, not running.", candidate)
            blocker = ExecutionBlocker(None)
            return (blocker, blocker)

        ref_images_dict = None
        if ref_images is not None and ref_images.shape[0] > 0:
            ref_images_dict = {f"ref_image_{i}": ref_images[i:i + 1] for i in range(ref_images.shape[0])}
        else:
            log.warning("[MuseMinimaxRefineV14] No ref_images connected — continuing from text/keyframes only. "
                        "Wire in the same reference photos the candidate used.")

        ref_audios_dict = None
        _ref_audio_slots = [a for a in (ref_audio_1, ref_audio_2, ref_audio_3) if a is not None]
        if _ref_audio_slots:
            ref_audios_dict = {f"ref_audio_{i}": a for i, a in enumerate(_ref_audio_slots)}
        elif ref_images_dict:
            log.warning("[MuseMinimaxRefineV14] No ref_audio connected — if the original candidate used voice "
                        "cloning, this refine pass has nothing telling it what voice to keep.")

        bundle = chosen_latent.get("_muse_scout_bundle") if isinstance(chosen_latent, dict) else None
        if not bundle:
            log.warning("[MuseMinimaxRefineV14] Candidate %d has no saved chunk bundle — this node only "
                        "finishes Latent-Only Seed Hunt candidates (two_stage_seed_hunt_latent_only=True on "
                        "the Beta Director). Blocking, not running.", candidate)
            blocker = ExecutionBlocker(None)
            return (blocker, blocker)

        bundle_dir = bundle.get("dir")
        chunk_count = int(bundle.get("chunk_count") or 0)
        carry_length = int(bundle.get("carry_length") or 39)
        if not bundle_dir or not os.path.isdir(bundle_dir) or chunk_count < 1:
            log.warning("[MuseMinimaxRefineV14] Candidate %d's saved chunk bundle is missing or empty (%s) — "
                        "it may already have been cleaned up by an earlier Refine run on this candidate, or "
                        "ComfyUI's own temp folder was cleared. Re-run Seed Hunt scouting to generate a fresh "
                        "one. Blocking, not running.", candidate, bundle_dir)
            blocker = ExecutionBlocker(None)
            return (blocker, blocker)

        all_images = []
        all_waveform = []
        audio_sample_rate = None
        carry_images = None
        carry_audio = None
        carry_context_latent = None
        for chunk_idx in range(chunk_count):
            chunk_path = os.path.join(bundle_dir, f"chunk_{chunk_idx + 1:04d}.pt")
            if not os.path.isfile(chunk_path):
                log.warning("[MuseMinimaxRefineV14] Chunk %d/%d is missing from candidate %d's saved bundle "
                            "(%s) — stopping here rather than silently returning a partial video.",
                            chunk_idx + 1, chunk_count, candidate, chunk_path)
                blocker = ExecutionBlocker(None)
                return (blocker, blocker)
            saved = _load_scout_chunk(chunk_path)
            saved_latent = saved["latent"]
            # New V1.2B candidates carry their real Seed Hunt seed and Stage-1 split
            # on the latent itself; widget values remain fallbacks for older candidates.
            latent_meta = saved_latent if isinstance(saved_latent, dict) else {}
            resolved_seed = int(latent_meta.get("_muse_seed_used", embedded.get("_muse_seed_used", seed)))
            resolved_first_pass_steps = int(latent_meta.get(
                "_muse_first_pass_steps_used",
                embedded.get("_muse_first_pass_steps_used", two_stage_first_pass_steps),
            ))
            # Match the normal sequential two-stage path: once a refined predecessor
            # exists, its actual high-resolution final frame is the continuation
            # chunk's Stage-2 first-frame anchor. The scout bundle's saved first_frame
            # came from the low-resolution Stage-1 preview and can disagree subtly
            # with the refined carry, causing geometry/exposure to reform at the seam.
            chunk_first = (
                carry_images[-1:]
                if chunk_idx > 0 and carry_images is not None and carry_images.shape[0] > 0
                else (first_frame if first_frame is not None else saved.get("first_frame"))
            )
            chunk_last = last_frame if last_frame is not None else saved.get("last_frame")
            chunk_frame_count = saved.get("frame_count")
            chunk_guides = saved.get("guide_frames") or []
            chunk_ref_images = saved.get("ref_images") or ref_images_dict
            # Sequential rebuild currently applies to pure First/Last-Frame chunks.
            # Hybrid/reference chunks retain their saved multimodal Stage-1 conditioning.
            if (chunk_idx > 0 and carry_images is not None and chunk_first is not None
                    and not chunk_guides and not chunk_ref_images):
                saved_latent = _rebuild_stage1_continuation(
                    resolved_model, clip, vae, audio_vae, saved["prompt"], saved_latent,
                    chunk_first, chunk_last, chunk_frame_count, resolved_seed, steps,
                    resolved_first_pass_steps, sampler_name, scheduler,
                    carry_images, carry_audio, carry_length,
                )
            chunk_images, chunk_audio, chunk_sampled = _refine_one_chunk_beta(
                resolved_model, clip, vae, audio_vae, saved["prompt"], saved_latent,
                ref_image_size, resolved_seed, steps, resolved_first_pass_steps,
                sampler_name, scheduler, two_stage_latent_upscale_model, two_stage_target_megapixels,
                chunk_ref_images, ref_audios_dict, chunk_first, chunk_last, chunk_frame_count, chunk_guides,
                carry_images, carry_audio, carry_length,
                False, carry_context_latent,  # V1.2B-matched pixel/VAE carry; legacy raw toggle ignored.
                log_label=f"candidate={candidate} chunk={chunk_idx + 1}/{chunk_count}",
            )
            all_images.append(chunk_images)
            all_waveform.append(chunk_audio["waveform"])
            audio_sample_rate = chunk_audio["sample_rate"]
            carry_images = chunk_images
            carry_audio = chunk_audio
            carry_context_latent = chunk_sampled

        images = torch.cat(all_images, dim=0)
        waveform = torch.cat(all_waveform, dim=-1)
        audio = {"waveform": waveform, "sample_rate": audio_sample_rate}
        return (images, audio)


NODE_CLASS_MAPPINGS = {"MuseMinimaxRefineV14": MuseMinimaxRefineV14}
NODE_DISPLAY_NAME_MAPPINGS = {"MuseMinimaxRefineV14": "Muse Minimax Refine V1.4 (Beta-matched)"}
