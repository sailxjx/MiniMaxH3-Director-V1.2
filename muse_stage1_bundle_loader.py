"""Load a persisted Muse Stage-1 scout bundle for a later sequential Refine pass."""

from __future__ import annotations

import glob
import logging
import os

import folder_paths
import torch


log = logging.getLogger(__name__)


def _under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == os.path.realpath(root)
    except ValueError:
        return False


class MuseStage1ScoutBundleLoad:
    """Turn one persisted multi-group scout bundle into Refine's LATENT plus IMAGE inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bundle_directory": ("STRING", {"default": "", "multiline": False}),
                "candidate_index": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
                "carry_length_frames": ("INT", {"default": 39, "min": 22, "max": 90, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "MUSE_REF_IMAGE_SET")
    RETURN_NAMES = ("candidate_latent", "reference_image_set")
    FUNCTION = "load"
    CATEGORY = "Muse Collective"

    def load(self, bundle_directory: str, candidate_index: int, carry_length_frames: int):
        root = os.path.realpath(bundle_directory)
        allowed_root = os.path.join(folder_paths.get_output_directory(), "latent", "erase_tomorrow", "muse_stage1_scout")
        if not _under(root, allowed_root):
            raise ValueError("Stage-1 bundle must be inside ComfyUI output/latent/erase_tomorrow/muse_stage1_scout")
        candidate_dir = os.path.join(root, f"candidate_{int(candidate_index)}")
        chunk_paths = sorted(glob.glob(os.path.join(candidate_dir, "chunk_*.pt")))
        if not chunk_paths:
            raise FileNotFoundError(f"No persisted Stage-1 chunks found at {candidate_dir}")
        expected_names = [f"chunk_{i + 1:04d}.pt" for i in range(len(chunk_paths))]
        if [os.path.basename(path) for path in chunk_paths] != expected_names:
            raise ValueError("Stage-1 chunks must be contiguous and start at chunk_0001.pt")
        reference_sets = []
        for path in chunk_paths:
            if not _under(path, allowed_root):
                raise ValueError("Stage-1 chunk resolves outside the scout output root")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(saved, dict) or not isinstance(saved.get("latent"), dict):
                raise ValueError(f"Invalid Stage-1 latent payload in {path}")
            raw_references = saved.get("ref_images")
            if raw_references is None:
                raw_references = {}
            if not isinstance(raw_references, dict):
                raise ValueError(f"Invalid Stage-1 reference image payload in {path}")
            references = {
                key: value for key, value in raw_references.items()
                if hasattr(value, "shape")
            }
            # Reference-mode chunks are allowed to rely entirely on the carried AV
            # state and their own text prompt.  Native prefix reuse deliberately
            # persists such chunks with ref_images={} so that a subject which has
            # left frame is not reintroduced during Stage-2.  Preserve the empty
            # slot in the per-chunk list; MuseMinimaxRefineV2 routes it through its
            # existing text-only conditioning path.
            if raw_references and not references:
                raise ValueError(f"Stage-1 reference images are unusable in {path}")
            reference_sets.append(references)
        if not isinstance(saved, dict) or not isinstance(saved.get("latent"), dict):
            raise ValueError(f"Invalid Stage-1 latent payload in {chunk_paths[-1]}")
        latent = dict(saved["latent"])
        latent["_muse_scout_bundle"] = {
            "dir": candidate_dir,
            "chunk_count": len(chunk_paths),
            "carry_length": int(carry_length_frames),
        }
        log.info("[MuseStage1ScoutBundleLoad] Loaded %d Stage-1 chunk(s) from %s", len(chunk_paths), candidate_dir)
        # Keep each Stage-1 reference tensor at its original shape.  ComfyUI's
        # IMAGE batch cannot hold the mixed native resolutions from the accepted
        # reference package; padding or resampling here would change Refine's
        # visual conditioning.  MuseMinimaxRefineV2 consumes this typed mapping
        # directly and forwards it to MiniMaxH3ReferenceToVideo.
        return (latent, {"__muse_per_chunk_ref_images__": reference_sets})


NODE_CLASS_MAPPINGS = {
    "MuseStage1ScoutBundleLoad": MuseStage1ScoutBundleLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseStage1ScoutBundleLoad": "Muse Stage-1 Scout Bundle Load",
}
