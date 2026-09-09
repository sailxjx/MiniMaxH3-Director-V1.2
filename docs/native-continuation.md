# Persistent scouting and native suffix continuation

This fork carries the production continuation adapters on top of upstream
`022624d`. The upstream sentence-level voice attribution and saved-total-step
sigma schedule fixes remain intact. No ComfyUI core files are included here.

## Entry points

| Need | Node / input | Result |
| --- | --- | --- |
| Persist scouting | Director, `two_stage_seed_hunt_latent_only` | One saved Stage1 AV payload per group, plus a resume manifest |
| Load a reviewed candidate later | `MuseStage1ScoutBundleLoad` | `candidate_latent` and mixed-resolution `reference_image_set` |
| Re-sample a suffix | Director, `stage1_resume_manifest` | Verified prefix copied into a new bundle; complete suffix sampled |
| Keep reviewed Stage1 states for Refine | Refine V2, `preserve_stage1_latents` | Skips the continuation Stage1 rebuild |
| Route voices separately per group | Refine V2, `group_audio_slots` | Explicit one-based wired audio slots per group |
| Save actual Refine AV tensors | Refine V2, `refine_latent_directory` | Exclusive-write CPU tensor checkpoints and SHA-256 records |

These are advanced backend/API controls. The timeline editor has no dedicated
controls for `chunk_frames`, `seed_override`, or `disable_previous_audio` yet.
Keep the authored JSON externally when using them; editing/rebuilding the
timeline in the visual editor may replace those fields.

## Dependencies and compatibility

- Use the normal upstream H3/ComfyUI dependencies described in the README.
- Raw carry requires `MiniMaxH3GeneratedAVMaskedContext` from
  `ComfyUI-H3-Motion-Context-MultiRef`, including its H3 mask support.
- Explicit Hybrid media groups additionally require the registered
  `MiniMaxH3AudioConditioningT8` node from `ComfyUI-MiniMax-H3-Extend`.
  A missing dependency raises an error; no alternative conditioning is substituted.
- A connected Refine model takes precedence, so graph-applied LoRAs survive.
  With no model connected, the checkpoint/embedded model fallback remains.
- New Refine inputs are optional. Empty `group_audio_slots` keeps global wired
  voice routing, `preserve_stage1_latents=False` keeps Stage1 rebuilding, and an
  empty checkpoint directory performs no checkpoint writes.
- Explicit Hybrid groups now retain keyframes and independent media together.
  Legacy Hybrid groups previously dispatched through First/Last Frame; review
  those workflows before rendering with this fork.

## Scout, inspect, resume

1. Author `timeline_data.chunk_frames` as one integer frame count per group,
   each on H3's `17k+5` grid. Their sum must equal `duration_seconds * 24`.
   Group CUTs remain available at their corresponding weighted timestamps.
2. Generate a native Reference or Hybrid scout with two-stage sampling and
   latent-only scouting enabled. Persisted bundles live under
   `ComfyUI/output/latent/erase_tomorrow/muse_stage1_scout/`.
   The `erase_tomorrow` component is retained for existing bundle compatibility.
3. Preserve `candidate_N/chunk_0001.pt`, subsequent numbered payloads, and
   `native_resume.json` together. The manifest records the sampling contract,
   timeline, prompts' hashes, input reference hashes, latent hashes, exact
   decoded/trimmed/delivered frame counts, and whether each group was sampled.
4. Pass the **contents** of that manifest to `stage1_resume_manifest`.
   Use a one-based `render_chunk_start` after a nonempty prefix and set
   `render_chunk_end` to include the final group. Supply explicit per-group
   prompt overrides; the prefix prompt and timeline must match the original.
5. Keep global sampling settings and prefix references unchanged. Change only
   the suffix's duration, prompt, local references or `seed_override` as needed.
   A suffix seed override must be a JSON integer in `[0, 2**64)`; groups without
   one use the original candidate seed. Reusing a prefix containing a previous
   override validates that group's actual saved seed.

Resume supports explicit Reference/Hybrid group modes, with a Hybrid outer mode
required for Hybrid groups. It rejects Lip Sync, timed guides, Seed Hunt,
long-form mode, seam interpolation and alternate carry paths. All suffix groups
must be regenerated. Old bundles without frame/reference evidence need an
audited manifest before reuse; missing facts are not inferred.

Only the reused prefix's saved latent is decoded; its sampler is never invoked.
Copying it into the new job checks the bytes and refuses to overwrite a file.
Decode reproducibility and the new seam still need visual/audio review.

## Frame accounting

The first group uses the H3 `17k+5` frame grid. For raw-carry successors, the
delivered tail is the nearest positive multiple of 17 to the requested frame
count; sampling adds the protected carry length. Thus `sampled - protected =
delivered`. Extending a suffix by 17 requested frames extends its delivered
tail by 17 frames when the carry configuration is unchanged.

This corrects the upstream raw-carry frame budget and can change multi-group
output duration relative to upstream. Read the per-group log/manifest for the
exact result. No interpolation, exposure correction or additional frame removal
is performed by these adapters. Existing upstream media options are separate.

## A non-dialogue tail and later Refine

Set `disable_previous_audio=true` on a resumed **Reference** suffix group.
Director omits both the previous decoded-audio soft reference and its raw audio
latent carry. Visual carry is retained; the current group's audio latent is
fresh. This does not mute output or guarantee silence: the model can still
generate breathing or sound. Describe the intended non-dialogue action in the
group prompt and review the result.

For Refine, connect Bundle Load's latent to a candidate slot and select that
slot (Refine slots are 1-based; loader candidate indices are 0-based). Connect
`reference_image_set` to `ref_images_bundle`; it preserves each group's original
reference tensors and their individual shapes. Wire the intended model and
required audio sources explicitly.

Enable `preserve_stage1_latents` and raw carry. Supply `group_audio_slots` as a
JSON array containing one array of wired slot IDs (1–3) for each saved group.
An empty inner array supplies no voice reference for that group. Slots are
compacted in the declared order to match prompt audio numbering. Also pass the
matching timeline, including the tail's `disable_previous_audio` flag. The flag
requires explicit slot lists and is rejected for the opening group or Hybrid.

`refine_latent_directory` is an optional unique output-relative subdirectory.
It saves actual sampled video/audio tensors before decode, verifies readback
equality, and records shapes, hash and sampling metadata. It refuses file
overwrite and paths outside the output directory. These are tensor inspection
checkpoints, not a new Refine-resume API and not provider video files.

## Storage and trust

Scout caches persist across runs and can occupy substantial disk space.
Retention/cleanup is an explicit operator task; Refine does not delete sources.
Only load your own trusted local bundles: `torch.load(weights_only=False)` can
execute pickle payloads. Path and hash checks verify location/integrity, not
the trustworthiness of an arbitrary downloaded pickle. Do not upload real
production bundles, media, credentials or service configuration to this repo.

## Validation

Run `python -m unittest discover -s tests -v` from the repository root.
Tests cover manifest integrity and immutable prefixes, mixed seeds, per-group
audio policy, bundle shape/identity routing, gaps/path rejection, and execution
of isolated production AST blocks for native carry, Hybrid anchors and the two
upstream fixes. These CPU tests do not import the full ComfyUI/GPU stack.

Before deployment, test registration in your ComfyUI environment and run a
short native scout → suffix resume → preserved Refine workflow. Verify output
frames, AV tensors and the seam. This fork migration has not deployed or run a
new GPU render on the production servers.
