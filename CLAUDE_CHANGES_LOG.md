# Claude Changes Log — Muse Minimax Director V1.4 / Refine V2

## 2026-09-16 (later) — GPU-confirmed: previous_audio_tail and disable_previous_audio fix
## two DIFFERENT symptoms of the same bug; ship them together, not either alone

**Do not re-litigate this without reading this entry first — it corrects an assumption made
earlier the same day (below) after real GPU renders came back.**

Two real turbo-8-step renders exercised `previous_audio_tail` alone (no `disable_previous_audio`)
on the exact `b010_mod001` G03 and `b060a` G03 hazard cases from the entry below. User-verified
listening result:

- `b060a` G03: audio problem fully fixed.
- `b010_mod001` G03: the opening ~5s of audio was still wrong; everything after that was normal.

This is fully consistent with the mechanism, not a contradiction of it: `previous_audio_tail` adds
an explicit, correctly-labeled soft `ref_audio` reference ("previous shot's own score/ambience")
that the chunk can condition the REST of its generation on — this is exactly why the later portion
came out normal in both cases, where before (with neither fix) it hallucinated fully ungrounded,
often unintelligible audio for the whole remainder. But `previous_audio_tail` never touches the
hard freeze itself — `raw_latent_carry_test`'s `MiniMaxH3GeneratedAVMaskedContext` unconditionally
copies the previous chunk's real decoded audio into this chunk's opening ~39 frames (~1.6s)
regardless of this flag. That frozen opening is still the previous speaker's real, out-of-context
voice. Whether that reads as "broken" is content-dependent: `b060a`'s G2→G3 cut stays inside the
same continuous close-up domestic scene, so the frozen fragment plausibly passed as a natural
vocal trail-off; `b010_mod001`'s G2→G3 cut is a bigger scene/mood shift (an argument → a quiet
first-person watching shot), so the identical mechanism read as an obvious non-sequitur. Relying on
`previous_audio_tail` alone to fix this class of bug is therefore a content-dependent gamble, not a
fix — it happened to pass for one of the two real cases tested and not the other.

**Correct default going forward: turn both on together** for the same hazard condition (a chunk
with no audio references of its own, immediately following a chunk that had real spoken dialogue).
`disable_previous_audio` removes the frozen-opening bleed; `previous_audio_tail` keeps the rest of
the chunk grounded instead of unconditioned. Neither one closes both gaps alone. Implemented as the
paired smart default in `tools/build_muse_stage1_config.py` (erase-tomorrow repo) — see that file's
own dated comment block for the exact condition and citations.

GPU validation of the *combined* fix (both flags together) is still pending as of this entry —
only `previous_audio_tail` alone has been GPU-tested so far. Do not assume the combination is
GPU-confirmed until a render with both flags on has actually been listened to.

## 2026-09-16 — disable_previous_audio no longer requires a native-resume Reference suffix

**Root cause traced.** A trailing zero-`ref_audio` chunk immediately following a chunk with real
spoken dialogue was reproducibly inheriting the *previous* chunk's actual voice (correct timbre,
nonsense content) for its first ~39 frames, then generating fully ungrounded, unintelligible audio
for the remainder. Traced to `raw_latent_carry_test`'s joint audio+video latent hard-freeze
(`MiniMaxH3GeneratedAVMaskedContext`): with no per-chunk opt-out, every chunk after the first gets
the previous chunk's *actual decoded audio* frozen into its own opening frames regardless of
whether that chunk wants any audio continuity at all. `previous_audio_tail` (2026-09-15, above) does
not touch this — it only adds an explicit *soft* reference alongside the same unconditional freeze,
so it cannot prevent a live speaker's real voice from bleeding into a chunk that has nothing of its
own to say. This exact failure and its fix are already on record in this project's own accepted
history (`b010` C02 "choice hold" and `b060a`'s `production_native_stage1_v002`): both used
`disable_previous_audio` (video-only carry — freeze video, leave audio fully unconditioned on that
chunk) to keep a live-speech tail from bleeding into a silent hold, and it worked. But that path
only existed through the native Stage-1 resume runner (`run_muse_controlled_resume.py`); a plain,
one-shot multi-chunk submission (`run_muse_controlled_stage1.py`, this project's normal path) had no
way to reach it — its validation gate raised unless `_native_resume_plan is not None`.

**Fix.** `_video_only_carry_inject` never depended on native-resume state or on which mode the
current chunk's own positive conditioning uses — only on `chunk_idx > 0` and this chunk's own raw
carry-source latent, both present in a plain one-shot call. The gate now only checks
`chunk_idx == 0` (still meaningless there — no previous chunk to carry video from). Also fixed a
second, previously unconditional freeze: two-stage sampling's post-upscale re-freeze
(`prev_chunk_final_context_latent` re-injection, "Stage 2 raw-latent carry") always called
`MiniMaxH3GeneratedAVMaskedContext` regardless of `disable_previous_audio`, which would have quietly
reinstated the audio-inclusive freeze the Stage-1 injection had just opted out of, on any run with
`two_stage_sampling=True` (i.e. every 768p+ job this project runs). Both sites now dispatch the same
way. No mode restriction remains either — the mechanism is a pure tensor operation on the joint
latent, independent of Reference vs Hybrid conditioning; the old `generation_mode == "Reference"`
requirement only reflected the one context this had CPU tests for.

Files touched: `muse_minimax_director.py`, `tests/test_disable_previous_audio_outside_resume.py`
(new). All 57 existing + new CPU tests pass. GPU render validation pending.

## 2026-09-15 — Previous-audio tail appended last; explicit opt-in under prompt override

**1. Slot order.** A continuation chunk's previous-audio tail used to be inserted first, as
`ref_audio_0`, with the authored voices after it ("the three-slot audio pool does not [have room],
so the tail claims slot 0 to avoid being dropped"). That renumbered every authored `<Audio N>` by
one and reordered the references for no gain: neither the core Reference node
(`comfy_extras/nodes_minimax_h3.py` consumes `ref_audios` in dict order as equal reference blocks)
nor T8 (this node passes `prompt_primary_audio_ordinal=0`) treats slot 0 specially. The authored
voices now take slots 0.. in list order; the tail is appended after them when it applies and is
skipped with a log line when the three slots are already taken. `<Audio N>` is therefore N in every
group, and the tail is `<Audio k+1>`.

**2. Opt-in under an override.** `_resolve_carry_reference_injection` keeps both carriers off under
`use_prompt_override` by default and still rejects `carry_reference_injection.*: true` there. A new
`timeline_data.previous_audio_tail: true` re-enables only the audio tail under an override; because
the tail is now last, an overriding prompt can name it as `<Audio k+1>` and the override tag check
passes. `picture_anchor` stays off under an override: Hybrid groups already lock a continuation's
first frame to the predecessor's last decoded frame through T8 `first_frame` (`chunk_first =
prev_chunk_images[-1:]`), and the Reference-mode trailing still made later groups learn identity
from their own predecessor (see the comment above `use_generated_picture_anchor`).

Files touched: `muse_minimax_director.py`, `tests/test_slot_contract.py`.

## 2026-09-13 — Prompt-override reference slot contract; exact explicit raw-carry frame budget

**1. Explicit group lengths could not be delivered under raw carry.** `timeline_data.chunk_frames`
was validated as 17k+5 for every group, but a raw-carry continuation is sampled as request + carry
and loses the carry to the post-decode trim, so its deliverable length is always a multiple of 17.
No legal request survived; the frame plan silently snapped it (b060a 768p v001 asked 243 per
continuation group, received 238, 651 frames against a 661 declaration).
`_validate_explicit_chunk_frames` now checks request + carry for every group after the first
(raw carry applies there whatever `continuityFromPrev` says) and an explicit request is delivered
exactly. Seconds-based UI buckets keep the nearest representable run.

**2. Hidden reference slots under `use_prompt_override`.** Continuation chunks receive two
node-owned carriers: the generated predecessor still, appended after the group's own images as a
`<Picture N>`, and the previous chunk's audio tail, inserted at `ref_audio_0` ahead of the group's
voices (the image pool has room after a capped character list; the three-slot audio pool does not,
so the tail claims slot 0 to avoid being dropped). Both are declared only by the prompt this node
compiles itself. Upstream V1.0 had no prompt override; the override later replaced the text but not
the slots, so authored prompts met an unnamed still and voices shifted by one slot.
`_resolve_carry_reference_injection` turns both carriers off whenever an override is active and
rejects an explicit `true` for either; without an override the upstream default is unchanged.
Evidence: a same-seed seam canary with the carriers on versus off measured seam z 1.86 versus 2.56,
both passing. Comparison: AIMixer ComfyUI_MiniMaxH3_Director keeps `<Picture N>` equal to slot N and
carries continuity through motion-context latents and keyframes, never through reference slots; the
T8 conditioning node's `strict_prompt_tags` rejects a tag beyond the connected count.

**3. Tag check.** `_validate_override_media_tags` runs for Reference chunks under an override: a tag
beyond the connected count is fatal, and every connected slot must be named. Hybrid chunks keep the
T8 strict check.

**Behaviour note.** A Stage-1 resume or suffix rerun of a bundle generated with the carriers on now
samples its suffix without them when the request uses an override.

**Server history.** Unreleased edits hand-deployed to the pool on 2026-09-12 (director sha256 prefixes
09cec316, 9f98a7a2, 992253872ecc) are superseded by this commit. They included an unconditional
carrier removal that was later turned back into a switch, and a guard that demanded declarations for
node-owned slots. Their bytes are archived in comfyui-fleet
`archives/20260913-muse-unreleased-server-drift`. Files touched: `muse_minimax_director.py`,
`tests/test_slot_contract.py`, `tests/test_runtime_integration.py`.

## 2026-09-11 — Matched Stage-2 target range to the latent upscaler

The Director and Refine V2 wrapper schemas now expose the underlying
`MinimaxH3LatentUpscaler3D` target range through 16.0 megapixels instead of
stopping at 2.0 megapixels. This changes only the accepted input range; the
Stage-1 resolution controls and Stage-2 sampling implementation are unchanged.

## 2026-09-11 — Fixed multi-group 2K Refine GPU retention

After each group finishes, Refine V2 now keeps its decoded images, waveform, and any
next-group raw AV carry latent as detached CPU tensors. It releases the corresponding
GPU tensors before the following group starts. The sampled bytes and ordering remain
unchanged; the change prevents the previous full-resolution group from occupying GPU
memory while the next group's denoising activations are allocated.

## 2026-09-11 — Added exact Stage-1 resolution selection

`base_resolution` now accepts the project canvases `960x544` and `1344x768`. A fixed
selection bypasses the stock megapixel calculation, which rounds a 16:9 1 MP request to
`1376x768`. The default remains `auto`, so saved workflows retain the existing Aspect
Ratio + Megapixels + Multiple Of behavior. The new input is optional and appended after
the established inputs to preserve positional compatibility.

## 2026-09-08 00:39 BST — Fixed: sentence-level speaker controls weakly attached voice references after dialogue

**Reproduction:** two visible female character references, two distinct Voice Reference
audio clips, and one sentence assigned to each character compiled correct Audio-to-Subject
definitions, but the rendered voices crossed or ignored the supplied timbre. Replacing one
clip with an unmistakable male voice produced two female voices, ruling out a simple audio
tensor swap and showing that H3 was falling back to visual voice priors.

**Root cause:** the newer per-sentence `dialogueSpeakers` path compiled dialogue as
`<d>[Language] ...</d> (Sx)`. The official Ref2VA structure binds the visible subject and
global speaker before the vocal event: `<Subject N> (Sx) ... <d>...</d>`. The older
whole-CUT speaker path already did this; only the sentence-level path was malformed.

**Fix:** `_wrap_dialogue` now optionally accepts corresponding Subject IDs. In
Reference/Hybrid sentence-level compilation it emits
`<Subject N> (Sx): <d>[Language] ...</d>` for every assigned quoted sentence. First/Last
Frame mode has no Subject abstraction and retains its existing event-order `(Sx)` form.
Audio loading, dictionary order, retention modes, and Audio-to-Subject definitions were
not changed; inspection confirmed those were already ordered correctly.

**Verification:** Python syntax compilation plus focused dialogue-wrapper checks cover two
separate subjects, per-sentence attribution, and the unchanged no-Subject fallback. A full
render should be repeated with the same two-character/two-voice test after restarting
ComfyUI so the updated module is loaded.

## 2026-09-06 — Fixed: Refine V2 could continue a Seed Hunt candidate against the wrong sigma schedule, producing garbled audio at the Stage-2 seam

**Root cause, confirmed via a real reproducible test, not a guess:** `MuseMinimaxRefineV2`
rebuilds the full sigma schedule for its own Stage-2 continuation pass from its **own**
`steps` widget (`BasicScheduler(..., steps=steps)`), while only restoring the *split
point* (`_muse_first_pass_steps_used`) from the picked candidate's own embedded metadata.
If the Refine node's `steps` widget doesn't happen to match the total step count the
Director actually generated that candidate with, the "remaining schedule" Refine
computes is a genuinely different set of sigma values than what the Stage-1 latent's
real noise level corresponds to — a real numerical mismatch, not a quality-only issue.

**How this was actually diagnosed:** confirmed reproducible twice on the same seed
(Director `steps=10`, Refine `steps=8` — mismatched) as garbled audio right around the
Stage-2 continuation point; confirmed clean on the Director's own single-pass path
(no Refine involved, so no mismatch possible); confirmed clean on a separate
Combo V2 test purely because its own Director and Refine widgets both happened to be
set to the same value (8 and 8). Andy caught the actual mismatch by directly comparing
widget values side by side between two saved workflows — manually setting Refine's
`steps` to match the Director's (10) immediately fixed the same failing test. Several
earlier theories this session (temporal chunking, ref_audio wiring, V1.4-vs-Combo-V2
code differences) were investigated and ruled out along the way; none of them were it.

**The fix:** the Director now also embeds the real total step count on the candidate
latent (`_muse_steps_used`), the same pattern already used for `_muse_seed_used` and
`_muse_first_pass_steps_used`. Refine V2 now restores and uses that value
(`resolved_steps`) instead of its own local `steps` widget when rebuilding the
schedule for both the sequential multi-chunk continuation path and the main per-chunk
refine call — so it's correct automatically regardless of what its own widget says.

**Status: confirmed working end-to-end, `steps` widget now hidden.** First
confirmation attempt after a restart still showed the old behavior — turned out a
second, stale, non-git-tracked copy of the Director (`custom_nodes/
MiniMaxH3-Director-V1.4`, last touched 2026-09-03, missing several other fixes too)
was silently winning the `MuseMinimaxDirectorV1_4` class registration over this repo's
own copy — real folder name collision, not a code bug, and it meant every "Director"
run that whole evening was actually going through months-old unpatched code regardless
of what got edited here. Moved that stale folder out of `custom_nodes` (to
`F:\Custom Node Backup`) to remove the collision; after that, a real restart, and a
fresh Seed Hunt candidate, the Refine V2 log line correctly read `steps=10` (matching
the Director, not Refine's own mismatched `steps=8` widget), and the audio came out
clean. `steps` is now hidden on Refine V2's panel the same way `seed` and
`two_stage_first_pass_steps` already are (kept in the schema for saved-workflow
positional compatibility, just no longer user-editable) — same day, once confirmed.

**Also fixed on Combo V2** (`Muse-MiniMax-Director-Combo-V2/muse_minimax_director.py`,
its own changelog has the matching entry) — it embeds candidates via the identical
pattern (ported from V1.4 earlier the same day) and shares the same `MuseMinimaxRefineV2`
node, so it had the exact same latent bug; only reason it didn't show up in testing is
that Combo V2's own Director and Refine widgets happened to already match (8 and 8).

Files touched: `muse_minimax_director.py`, `muse_minimax_refine_v2.py`.

## 2026-09-06 (same day) — UI: Settings section always visible, no longer hidden behind a gear icon

**What changed:** the Director's timeline UI had a "📷 References" heading with a
tiny ⚙ gear button that toggled the Timeline save/load controls and Analyze Backend
settings hidden/shown — squeezed in between the heading and the actual reference
picture grid when open. Real user feedback (a YouTube comment) confirmed nobody
noticed the gear existed. Replaced with a permanently visible "⚙️ Settings" section,
placed *above* References entirely — so References is now immediately followed by
its own picture grid with nothing in between, and Settings can't be missed.

No data/widget changes — this is display-only, so it applies automatically to any
already-saved workflow the moment someone updates and reopens the node, nothing needs
to be rebuilt.

Files touched: `js/muse_minimax_director.js`.


## 2026-09-08 01:26 BST  Aligned the complete Voice Reference prompt chain with MiniMax official Ref2VA guidance

The remaining generated Voice Reference prose was audited against the official Ref2VA
guide. The compiler now uses the official retention meaning: the target speaker follows
each Audio reference's voice timbre and measured delivery without copying the original
signal. It also names every Audio-to-Subject voice relationship in the summary and cites
the matching Audio reference at the actual spoken event, producing an explicit chain from
Subject N to Sx to Audio N to dialogue.

Only standalone Ref Audio with Voice Reference retention receives this voice-timbre chain.
Fully copied lip-sync/song audio, partial and weak audio modes, carry audio, First/Last
Frame fallback behavior, and physical reference-audio ordering remain unchanged.

Verification: Python syntax compilation, git diff validation, and focused source checks
passed. A restarted ComfyUI render is still required for perceptual confirmation because
Ref2VA voice adherence is generative rather than deterministic.


## 2026-09-08 01:36 BST  Removed duplicated speech verb from Voice Reference vocal events

The first official-format pass inserted an automatic says before each dialogue even
though the user's CUT prose commonly already says says, replies, asks, or another speech
verb. This produced repetitive text such as says, Subject 1 ... says:.

The generated binding now ends after the matching Audio reference with a colon:
Subject N (Sx), using the voice timbre and measured delivery from Audio N: dialogue.
The user's own speech verb remains untouched. Subject, speaker, Audio, language, dialogue,
retention, and summary mappings are otherwise unchanged.

Verification: exact single replacement, Python syntax compilation, and git diff check.
