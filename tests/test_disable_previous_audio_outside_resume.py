"""CPU-only tests for disable_previous_audio outside the native-resume path.

Regression coverage for two changes made together:
  1. The validation gate on `disable_previous_audio` no longer requires
     `_native_resume_plan is not None` or `generation_mode == "Reference"` --
     only that chunk_idx > 0 (there is no previous chunk to carry video from
     on chunk 1). The underlying _video_only_carry_inject mechanism never
     depended on native-resume state or on which mode a chunk's own positive
     conditioning uses; the old gate only reflected the narrower scope this
     had been validated in (see the b010 C02 choice-hold fix, native-resume-
     only). This was widened after confirming the identical hazard (a
     trailing zero-ref_audio chunk inheriting a live speaker's real voice
     through the blind joint-AV freeze) reproduces in a plain, non-resumed
     multi-chunk stage1 call, in both Reference and Hybrid mode.
  2. The Stage 2 (two-stage sampling, post-upscale) raw-latent re-freeze now
     also respects disable_previous_audio. It previously always called
     MiniMaxH3GeneratedAVMaskedContext unconditionally, which would silently
     re-instate the joint-AV freeze (audio included) that the Stage-1
     injection had just opted out of.
"""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "muse_minimax_director.py").read_text(encoding="utf-8")
DIRECTOR = ast.parse(SOURCE)


def execute_node(node, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<production AST>", "exec"), namespace)


class ValidationGateTests(unittest.TestCase):
    def _gate(self):
        return next(
            n for n in ast.walk(DIRECTOR)
            if isinstance(n, ast.If) and "disable_previous_audio and chunk_idx == 0" in ast.unparse(n.test)
        )

    def test_raises_only_on_chunk_zero(self):
        gate = self._gate()
        for chunk_idx, disabled, should_raise in (
            (0, True, True),
            (0, False, False),
            (1, True, False),
            (2, True, False),
        ):
            with self.subTest(chunk_idx=chunk_idx, disabled=disabled):
                ns = dict(disable_previous_audio=disabled, chunk_idx=chunk_idx)
                if should_raise:
                    with self.assertRaises(ValueError):
                        execute_node(gate, ns)
                else:
                    execute_node(gate, ns)  # must not raise

    def test_no_longer_mentions_native_resume_or_reference_mode(self):
        gate = self._gate()
        unparsed = ast.unparse(gate)
        self.assertNotIn("_native_resume_plan", unparsed)
        self.assertNotIn("generation_mode", unparsed)


class _FakeSamples:
    """Stands in for the NestedTensor(video, audio) pair .unbind() normally returns."""
    def __init__(self, tag):
        self.tag = tag

    def unbind(self):
        return (f"{self.tag}_video", f"{self.tag}_audio")

    def __eq__(self, other):
        return isinstance(other, _FakeSamples) and self.tag == other.tag


class Stage1DispatchTests(unittest.TestCase):
    def test_stage1_injection_dispatches_on_the_flag(self):
        branch = next(
            n for n in ast.walk(DIRECTOR)
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "disable_previous_audio"
            and "chunk %d previous audio disabled" in ast.unparse(n.body)
        )
        for disabled in (True, False):
            calls = {"video_only": [], "masked_context": []}
            ns = dict(
                disable_previous_audio=disabled,
                raw_carry_stage1_source={"samples": _FakeSamples("stage1")},
                vae_reencode_carry_length=39,
                latent="fresh_latent",
                log=__import__("logging").getLogger("test"),
                chunk_idx=1,
                _video_only_carry_inject=lambda latent, video, carry_n: (calls["video_only"].append((latent, video, carry_n)) or ("carried", 39)),
                _execute_comfy_node=lambda *a, **kw: (calls["masked_context"].append(kw) or ("carried", 39)),
                _unpack_node_result=lambda x: x,
                MiniMaxH3GeneratedAVMaskedContext="mask_context_node",
                align_frame_count=lambda n: n,
                importlib=__import__("importlib"),
            )
            # The real branch imports its own helper module for
            # _require_h3_mask_support; stub it out since this test only
            # exercises the dispatch, not that helper's own behavior.
            branch_src = ast.unparse(branch)
            branch_src = branch_src.replace(
                "importlib.import_module(MiniMaxH3GeneratedAVMaskedContext.__module__)._require_h3_mask_support()",
                "None",
            )
            stub = ast.parse(branch_src).body[0]
            execute_node(stub, ns)
            if disabled:
                self.assertEqual(len(calls["video_only"]), 1)
                self.assertEqual(len(calls["masked_context"]), 0)
            else:
                self.assertEqual(len(calls["video_only"]), 0)
                self.assertEqual(len(calls["masked_context"]), 1)
                self.assertEqual(calls["masked_context"][0]["source_latent"], {"samples": _FakeSamples("stage1")})


class Stage2DispatchTests(unittest.TestCase):
    def test_stage2_refreeze_also_dispatches_on_the_flag(self):
        """The second (post-upscale) injection must not silently reinstate the
        audio-inclusive freeze once Stage 1 opted out of it."""
        branch = next(
            n for n in ast.walk(DIRECTOR)
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "disable_previous_audio"
            and "Stage 2 previous audio disabled" in ast.unparse(n.body)
        )
        for disabled in (True, False):
            calls = {"video_only": [], "masked_context": []}
            ns = dict(
                disable_previous_audio=disabled,
                prev_chunk_final_context_latent={"samples": _FakeSamples("stage2")},
                recombined="recombined_latent",
                vae_reencode_carry_length=39,
                chunk_idx=1,
                log=__import__("logging").getLogger("test"),
                _video_only_carry_inject=lambda latent, video, carry_n: (calls["video_only"].append((latent, video, carry_n)) or ("carried", 39)),
                _execute_comfy_node=lambda *a, **kw: (calls["masked_context"].append(kw) or ("carried", 39)),
                _unpack_node_result=lambda x: x,
                MiniMaxH3GeneratedAVMaskedContext="mask_context_node",
                align_frame_count=lambda n: n,
            )
            execute_node(branch, ns)
            if disabled:
                self.assertEqual(len(calls["video_only"]), 1)
                self.assertEqual(len(calls["masked_context"]), 0)
            else:
                self.assertEqual(len(calls["video_only"]), 0)
                self.assertEqual(len(calls["masked_context"]), 1)
                self.assertEqual(calls["masked_context"][0]["source_latent"], {"samples": _FakeSamples("stage2")})


if __name__ == "__main__":
    unittest.main()
