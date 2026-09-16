"""CPU-only execution of isolated production AST blocks; no GPU/ComfyUI import."""
import ast
import inspect
import json
import logging
import math
from pathlib import Path
import re
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
DIRECTOR = ast.parse((ROOT / 'muse_minimax_director.py').read_text(encoding='utf-8'))
REFINE = ast.parse((ROOT / 'muse_minimax_refine_v2.py').read_text(encoding='utf-8'))


def execute_node(node, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<production AST>', 'exec'), namespace)


def assignment(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))


def function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


class IntegrationTests(unittest.TestCase):
    def test_stage2_wrapper_megapixel_range_matches_upscaler(self):
        def target_max_values(tree):
            values = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Dict):
                    continue
                for key, value in zip(node.keys, node.values):
                    if not (isinstance(key, ast.Constant)
                            and key.value == 'two_stage_target_megapixels'):
                        continue
                    metadata = value.elts[1]
                    fields = {
                        field_key.value: ast.literal_eval(field_value)
                        for field_key, field_value in zip(metadata.keys, metadata.values)
                        if isinstance(field_key, ast.Constant)
                    }
                    values.append(fields['max'])
            return values

        self.assertEqual(target_max_values(DIRECTOR), [16.0])
        self.assertEqual(target_max_values(REFINE), [16.0])

    def test_fixed_base_resolution_bypasses_legacy_megapixel_rounding(self):
        ns = {}
        for name in ('BASE_RESOLUTION_OPTIONS', 'FIXED_BASE_RESOLUTIONS'):
            execute_node(assignment(DIRECTOR, name), ns)
        ns['_resolve_resolution'] = lambda *args: (1376, 768)
        execute_node(function(DIRECTOR, '_resolve_base_resolution'), ns)
        self.assertEqual(ns['_resolve_base_resolution']('1344x768', '16:9', 1.0, 32), (1344, 768))
        self.assertEqual(ns['_resolve_base_resolution']('960x544', '16:9', 0.5, 32), (960, 544))
        self.assertEqual(ns['_resolve_base_resolution']('auto', '16:9', 1.0, 32), (1376, 768))
        with self.assertRaises(ValueError):
            ns['_resolve_base_resolution']('1920x1080', '16:9', 1.0, 32)

    def test_base_resolution_is_optional_and_backward_compatible(self):
        node = function(DIRECTOR, 'execute')
        stub = ast.FunctionDef(name='signature', args=node.args, body=[ast.Pass()], decorator_list=[])
        ns = {}
        execute_node(stub, ns)
        self.assertEqual(inspect.signature(ns['signature']).parameters['base_resolution'].default, 'auto')

    def test_multigroup_refine_offloads_completed_outputs_and_carry_latent(self):
        source = ast.unparse(function(REFINE, 'execute'))
        self.assertIn("chunk_images.detach().to(device='cpu').contiguous()", source)
        self.assertIn("chunk_audio['waveform'].detach().to(device='cpu').contiguous()", source)
        self.assertIn('_copy_av_latent_to_cpu(chunk_sampled)', source)
        self.assertIn('torch.cuda.empty_cache()', source)

    def test_upstream_sentence_voice_binding(self):
        ns = {'re': re}
        for name in ('_DIALOGUE_RE', '_REPEATED_PUNCT_RE', '_DECORATIVE_RE'):
            execute_node(assignment(DIRECTOR, name), ns)
        for name in ('_collapse_repeated_punct', '_normalize_dialogue_text', '_wrap_dialogue'):
            execute_node(function(DIRECTOR, name), ns)
        result = ns['_wrap_dialogue']('"Hello" "Wait"', 'English', [1, 2], [2, 1], [3, 2])
        self.assertIn('<Subject 2> (S1)', result)
        self.assertIn('<Audio 3>', result)
        self.assertIn('<Subject 1> (S2)', result)
        self.assertIn('<Audio 2>', result)

    def test_upstream_saved_sigma_schedule(self):
        node = assignment(REFINE, 'resolved_steps')
        for meta, expected in [({'_muse_steps_used': 10}, 10), ({}, 12)]:
            ns = {'latent_meta': meta, 'embedded': {'_muse_steps_used': 12}, 'steps': 8}
            execute_node(node, ns)
            self.assertEqual(ns['resolved_steps'], expected)
        calls = [n for n in ast.walk(REFINE) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id in ('_refine_one_chunk_beta', '_rebuild_stage1_continuation')]
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertTrue(any(isinstance(a, ast.Name) and a.id == 'resolved_steps' for a in call.args))

    def test_new_refine_inputs_are_optional(self):
        ns = {}
        node = function(REFINE, 'execute')
        stub = ast.FunctionDef(name='signature', args=node.args, body=[ast.Pass()], decorator_list=[])
        execute_node(stub, ns)
        params = inspect.signature(ns['signature']).parameters
        for name, default in [('preserve_stage1_latents', False), ('group_audio_slots', ''),
                              ('refine_latent_directory', ''), ('ref_images_bundle', None)]:
            self.assertEqual(params[name].default, default)

    def test_refine_preserves_empty_per_chunk_reference_scope(self):
        ns = {}
        execute_node(function(REFINE, '_normalize_reference_image_bundle'), ns)
        image = types.SimpleNamespace(shape=(1, 8, 16, 3))
        global_images, per_chunk = ns['_normalize_reference_image_bundle']({
            '__muse_per_chunk_ref_images__': [{'ref_image_0': image}, {}],
        })
        self.assertEqual(global_images, {})
        self.assertIs(per_chunk[0]['ref_image_0'], image)
        self.assertEqual(per_chunk[1], {})
        with self.assertRaisesRegex(ValueError, 'unusable'):
            ns['_normalize_reference_image_bundle']({
                '__muse_per_chunk_ref_images__': [{'ref_image_0': object()}],
            })

    def test_hybrid_uses_opening_then_predecessor_frame(self):
        branch = next(n for n in ast.walk(DIRECTOR) if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Name) and n.test.id == 'use_hybrid_chunk'
                      and 'configured_first' in ast.unparse(n.body))
        for previous in (None, ['previous']):
            ns = dict(use_hybrid_chunk=True, configured_first='opening', configured_last='last', prev_chunk_images=previous)
            execute_node(branch, ns)
            self.assertEqual(ns['chunk_first'], 'opening' if previous is None else ['previous'])

    def test_previous_audio_soft_reference_guard(self):
        branch = next(n for n in ast.walk(DIRECTOR) if isinstance(n, ast.If)
                      and 'not disable_previous_audio' in ast.unparse(n.test))
        expression = compile(ast.Expression(branch.test), '<audio guard>', 'eval')
        ns = dict(prev_chunk_audio=object(), disable_previous_audio=False,
                  has_fully_copied_audio=False, has_explicit_hybrid_ref_audios=False,
                  _carry_previous_audio=True)
        self.assertTrue(eval(expression, ns))
        for flag in ('disable_previous_audio', 'has_fully_copied_audio', 'has_explicit_hybrid_ref_audios'):
            self.assertFalse(eval(expression, dict(ns, **{flag: True})))
        # The carrier policy (off under prompt_override) also removes the soft tail.
        self.assertFalse(eval(expression, dict(ns, _carry_previous_audio=False)))

    def test_raw_av_carry_default_and_video_only_dispatch(self):
        fn = function(REFINE, '_refine_one_chunk_beta')
        branch = next(n for n in fn.body if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Name) and n.test.id == 'disable_previous_audio')
        # Remove only the import to inject a CPU stub for the production helper.
        branch = ast.parse(ast.unparse(branch)).body[0]
        branch.body = [n for n in branch.body if not isinstance(n, ast.ImportFrom)]
        for disabled in (True, False):
            calls = []
            def call(*args, **kwargs):
                calls.append((args, kwargs))
                return ('carried', 39)
            ns = dict(disable_previous_audio=disabled, carry_context_latent={'samples': 'AV'},
                      MiniMaxH3GeneratedAVMaskedContext='context', recombined='fresh', carry_length=39,
                      inject_video_only=call, log=logging.getLogger(), log_label='test',
                      raw_latent_carry_test=True, _execute_comfy_node=call, _unpack_node_result=lambda x: x)
            execute_node(branch, ns)
            self.assertEqual(ns['carry_trim_frames'], 39)
            if disabled:
                self.assertEqual(calls[0][0][0], 'fresh')
            else:
                self.assertEqual(calls[0][1]['source_latent'], {'samples': 'AV'})

    def test_explicit_frame_buckets_keep_cuts(self):
        branch = next(n for n in ast.walk(DIRECTOR) if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Name) and n.test.id == 'explicit_chunk_frames')
        ns = dict(explicit_chunk_frames=[158, 277], duration_seconds=435 / 24,
                  align_frame_count=lambda x: 5 + 17 * round((x - 5) / 17),
                  vae_reencode_carry_length=39, raw_latent_carry_test=False,
                  tdata={'chunks': [{'segments': [{'prompt': 'a'}]}, {'segments': [{'prompt': 'b'}]}]})
        execute_node(function(DIRECTOR, '_validate_explicit_chunk_frames'), ns)
        execute_node(branch, ns)
        self.assertEqual(ns['buckets'][1][0]['prompt'], 'b')
        self.assertEqual(ns['buckets'][1][0]['_abs_start'], 158 / 24)
        ns['explicit_chunk_frames'] = [158.9, 277]
        with self.assertRaises(ValueError):
            execute_node(branch, ns)

    def test_seed_override_is_scoped_and_uint64(self):
        branch = next(n for n in ast.walk(DIRECTOR) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "'seed_override' in this_chunk_data")
        for value in (0, 2**64 - 1):
            ns = dict(this_chunk_data={'seed_override': value}, _native_resume_plan={}, _suffix_base_seed=12)
            execute_node(branch, ns)
            self.assertEqual(ns['pass_seed'], value)
        for value in (True, -1, 2**64, '1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                execute_node(branch, dict(this_chunk_data={'seed_override': value}, _native_resume_plan={}))
        ns = dict(this_chunk_data={}, _native_resume_plan={}, _suffix_base_seed=12, pass_seed=99)
        execute_node(branch, ns)
        self.assertEqual(ns['pass_seed'], 12)

    def test_prefix_restore_has_no_sampler(self):
        branch = next(n for n in ast.walk(DIRECTOR) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == '_native_resume_plan is not None')
        calls = [n for n in ast.walk(branch) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == '_execute_comfy_node']
        self.assertEqual([ast.unparse(n.args[0]) for n in calls], ['VAEDecode', 'VAEDecodeAudio'])


if __name__ == '__main__':
    unittest.main()
