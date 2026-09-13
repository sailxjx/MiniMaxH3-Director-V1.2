"""CPU-only tests for the prompt-override reference slot contract and frame budget."""
import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'muse_minimax_director.py').read_text(encoding='utf-8')
DIRECTOR = ast.parse(SOURCE)


def node_named(name):
    for node in DIRECTOR.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node
    raise LookupError(name)


def load(*names, **namespace):
    namespace.setdefault('re', re)
    for name in names:
        module = ast.Module(body=[node_named(name)], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<production AST>', 'exec'), namespace)
    return namespace


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


class CarrierPolicyTests(unittest.TestCase):
    def setUp(self):
        self.resolve = load('_CARRY_REFERENCE_KEYS', '_resolve_carry_reference_injection')['_resolve_carry_reference_injection']

    def test_upstream_default_without_override(self):
        self.assertEqual(self.resolve({}, False), (True, True))
        self.assertEqual(self.resolve({'carry_reference_injection': {'picture_anchor': False}}, False), (False, True))

    def test_override_disables_both_carriers(self):
        self.assertEqual(self.resolve({}, True), (False, False))
        self.assertEqual(self.resolve({'carry_reference_injection': {'picture_anchor': False,
                                                                     'previous_audio_tail': False}}, True), (False, False))

    def test_override_refuses_an_explicit_carrier(self):
        with self.assertRaisesRegex(ValueError, 'previous_audio_tail cannot be enabled together'):
            self.resolve({'carry_reference_injection': {'previous_audio_tail': True}}, True)

    def test_malformed_switches_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'must be an object'):
            self.resolve({'carry_reference_injection': []}, False)
        with self.assertRaisesRegex(ValueError, 'Unknown carry_reference_injection key'):
            self.resolve({'carry_reference_injection': {'anchor': True}}, False)
        with self.assertRaisesRegex(ValueError, 'JSON booleans'):
            self.resolve({'carry_reference_injection': {'picture_anchor': 1}}, False)


class FrameBudgetTests(unittest.TestCase):
    def setUp(self):
        self.validate = load('_validate_explicit_chunk_frames',
                             align_frame_count=align_frame_count)['_validate_explicit_chunk_frames']

    def test_first_group_and_carry_free_groups_are_17k_plus_5(self):
        self.validate([243], 39)
        self.validate([243, 243, 175], 0)
        with self.assertRaisesRegex(ValueError, 'legal H3 17k\\+5'):
            self.validate([238], 39)

    def test_raw_carry_continuations_are_multiples_of_17(self):
        self.validate([260, 255, 187], 39)
        self.validate([141, 136, 136], 90)
        with self.assertRaisesRegex(ValueError, r'chunk_frames\[1\]=243 plus the 39-frame raw carry'):
            self.validate([243, 243, 175], 39)

    def test_shape_errors(self):
        for frames in ([], [243.0], 'x', [3]):
            with self.assertRaises(ValueError):
                self.validate(frames, 39)


class MediaTagTests(unittest.TestCase):
    def setUp(self):
        self.validate = load('_MEDIA_TAG_RE', '_validate_override_media_tags')['_validate_override_media_tags']

    def test_exact_declaration_passes(self):
        prompt = '<Picture 1> and <Picture 2> sit; <Audio 1> speaks, <Audio 2> answers.'
        self.validate(2, prompt, {'picture': 2, 'audio': 2, 'video': 0})

    def test_reused_assets_keep_local_numbering(self):
        # b070 v010 G02 re-listed four images and no audio: under an override the
        # node adds nothing, so exactly <Picture 1-4> is correct.
        prompt = ' '.join(f'<Picture {n}>' for n in range(1, 5))
        self.validate(2, prompt, {'picture': 4, 'audio': 0, 'video': 0})

    def test_dangling_tag_is_fatal(self):
        with self.assertRaisesRegex(ValueError, r'<Picture 5> not connected \(connected picture count is 4\)'):
            self.validate(2, ' '.join(f'<Picture {n}>' for n in range(1, 6)), {'picture': 4})

    def test_unnamed_connected_slot_is_fatal(self):
        with self.assertRaisesRegex(ValueError, r'<Audio 2> connected but never named'):
            self.validate(1, '<Audio 1>', {'audio': 2})

    def test_video_paired_audio_counts_toward_audio_tags(self):
        self.validate(1, '<Video 1> <Audio 1> <Audio 2>', {'video': 1, 'audio': 2})


class WiringTests(unittest.TestCase):
    def test_carriers_follow_the_policy(self):
        self.assertIn('and not has_explicit_hybrid_ref_audios\n                        and _carry_previous_audio', SOURCE)
        self.assertIn('prev_chunk_images is not None and not vae_reencode_carry_test\n'
                      '                        and _carry_picture_anchor', SOURCE)

    def test_policy_is_resolved_once_after_parsing(self):
        self.assertEqual(SOURCE.count('_resolve_carry_reference_injection(\n            tdata, _prompt_override_active)'), 1)

    def test_explicit_frames_are_delivered_exactly_and_ui_buckets_still_snap(self):
        self.assertIn('requested_visible_frames if explicit_chunk_frames', SOURCE)
        self.assertIn('else max(17, 17 * int(math.floor(requested_visible_frames / 17.0 + 0.5)))', SOURCE)

    def test_tag_validation_runs_before_the_prompt_is_recorded(self):
        guard = SOURCE.index('_validate_override_media_tags(chunk_idx + 1, chunk_prompt')
        record = SOURCE.index('compiled_prompts.append(f"--- Chunk {chunk_idx + 1}/{num_chunks}')
        self.assertLess(guard, record)


if __name__ == '__main__':
    unittest.main()
