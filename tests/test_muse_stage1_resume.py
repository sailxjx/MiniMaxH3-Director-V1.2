import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from muse_stage1_resume import SCHEMA, digest, validate_manifest, load_prefix, copy_prefix, record_chunk


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / 'output'
        self.inputs = self.root / 'input'
        self.inputs.mkdir()
        (self.inputs / 'identity.png').write_bytes(b'reference')
        self.bundle = self.output / 'latent/erase_tomorrow/muse_stage1_scout/source/candidate_0'
        self.bundle.mkdir(parents=True)
        (self.bundle / 'chunk_0001.pt').write_bytes(b'trusted sampled AV state')
        self.contract = {'seed': 123, 'first_pass_steps': 8}
        self.timeline = {'chunk_frames': [158, 260], 'chunks': [{'localCharacters': [{'file': 'identity.png'}]}, {}]}
        self.prompts = ['prefix prompt', 'suffix prompt']
        self.entry = {'index': 0, 'requested_frames': 158, 'decoded_frames': 158, 'delivered_frames': 158,
            'trim_frames': 0, 'sha256': digest(self.bundle / 'chunk_0001.pt'),
            'prompt_sha256': hashlib.sha256(self.prompts[0].encode()).hexdigest()}
        self.spec = {'schema_version': SCHEMA, 'contract': self.contract, 'timeline': self.timeline,
            'candidate_directory': str(self.bundle), 'chunks': [self.entry],
            'reference_files': [{'remote_name': 'identity.png', 'sha256': digest(self.inputs / 'identity.png')}]}

    def validate(self, spec=None, timeline=None, start=1, end=1):
        return validate_manifest(spec or self.spec, self.contract, timeline or self.timeline,
                                 self.prompts, start, end, self.output, self.inputs)

    def test_suffix_frame_change_allowed(self):
        timeline = copy.deepcopy(self.timeline)
        timeline['chunk_frames'][1] = 277
        self.assertEqual(self.validate(timeline=timeline)['start'], 1)

    def test_prefix_frame_change_rejected(self):
        timeline = copy.deepcopy(self.timeline)
        timeline['chunk_frames'][0] = 175
        with self.assertRaisesRegex(ValueError, 'prefix timeline'):
            self.validate(timeline=timeline)

    def test_hybrid_prefix_mode_and_first_frame_immutable(self):
        self.timeline['chunks'][0].update(generation_mode='Hybrid',firstFrameImage={'file':'identity.png'})
        for key,value in [('generation_mode','Reference'),('firstFrameImage',{})]:
            timeline=copy.deepcopy(self.timeline)
            timeline['chunks'][0][key]=value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'prefix timeline'):
                self.validate(timeline=timeline)

    def test_seed_change_rejected(self):
        spec = copy.deepcopy(self.spec)
        spec['contract']['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'sampling contract'):
            self.validate(spec)

    def test_prompt_change_rejected(self):
        self.prompts[0] = 'different'
        with self.assertRaisesRegex(ValueError, 'prompt changed'):
            self.validate()

    def test_reference_change_rejected(self):
        (self.inputs / 'identity.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'reference bytes'):
            self.validate()

    def test_latent_corruption_rejected(self):
        (self.bundle / 'chunk_0001.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.validate()

    def test_interior_only_rejected(self):
        timeline = copy.deepcopy(self.timeline)
        timeline['chunk_frames'].append(158)
        timeline['chunks'].append({})
        with self.assertRaisesRegex(ValueError, 'full suffix'):
            self.validate(timeline=timeline, end=1)

    def test_outside_bundle_rejected(self):
        spec = copy.deepcopy(self.spec)
        spec['candidate_directory'] = str(self.inputs)
        with self.assertRaisesRegex(ValueError, 'scout output root'):
            self.validate(spec)

    def test_copy_and_record_do_not_mutate_source(self):
        entry = self.validate()['prefix'][0]
        original = digest(entry['source_path'])
        dest = self.output / 'new/candidate_0/chunk_0001.pt'
        copy_prefix(entry, dest)
        self.assertEqual(digest(dest), original)
        self.assertEqual(digest(entry['source_path']), original)
        with self.assertRaises(ValueError):
            copy_prefix(entry, dest)
        record_chunk(dest.parent, 0, self.contract, self.timeline, self.prompts[0], 158, 158, 0, 158,
                     self.spec['reference_files'], reused_from=entry['source_path'])
        record = json.loads((dest.parent / 'native_resume.json').read_text())
        self.assertFalse(record['chunks'][0]['sampling_executed'])

    def test_loader_rechecks_before_deserialization(self):
        entry = self.validate()['prefix'][0]
        (self.bundle / 'chunk_0001.pt').write_bytes(b'changed-after-check')
        def must_not_load(path):
            self.fail('Unverified pickle was loaded')
        with self.assertRaisesRegex(ValueError, 'changed after preflight'):
            load_prefix(entry, must_not_load, self.contract)


if __name__ == '__main__':
    unittest.main()
