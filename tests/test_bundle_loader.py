"""Bundle routing and pre-deserialization checks with small CPU stubs."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


class BundleTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)
        self.root = self.output / 'latent/erase_tomorrow/muse_stage1_scout/run'
        self.candidate = self.root / 'candidate_0'
        self.candidate.mkdir(parents=True)
        self.calls = []
        self.refs = [types.SimpleNamespace(shape=(1, 8, 16, 3)), types.SimpleNamespace(shape=(1, 16, 8, 3))]
        self.reference_payloads = [
            {'ref_image_0': self.refs[0]},
            {'ref_image_0': self.refs[1]},
        ]
        def load(path, **kwargs):
            self.calls.append(path)
            index = int(Path(path).stem.split('_')[1]) - 1
            return {'latent': {'samples': index}, 'ref_images': self.reference_payloads[index]}
        modules = {'folder_paths': types.SimpleNamespace(get_output_directory=lambda: str(self.output)),
                   'torch': types.SimpleNamespace(load=load)}
        spec = importlib.util.spec_from_file_location('bundle_under_test', Path(__file__).resolve().parents[1] / 'muse_stage1_bundle_loader.py')
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', modules):
            spec.loader.exec_module(self.module)

    def test_preserves_per_group_reference_identity_and_shape(self):
        for index in (1, 2):
            (self.candidate / f'chunk_{index:04d}.pt').touch()
        latent, bundle = self.module.MuseStage1ScoutBundleLoad().load(str(self.root), 0, 39)
        refs = bundle['__muse_per_chunk_ref_images__']
        self.assertIs(refs[0]['ref_image_0'], self.refs[0])
        self.assertIs(refs[1]['ref_image_0'], self.refs[1])
        self.assertEqual(latent['_muse_scout_bundle']['chunk_count'], 2)
        self.assertEqual(latent['samples'], 1)

    def test_preserves_intentionally_empty_reference_chunk(self):
        for index in (1, 2):
            (self.candidate / f'chunk_{index:04d}.pt').touch()
        self.reference_payloads[1] = {}
        latent, bundle = self.module.MuseStage1ScoutBundleLoad().load(str(self.root), 0, 39)
        refs = bundle['__muse_per_chunk_ref_images__']
        self.assertIs(refs[0]['ref_image_0'], self.refs[0])
        self.assertEqual(refs[1], {})
        self.assertEqual(latent['_muse_scout_bundle']['chunk_count'], 2)

    def test_rejects_nonempty_reference_payload_without_tensors(self):
        (self.candidate / 'chunk_0001.pt').touch()
        self.reference_payloads[0] = {'ref_image_0': object()}
        with self.assertRaisesRegex(ValueError, 'unusable'):
            self.module.MuseStage1ScoutBundleLoad().load(str(self.root), 0, 39)

    def test_gap_rejected_before_deserialization(self):
        for index in (1, 3):
            (self.candidate / f'chunk_{index:04d}.pt').touch()
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            self.module.MuseStage1ScoutBundleLoad().load(str(self.root), 0, 39)
        self.assertEqual(self.calls, [])

    def test_outside_root_rejected_before_deserialization(self):
        with self.assertRaisesRegex(ValueError, 'inside'):
            self.module.MuseStage1ScoutBundleLoad().load(str(self.output), 0, 39)
        self.assertEqual(self.calls, [])

    def test_empty_bundle_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self.module.MuseStage1ScoutBundleLoad().load(str(self.root), 0, 39)


if __name__ == '__main__':
    unittest.main()
