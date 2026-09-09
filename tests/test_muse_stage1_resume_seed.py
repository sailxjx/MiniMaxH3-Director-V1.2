import copy
import unittest
import test_muse_stage1_resume as fixtures
from muse_stage1_resume import load_prefix


class MixedSeedTests(unittest.TestCase):
    setUp = fixtures.ResumeTests.setUp
    validate = fixtures.ResumeTests.validate

    def test_recorded_override_loads_without_changing_global_seed(self):
        self.timeline['chunks'][0]['seed_override'] = 456
        entry = self.validate()['prefix'][0]
        payload = {'latent': {'samples': {}, '_muse_seed_used': 456,
                              '_muse_first_pass_steps_used': 8},
                   'prompt': self.prompts[0]}
        self.assertIs(load_prefix(entry, lambda _: payload, self.contract), payload)
        self.assertEqual(self.contract['seed'], 123)
        payload['latent']['_muse_seed_used'] = 123
        with self.assertRaisesRegex(ValueError, 'seed mismatch'):
            load_prefix(entry, lambda _: payload, self.contract)

    def test_override_change_rejected(self):
        self.timeline['chunks'][0]['seed_override'] = 456
        changed = copy.deepcopy(self.timeline)
        changed['chunks'][0]['seed_override'] = 789
        with self.assertRaisesRegex(ValueError, 'prefix timeline'):
            self.validate(timeline=changed)

    def test_invalid_recorded_seed_rejected(self):
        for seed in (True, -1, 2**64, '456'):
            self.timeline['chunks'][0]['seed_override'] = seed
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, 'uint64'):
                self.validate()
