import ast
from pathlib import Path
import sys
import unittest
from muse_refine_audio_control import resolve_controls


class Controls(unittest.TestCase):
    def test_legacy(self):
        a, b = object(), object()
        out = resolve_controls('', '', 2, [a, None, b], True, True)
        self.assertEqual(out[0]['references'], {'ref_audio_0': a, 'ref_audio_1': b})
        self.assertFalse(out[1]['disable_previous_audio'])

    def test_silent_tail(self):
        a, b = object(), object()
        timeline = {'chunks': [{}, {}, {'generation_mode': 'Reference', 'disable_previous_audio': True}]}
        out = resolve_controls(timeline, '[[],[1,2],[]]', 3, [a,b,None], True, True)
        self.assertIsNone(out[0]['references'])
        self.assertEqual(out[1]['references'], {'ref_audio_0': a, 'ref_audio_1': b})
        self.assertIsNone(out[2]['references'])
        self.assertTrue(out[2]['disable_previous_audio'])

    def test_invalid_slots(self):
        for slots in ('[]', '[[true]]', '[[4]]', '[[1,1]]', '[[2]]', '{}'):
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                resolve_controls({}, slots, 1, [object(),None,None], True, True)

    def test_guards(self):
        for group in ({'disable_previous_audio': 'true'}, {'disable_previous_audio': True},
                      {'disable_previous_audio': True, 'generation_mode':'Hybrid'}):
            with self.subTest(group=group), self.assertRaises(ValueError):
                resolve_controls({'chunks':[{},group]}, '[[],[]]', 2, [None]*3, True, True)
        timeline={'chunks':[{}, {'disable_previous_audio':True,'generation_mode':'Reference'}]}
        for args in (('', True,True), ('[[],[]]',False,True), ('[[],[]]',True,False)):
            with self.subTest(args=args),self.assertRaises(ValueError):
                resolve_controls(timeline,args[0],2,[None]*3,args[1],args[2])


if __name__ == '__main__': unittest.main()
