"""Final-layout refinement must preserve both output and the selected objective."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfxbas import packer, refinement
from sfxbas.simulator import estimate, simulate


class Refinement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=(Path(__file__).resolve().parents[1]/'examples/amigabasic.prg').read_bytes()[2:]
        cls.opts=packer.Options()
        cls.original=packer.build(cls.data,.75,cls.opts)

    def test_real_layout_improves_size_and_copy_work(self):
        before=self.original
        after=refinement.refine_layout(self.data,before,self.opts)
        self.assertLess(len(after.prg),len(before.prg))
        self.assertLess(estimate(after.layout,after.parts),estimate(before.layout,before.parts))
        self.assertTrue(simulate(after.prg,after.layout,self.data))
        self.assertEqual(after.lam,before.lam)
        self.assertEqual(after.cycle_ties,before.cycle_ties)

    def test_failed_refinement_retains_original(self):
        with patch.object(packer,'place',side_effect=packer.NoFit('no safe layout')):
            self.assertIs(refinement.refine_layout(self.data,self.original,self.opts),self.original)

    def test_larger_candidate_cannot_replace_size_winner(self):
        larger=replace(self.original,prg=self.original.prg+b'padding')
        with patch.object(refinement,'feedback_place',return_value=lambda *a:larger):
            self.assertIs(refinement.refine_layout(self.data,self.original,self.opts),self.original)

    def test_equal_candidate_keeps_original(self):
        with patch.object(refinement,'feedback_place',return_value=lambda *a:self.original):
            self.assertIs(refinement.refine_layout(self.data,self.original,self.opts),self.original)

    def test_time_search_reports_deadline_met_by_refinement(self):
        from sfxbas import cli
        before=SimpleNamespace(prg=bytes(20),lam=0,seconds=5.01)
        after=SimpleNamespace(prg=bytes(19),lam=0,seconds=4.99)
        with patch.object(cli,'build_many',return_value={0:before}), \
                patch.object(cli,'refine_result',return_value=after), \
                patch.object(cli,'unpack_seconds',side_effect=lambda r:r.seconds),patch.object(cli,'say') as say:
            self.assertIs(cli.build_time(b'input',5,self.opts),after)
        self.assertNotIn('cannot be met',say.call_args.args[0])

    def test_source_failure_keeps_original(self):
        with patch.object(refinement,'refine_layout',side_effect=lambda t,r,opts:r), \
                patch.object(packer,'build',side_effect=packer.NoFit('source candidate does not fit')):
            self.assertIs(refinement.refine_result(self.data,self.original,self.opts),self.original)

    def test_source_regression_keeps_original(self):
        larger=replace(self.original,prg=self.original.prg+b'padding')
        with patch.object(refinement,'refine_layout',side_effect=lambda t,r,opts:r), \
                patch.object(packer,'build',return_value=larger) as build:
            self.assertIs(refinement.refine_result(self.data,self.original,self.opts),self.original)
            self.assertTrue(build.call_args.args[2].source_choices)


if __name__=='__main__':unittest.main()
