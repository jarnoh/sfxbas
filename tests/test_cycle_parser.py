"""Independent small-input checks for the cycle-aware DP prototype."""
from fractions import Fraction
import sys
from pathlib import Path
import random
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfxbas import matcher, packer
from sfxbas.simulator import estimate
from sfxbas.stream import can_pair
from sfxbas import cycle_parser


class CycleParser(unittest.TestCase):
    def test_single_matches_exhaustive_lexicographic_dp(self):
        rnd = random.Random(4)
        for case in range(400):
            t = bytes(rnd.randrange(4) for _ in range(rnd.randrange(2,35)))
            lam = rnd.choice((0,.25,1/12,2,10))
            price = Fraction(3)+Fraction(lam)
            costs = [(Fraction(0),0)]
            for p in range(1,len(t)+1):
                choices = []
                for a in range(p):
                    length = p-a
                    copy = any(t[a:p]==t[x:x+length] for x in range(a+1,len(t)-length+1))
                    choices.append((costs[a][0]+price+(0 if copy else length),costs[a][1]+5633))
                costs.append(min(choices))
            parts = cycle_parser.single(t,lam,0,bytes([1])*len(t),(),(),None)
            score = sum(price+(p.length if p.kind=='lit' else 0) for p in parts),5633*len(parts)
            self.assertEqual(score,costs[-1],case)
            self.check_output(t,parts)

    def test_paired_preserves_baseline_primary_cost_and_output(self):
        rnd = random.Random(8)
        for case in range(400):
            t = bytes(rnd.randrange(4) for _ in range(rnd.randrange(2,35)))
            lam = rnd.choice((0,.25,2,10))
            args = t,lam,0,bytes([1])*len(t),(),(),None
            def score(parts):
                parts = list(reversed(parts)); i = count = 0
                while i<len(parts):
                    i += 2 if i+1<len(parts) and can_pair(parts[i],parts[i+1]) else 1
                    count += 1
                return sum(p.length for p in parts if p.kind=='lit')+count*(6+2*lam),count*7099
            before, after = matcher.paired_parse(*args),cycle_parser.paired(*args)
            self.assertLessEqual(score(after),score(before),case)
            self.check_output(t,after)

    def check_output(self,t,parts):
        out = bytearray(len(t))
        for part in reversed(parts):
            for i in range(part.length-1,-1,-1):
                out[part.start+i] = t[part.start+i] if part.kind=='lit' else out[part.src+i]
        self.assertEqual(bytes(out),t)

    def test_build_uses_fewer_statements_at_no_larger_final_size(self):
        t = (Path(__file__).resolve().parents[1]/'examples/digi.prg').read_bytes()[2:]
        before = packer.build(t,0,packer.Options(paired=False,cycle_ties=False))
        after = packer.build(t,0,packer.Options(paired=False))
        self.assertLessEqual(len(after.prg),len(before.prg))
        self.assertLess(len(after.parts),len(before.parts))
        self.assertLess(estimate(after.layout,after.parts),estimate(before.layout,before.parts))
        self.assertTrue(after.cycle_ties)

    def test_build_selects_using_the_requested_objective(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from sfxbas.simulator import PAL_HZ
        baseline = SimpleNamespace(prg=bytes(10),layout=100,parts=())
        for opts,size,cycles,choose_new in [
            (packer.Options(),11,80,False),
            (packer.Options(),10,80,True),
            (packer.Options(speed='max'),11,80,True),
            (packer.Options(speed=.00001),11,80,False),
            (packer.Options(speed=1000000),11,80,True),
            (packer.Options(time_limit=90/PAL_HZ),11,80,True),
            (packer.Options(time_limit=200/PAL_HZ),11,80,False),
            (packer.Options(time_limit=70/PAL_HZ),11,80,True),
        ]:
            alternative = SimpleNamespace(prg=bytes(size),layout=cycles,parts=())
            def result(t,lam,policy,log):
                return alternative if policy.cycle_ties else baseline
            with self.subTest(opts=opts,size=size), patch.object(packer,'build_staged',result), \
                    patch.object(packer,'estimate',lambda layout,parts:layout):
                self.assertIs(packer.build(bytes(100),0,opts),alternative if choose_new else baseline)

    def test_build_retains_the_other_candidate_when_one_does_not_fit(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        valid = SimpleNamespace(prg=bytes(10),layout=100,parts=())
        for fail_cycle in (True,False):
            def result(t,lam,policy,log):
                if policy.cycle_ties == fail_cycle:raise packer.NoFit('candidate does not fit')
                return valid
            with self.subTest(fail_cycle=fail_cycle), patch.object(packer,'build_staged',result), \
                    patch.object(packer,'estimate',lambda layout,parts:layout):
                self.assertIs(packer.build(bytes(100),0),valid)

if __name__=='__main__':unittest.main()
