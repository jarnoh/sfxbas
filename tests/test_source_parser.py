"""Independent checks for the experimental match classes and ring-tree parser."""
import random
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfxbas import source_parser as r


class SourceParser(unittest.TestCase):
    def test_source_classes_against_exhaustive_search(self):
        rnd=random.Random(812)
        for case in range(100):
            n=rnd.randrange(5,65)
            t=bytes(rnd.randrange(4) for _ in range(n))
            ok=bytes(rnd.randrange(5)!=0 for _ in range(n))
            edges=tuple(sorted({rnd.randrange(n) for _ in range(3)}))
            distances=(1,3,12,32)
            for distance,(lengths,sources) in zip(distances,r.source_classes(t,ok,edges,distances)):
                for p in range(1,n+1):
                    best=0
                    for top in range(p+distance,n+1):
                        for length in range(1,min(p,top,255)+1):
                            if not all(ok[top-length:top]) or any(top-length<e<top for e in edges):break
                            if t[p-length:p]==t[top-length:top]:best=max(best,length)
                    self.assertEqual(lengths[p],best,(case,p,distance))
                    if best:
                        self.assertEqual(t[p-best:p],t[sources[p]:sources[p]+best])
                        self.assertGreaterEqual(sources[p]+best-p,distance)

    def test_paired_reconstruction_and_ring_wrap(self):
        rnd=random.Random(91)
        for case in range(100):
            n=rnd.randrange(5,70) if case<90 else rnd.randrange(510,800)
            t=bytes(rnd.randrange(4) for _ in range(n))
            parts=r.paired(t,rnd.choice((0,.75,10)),0,bytes([1])*n,(),(),None)
            out=bytearray(n)
            self.assertEqual(sum(p.length for p in parts),n)
            for p in reversed(parts):
                for i in range(p.length-1,-1,-1):
                    out[p.start+i]=t[p.start+i] if p.kind=='lit' else out[p.src+i]
            self.assertEqual(out,t,case)


if __name__=='__main__':unittest.main()
