"""Independent checks for the experimental match classes and ring-tree parser."""
import random
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfxbas import source_parser as r


class SourceParser(unittest.TestCase):
    def test_prefix_parse_keeps_sources_above_stop(self):
        from sfxbas import matcher,cycle_parser
        t=b'ABCD'*128
        for parse in (matcher.cached_parse,matcher.paired_parse,
                      cycle_parser.single,cycle_parser.paired,r.paired):
            with self.subTest(parser=parse.__module__+'.'+parse.__name__):
                parts=parse(t,10,0,bytes([1])*len(t),(),(),100)
                self.assertEqual(len(parts),1)
                part=parts[0]
                self.assertEqual((part.start,part.length,part.kind),(0,100,'match'))
                self.assertGreater(part.src+part.length,100)
                self.assertEqual(t[part.src:part.src+part.length],t[:100])

    def test_active_sources_against_sorted_ranks(self):
        import bisect
        rnd=random.Random(256)
        for n in (0,1,255,256,257,513,65536):
            active=r.ActiveSources(n)
            ranks=[]
            for op in range(2000):
                if n and rnd.random()<.5:
                    rank=rnd.randrange(n)
                    active.add(rank)
                    i=bisect.bisect_left(ranks,rank)
                    if i==len(ranks) or ranks[i]!=rank:ranks.insert(i,rank)
                query=rnd.randrange(n+1)
                i=bisect.bisect_left(ranks,query)
                expected=ranks[max(0,i-1):i]+ranks[i:i+1]
                self.assertEqual(active.neighbours(query),expected,(n,op,query))

    def test_pair_trees_preserve_independent_tree_ties_and_reuse(self):
        from sfxbas.matcher import MinTree
        rnd=random.Random(35)
        reference=(MinTree(256,255),MinTree(256,255))
        trees=r.PairTrees()
        for op in range(20000):
            slot=rnd.randrange(256)
            for tree in reference:tree.clear(slot)
            if rnd.random()<.8:
                cost,offset=rnd.randrange(5),rnd.randrange(5)
                state=(op,slot)
                reference[0].add(slot,(cost,state))
                reference[1].add(slot,(cost-offset,state))
                trees.set(slot,cost,state,offset)
            else:trees.set(slot)
            lo=rnd.randrange(256)
            hi=min(256,lo+rnd.randrange(1,256))
            for which,tree in enumerate(reference):
                self.assertEqual(trees.min(which,lo,hi),tree.min(lo,hi),(op,which))

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
