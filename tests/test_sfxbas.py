"""python3 -m unittest discover tests        (SFXBAS_SLOW=1 adds the big inputs)
python3 tests/test_sfxbas.py --update      rewrite golden.json after an intended output change
"""
import hashlib, json, os, random, subprocess, sys, tempfile, unittest, unittest.mock
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import inputs

GOLDEN = os.path.join(HERE, 'golden.json')
# name: (input, arguments). Slow ones only run with SFXBAS_SLOW=1.
CASES = {
    'digi-default': ('digi', []),
    'digi-lambda2': ('digi', ['2']),
    'digi-lambda40-keep0': ('digi', ['40', '-keep0']),
    'digi-size': ('digi', ['-size']),
    'digi-speed': ('digi', ['-speed']),
    'digi-time5': ('digi', ['-time=5']),
    'digi-charrom': ('digi', ['-charrom={rom}', '-charmin=16']),
    'digi-charrom-size': ('digi', ['-size', '-charrom={rom}']),
    'superbox1': ('superbox1', []),
    'superbox1-charrom-speed04': ('superbox1', ['-speed=0.4', '-charrom={rom}']),
    'parallaxative': ('parallaxative', []),
    'parallaxative-size': ('parallaxative', ['-size']),
    'raw': ('raw', []),
    'staged': ('staged', []),
    'staged-col': ('staged-col', []),
    'bitmap': ('bitmap', []),
    'bitmap-hi': ('bitmap-hi', []),
    'superscroller-speed4': ('superscroller', ['-speed=4']),
    'superbox1-raw': ('superbox1', ['-raw']),
    'parallaxative-raw': ('parallaxative', ['-raw']),
}
SLOW = {
    'stage-raw': ('stage', []),
    'col-raw': ('col', []),
}


def pack(src, args, out, paths):
    args = [a.format(**paths) for a in args]
    subprocess.run([sys.executable, '-m', 'sfxbas', src, *args, '-o', out], cwd=ROOT,
                   check=True, capture_output=True, env=dict(os.environ, SFXBAS_JOBS='1'))
    return hashlib.sha256(Path(out).read_bytes()).hexdigest()


def pack_cases(cases, paths, directory, report=False):
    """Each CLI runs in its own process; retain case order in the resulting hashes."""
    jobs = max(1, int(os.environ.get('SFXBAS_TEST_JOBS', '4')))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = {name: pool.submit(pack, paths[src], args,
                                    os.path.join(directory, name + '.out'), paths)
                   for name, (src, args) in cases.items()}
        names = {task: name for name, task in pending.items()}
        results = {}
        for task in as_completed(names):
            name = names[task]
            results[name] = task.result()
            if report:
                print('packed ' + name, flush=True)
        return {name: results[name] for name in cases}


class Golden(unittest.TestCase):
    """Packed files must not change unless we mean it."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = inputs.write_all(cls.tmp.name)
        cls.golden = json.loads(Path(GOLDEN).read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, name, case):
        src, args = case
        got = pack(self.paths[src], args, os.path.join(self.tmp.name, name + '.out'), self.paths)
        self.assertEqual(got, self.golden[name], name)


def golden_test(name, case, slow=False):
    def test(self):
        self.check(name, case)
    if slow:
        test = unittest.skipUnless(os.environ.get('SFXBAS_SLOW'), 'set SFXBAS_SLOW=1')(test)
    return test


# Individual methods let the parallel runner schedule golden cases within its worker budget.
for _name, _case in {**CASES, **SLOW}.items():
    setattr(Golden, 'test_' + _name.replace('-', '_'), golden_test(_name, _case, _name in SLOW))


class Cli(unittest.TestCase):
    def check_petcat_error(self, kind, message, args=()):
        with tempfile.TemporaryDirectory() as tmp:
            if kind != 'missing':
                tool = Path(tmp, 'petcat')
                tool.write_text('#!/bin/sh\nprintf "tokenizer diagnostic\\n" >&2\nexit 7\n')
                tool.chmod(0o755 if kind == 'failed' else 0o644)
            out = Path(tmp, 'packed.prg')
            env = dict(os.environ, PATH=tmp, SFXBAS_JOBS='2')
            result = subprocess.run([sys.executable, '-m', 'sfxbas', inputs.DIGI, *args, '-o', str(out)],
                                    cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(message, result.stderr)
            self.assertNotIn('Traceback', result.stderr)
            self.assertFalse(out.exists())

    def test_petcat_missing(self):
        self.check_petcat_error('missing', 'petcat not found')

    def test_petcat_not_executable(self):
        self.check_petcat_error('permission', 'cannot execute petcat')

    def test_petcat_failed(self):
        self.check_petcat_error('failed', 'petcat failed (exit 7): tokenizer diagnostic')

    def test_petcat_failed_parallel_time(self):
        self.check_petcat_error('failed', 'petcat failed (exit 7): tokenizer diagnostic', ('-time=5',))

    def test_petcat_failed_parallel_search(self):
        self.check_petcat_error('failed', 'petcat failed (exit 7): tokenizer diagnostic', ('-size',))


class SizeSearch(unittest.TestCase):
    def test_retains_baseline_and_intermediate_pool_candidates(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from sfxbas import cli
        from sfxbas.packer import Options
        # Exercise both guarantees: enhanced parses can regress or fail, and
        # round two can win even when later pooling rounds grow the PRG.
        for winner in ('baseline', 'round2', 'seed'):
            baseline = SimpleNamespace(prg=bytes(20), lam=0)
            intermediate = SimpleNamespace(prg=bytes(19), lam=0)
            def result(opts):
                if not opts.search_layout and opts.pool_rounds == 1:
                    result = baseline
                elif winner == 'round2' and opts.pool_rounds == 2 and not opts.search_layout:
                    result = intermediate
                elif winner == 'seed' and opts.pool_seed is not None and opts.pool_rounds == 3:
                    result = intermediate
                elif opts.pool_rounds == 4:
                    result = None
                else:
                    result = SimpleNamespace(prg=bytes(25), lam=0)
                return result
            def builds(t, keys, variants):
                return {key: [result(opts) for tag, opts in variants] for key in keys}
            with self.subTest(winner=winner), patch.object(cli, 'build_size_many', builds), \
                    patch.object(cli, 'unpack_seconds', return_value=1), patch.object(cli, 'say'), \
                    patch.object(cli, 'refine_result', side_effect=lambda t,r,opts,*run:r):
                result = cli.build_size(b'input', Options())
                self.assertIs(result, baseline if winner == 'baseline' else intermediate)


class PlacementCache(unittest.TestCase):
    def test_returned_results_cannot_mutate_cached_verified_layout(self):
        from sfxbas import packer
        from sfxbas.simulator import simulate
        t = Path(inputs.DIGI).read_bytes()[2:]
        opts = packer.Options(paired=False,cycle_ties=False)
        packer.cached_place.cache_clear()
        first = packer.build(t,0,opts)
        expected = first.prg
        first.parts.clear()
        first.layout.holes.append((0,65536))
        first.layout.chunks[0].pairs.clear()
        first.layout.first_chunk = 0
        second = packer.build(t,0,opts)
        self.assertEqual(second.prg,expected)
        self.assertTrue(second.parts)
        self.assertTrue(second.layout.chunks[0].pairs)
        self.assertTrue(simulate(second.prg,second.layout,t))
        self.assertGreater(packer.cached_place.cache_info().hits,0)


class Matcher(unittest.TestCase):
    def test_literal_pool_index_matches_independent_substring_search(self):
        from sfxbas.matcher import Part,pool_matches,MAX_PART_LEN
        rnd = random.Random(17)
        for case in range(400):
            t = bytes(rnd.randrange(5) for _ in range(rnd.randrange(1,160)))
            literals = []
            parts, start = [], 0
            while start<len(t):
                length = min(rnd.randrange(1,35),len(t)-start)
                kind = rnd.choice(('lit','lit','match'))
                parts.append(Part(start,length,kind,0))
                if kind=='lit':literals.append((start,length))
                start += length
            for minimum in (0,1,2,3):
                expected = []
                previous = []
                for a,n in literals:
                    for p in range(a+1,a+n+1):
                        length = max((L for L in range(1,min(p-a,MAX_PART_LEN)+1)
                                      if any(t[p-L:p] in owner for owner in previous)),default=0)
                        if length>=minimum:expected.append((p,length))
                    previous.append(t[a:a+n])
                self.assertEqual(pool_matches(t,parts,minimum),tuple(expected),(case,minimum))

    def test_literal_pool_never_crosses_owners_or_reads_itself(self):
        from sfxbas.matcher import Part,pool_matches
        self.assertEqual(pool_matches(b'ababab',[Part(0,6,'lit',0)]),())
        t = b'abCDabCD'
        parts = [Part(0,2,'lit',0),Part(2,2,'lit',0),Part(4,4,'lit',0)]
        self.assertEqual(pool_matches(t,parts),((6,2),(8,2)))

    def test_overlapping_literals_read_their_own_bytes(self):
        from sfxbas.stream import overlap_literals
        t = b'xyzABCDEFGHIJ' + b'GHIJKLM' + b'DEF'
        needs = [(('l', 20), 3, 0), (('l', 13), 7, 0), (('l', 3), 10, 0)]      # reading order
        shared, store = overlap_literals(t, needs)
        self.assertEqual(store, {3: (b'ABCDEFGHIJKLM', 0)})                     # one group, overlap GHIJ
        self.assertEqual(shared, {13: (3, 6), 20: (3, 3)})
        rng = random.Random(5)
        for case in range(200):
            t = bytes(rng.choice(b'ab\0') for _ in range(400))
            needs, s = [], 0
            while s < len(t) - 1:
                L = min(rng.randrange(1, 40), len(t) - s)
                needs.append((('l', s), L, 0)); s += L
            rng.shuffle(needs)
            shared, store = overlap_literals(t, needs)
            order = [key[1] for key, L, y in needs]
            for key, L, y in needs:
                a = key[1]
                b, o = shared.get(a, (a, None))
                group, own = store[b]
                if o is None: o = own
                self.assertEqual(group[o:o + L], t[a:a + L], case)
                self.assertGreaterEqual(order.index(b), order.index(a), case)   # its group is read last
            self.assertLessEqual(sum(len(g) for g, o in store.values()), sum(L for k, L, y in needs))

    def test_readable_match_beyond_nearby_candidates(self):
        from sfxbas.matcher import parse
        block = bytes(range(40))
        decoys = b''.join(bytes([100 + i]) * 36 + block[-4:] for i in range(25))
        t = block + decoys + block + block
        parts = parse(t, 0, base=0, can_read=lambda x: x < len(t) - len(block))
        self.assertEqual((parts[0].start, parts[0].length, parts[0].kind, parts[0].src),
                         (0, len(block), 'match', len(block) + len(decoys)))

    def test_three_byte_readable_alternative(self):
        from sfxbas.matcher import parse
        t = b'abcdefabc!XYZdef!abcdef'
        parts = parse(t, 0, base=0, can_read=lambda x: x < len(t) - 6)
        self.assertTrue(any(p.start == 3 and p.length == 3 and p.kind == 'match' for p in parts))

    def test_short_staging_matches(self):
        from sfxbas.matcher import parse
        for length in (1, 2, 3):
            with self.subTest(length=length):
                t = b'abc'[:length]
                parts = parse(t, 0, base=0x1000, can_read=lambda x: False, extra=[(0, t)])
                self.assertEqual([(p.length, p.kind, p.src) for p in parts], [(length, 'match', -0x1000)])

    def test_short_readable_alternatives(self):
        from sfxbas.matcher import parse
        for t, length, blocked, source in ((b'aQaRa', 1, 2, 4), (b'abQabRab', 2, 3, 6)):
            with self.subTest(length=length):
                parts = parse(t, 0, base=0, can_read=lambda x: not blocked <= x < blocked + length, stop=length)
                self.assertEqual([(p.length, p.kind, p.src) for p in parts], [(length, 'match', source)])

    def test_small_parses_match_exhaustive_search(self):
        from sfxbas.matcher import parse
        rnd = random.Random(4)
        for case in range(500):
            t = bytes(rnd.randrange(5) for _ in range(rnd.randrange(4, 20)))
            readable = [rnd.choice((False, True, True)) for _ in t]
            extra = [(-30, bytes(rnd.randrange(5) for _ in range(20)))] if case % 2 else []
            lam = rnd.choice((0, .25, 2, 10))
            best = [0] + [float('inf')] * len(t)
            for p in range(1, len(t) + 1):
                for a in range(p):
                    length = p - a
                    copy = any(t[a:p] == t[s:s + length] and all(readable[s:s + length])
                               for s in range(a + 1, len(t) - length + 1))
                    copy |= any(t[a:p] == data[s:s + length] and addr + s + length <= a
                                for addr, data in extra for s in range(len(data) - length + 1))
                    best[p] = min(best[p], best[a] + 3 + lam + (0 if copy else length))
            parts = parse(t, lam, base=0, can_read=lambda x: readable[x], extra=extra)
            score = sum(3 + lam + (p.length if p.kind == 'lit' else 0) for p in parts)
            self.assertEqual(score, best[-1], (case, t, readable, extra, lam))
            mem = {i: -1 for i in range(len(t))}
            for addr, data in extra: mem.update((addr + i, b) for i, b in enumerate(data))
            for p in reversed(parts):
                for i in range(p.length - 1, -1, -1):
                    if p.kind == 'match' and p.src >= 0: self.assertTrue(readable[p.src + i])
                    mem[p.start + i] = t[p.start + i] if p.kind == 'lit' else mem[p.src + i]
            self.assertEqual(bytes(mem[i] for i in range(len(t))), t)

    def test_staging_match_beyond_nearby_candidates(self):
        from sfxbas.matcher import parse
        block = bytes(range(40))
        data = block + b''.join(bytes([100 + i]) * 36 + block[-4:] for i in range(25))
        parts = parse(block, 0, base=4096, can_read=lambda x: True, extra=[(0, data)])
        self.assertEqual((parts[0].length, parts[0].kind, parts[0].src), (40, 'match', -4096))

    def test_parts_rebuild_input(self):
        from sfxbas import matcher
        from sfxbas.memory import readable_at
        t = Path(inputs.DIGI).read_bytes()[2:]
        for lam in (0, 10):
            parts = matcher.parse(t, lam, base=0x0801, can_read=readable_at)
            out = bytearray(len(t))
            for s, n, kind, src in reversed(parts):          # top down, like the decruncher
                self.assertEqual(kind in ('lit', 'match'), True)
                for i in range(n - 1, -1, -1):
                    out[s + i] = t[s + i] if kind == 'lit' else out[src + i]
            self.assertEqual(bytes(out), t)


    def test_spanned_min_tree_answers_like_full_tree(self):
        from sfxbas.matcher import MinTree
        rnd = random.Random(7)
        for case in range(60):
            n, span = rnd.choice((5, 300, 3000)), rnd.choice((1, 3, 8, 100, 765, 1530))
            full, spanned = MinTree(n), MinTree(n, span)
            for op in range(2000):
                r, i = rnd.random(), rnd.randrange(n)
                if r < 0.4:
                    x = (rnd.randrange(6), rnd.randrange(10 ** 6))   # few values: many ties
                    full.add(i, x); spanned.add(i, x)
                elif r < 0.6:
                    full.clear(i); spanned.clear(i)
                else:
                    hi = min(n, i + rnd.randint(0, span))
                    self.assertEqual(spanned.min(i, hi), full.min(i, hi), (case, op))


class Tokenizer(unittest.TestCase):
    def test_lines_from_shapes_match_petcat(self):
        from sfxbas import basic, packer
        lines = set()
        tokenize = basic.tokenize.__wrapped__
        def spy(text, base):
            lines.update(line for line in text.split('\n') if line.strip() and not line.startswith(';'))
            return tokenize(text, base)
        t = Path(inputs.DIGI).read_bytes()[2:]
        with tempfile.TemporaryDirectory() as d, unittest.mock.patch.object(basic, 'tokenize', spy):
            for o in (packer.Options(), packer.Options(paired=False, bar=8)):
                packer.build(t, 10, o)
            packer.build(Path(inputs.write_all(d)['staged']).read_bytes()[2:], 4)
        lines = sorted(lines)
        for line, body in zip(lines, basic.petcat(lines)):
            shaped = basic.from_shape(line)
            self.assertIsNotNone(shaped, line)
            self.assertEqual(shaped[2:], body[2:], line)              # the link is rewritten by tokenize()


class Simulator(unittest.TestCase):
    def test_stream_bounds_match_full_placement(self):
        from sfxbas import packer
        from sfxbas.matcher import Part
        from sfxbas.memory import BASIC_START
        rng = random.Random(34)
        t = bytes(900)
        for case in range(40):
            parts, s = [], 0
            while s < len(t):
                length = min(rng.choice((1, 2, 17, 64, 128, 255)), len(t) - s)
                parts.append(Part(s, length, rng.choice(('lit', 'match', 'rom')), rng.randrange(len(t))))
                s += length
            holes = [(BASIC_START + 80, BASIC_START + 160), (BASIC_START + 500, BASIC_START + 500)]
            parts = packer.cut_at_holes(parts, holes if case % 2 else ())
            plan = [(0xD000, 0xD100, 0xC000)] if case % 3 else ()
            for paired in (False, True):
                for tail in (1, 4, 32):
                    for overlap in (True, False):
                        with self.subTest(case=case, paired=paired, tail=tail, overlap=overlap):
                            placed = packer.Stream(t, parts, plan, paired, tail, overlap=overlap).place(0)
                            expected = placed.size, min(BASIC_START + len(t) - y - top for top, y in placed.reads)
                            self.assertEqual(packer.stream_top(t, parts, plan, paired, tail, overlap=overlap),
                                             expected)

    def test_fits_matches_full_placement(self):
        from sfxbas import packer
        from sfxbas.matcher import Part
        from sfxbas.memory import BASIC_START
        rng = random.Random(35)
        t = bytes(rng.randrange(4) for _ in range(900))
        for case in range(30):
            parts, s = [], 0
            while s < len(t):
                length = min(rng.choice((1, 2, 17, 64, 128, 255)), len(t) - s)
                parts.append(Part(s, length, rng.choice(('lit', 'lit', 'match')), rng.randrange(len(t))))
                s += length
            stream = packer.Stream(t, packer.cut_at_holes(parts), (), case % 2 == 0, rng.choice((1, 4, 32)))
            size, end = stream.place(0).size, BASIC_START + len(t)
            for B in range(0, size + 300, 7):
                sd, ss = rng.randrange(0x400, 0x800), rng.randrange(0x800, 0x1000)
                with self.subTest(case=case, B=B):
                    brute = all(top <= end - y for top, y in stream.place(sd, ss, B).reads)
                    self.assertEqual(stream.fits(sd, ss, B, end), brute)

    def test_rejects_corrupt_stream(self):
        from sfxbas import packer
        from sfxbas.simulator import simulate
        t = Path(inputs.DIGI).read_bytes()[2:]
        r = packer.build(t, 10)
        self.assertTrue(simulate(r.prg, r.layout, t))
        bad = bytearray(r.prg)
        bad[-100] ^= 0xFF                                    # a literal near the stream top
        self.assertFalse(simulate(bytes(bad), r.layout, t))


if __name__ == '__main__':
    if '--update' in sys.argv:
        import argparse
        parser = argparse.ArgumentParser(description='Update all golden hashes, or only named cases.')
        parser.add_argument('--update', nargs='*', metavar='CASE')
        parser.add_argument('--jobs', type=int, default=int(os.environ.get('SFXBAS_TEST_JOBS', '4')))
        args = parser.parse_args()
        cases = {**CASES, **SLOW}
        if args.update:
            unknown = set(args.update) - cases.keys()
            if unknown:
                parser.error('unknown cases: ' + ', '.join(sorted(unknown)))
            cases = {name: cases[name] for name in args.update}
        os.environ['SFXBAS_TEST_JOBS'] = str(max(1, args.jobs))
        with tempfile.TemporaryDirectory() as d:
            paths = inputs.write_all(d)
            golden = json.loads(Path(GOLDEN).read_text())
            golden.update(pack_cases(cases, paths, d, report=True))
        Path(GOLDEN).write_text(json.dumps(golden, indent=1))
        print('wrote', GOLDEN)
    else:
        unittest.main()
