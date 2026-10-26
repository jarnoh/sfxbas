"""Run packed files on VICE (x64sc) and check memory, BASIC pointers and the time estimate.

The PRG runs exactly as shipped. RUN executes three times: autostart, the loader, and the decruncher's last
statement. At the third one the output must be complete.
"""
import glob, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import inputs, vice
from sfxbas import packer
from sfxbas.basic import tokenize, BAR_ADDR, BAR_CHAR, BAR_PRINT_CHAR, BAR_WIDTH, TITLE, TITLE_SCREEN, Layout, part_loop
from sfxbas.memory import BASIC_START, COLOR_RAM, VIRTUAL_COLOR_RAM, has_color_ram, real_address
from sfxbas.packer import Options
from sfxbas.simulator import estimate

RUN = 0xA871                                             # BASIC's RUN
TIME_TOLERANCE = 0.10                                    # estimate() vs VICE
CHARGEN = sorted(glob.glob(os.path.join(os.path.dirname(os.path.realpath(vice.X64SC)), '..', 'share', 'vice',
                                        'C64', 'chargen-901225-01.bin')))
HAVE_VICE = os.path.exists(vice.X64SC) and shutil.which('petcat')


def unpack_on_vice(r, t):
    """Run r on VICE. Returns (mismatching addresses, (TXTTAB, VARTAB), unpack cycles)."""
    end = BASIC_START + len(t)
    ranges = [(BASIC_START - 1, min(end, VIRTUAL_COLOR_RAM) - 1, 'ram')]
    if has_color_ram(end): ranges.append((COLOR_RAM, real_address(end - 1) , 'io'))
    res = vice.run(None, raw_prg=b'\x01\x08' + r.prg, stop_at=RUN, stop_hit=3, dumps=ranges + [(0x2B, 0x2E)],
                   trace=[RUN], cycles=2_000_000_000)
    bad = []
    for a, b, bank in ranges:
        for i, got in enumerate(res[(a, b, bank)]):
            addr = a + i
            if bank == 'io':
                if got & 15 != t[addr - COLOR_RAM + VIRTUAL_COLOR_RAM - BASIC_START] & 15: bad.append(addr)
            elif addr == BASIC_START - 1:
                if got != 0: bad.append(addr)
            elif got != t[addr - BASIC_START]:
                bad.append(addr)
    zp = res[(0x2B, 0x2E)]
    runs = res.traces[RUN]
    return bad, (zp[0] + 256 * zp[1], zp[2] + 256 * zp[3]), runs[2] - runs[1]


@unittest.skipUnless(HAVE_VICE, 'needs x64sc and petcat')
class Vice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = inputs.write_all(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, name, lam=10, opts=Options(), load=None):
        t = Path(self.paths[name]).read_bytes()[2:]
        if load is None:
            r = packer.build(t, lam, opts)
        else:
            from sfxbas.cli import build_speed
            r = build_speed(t, opts, load)
        bad, (txttab, vartab), cycles = unpack_on_vice(r, t)
        self.assertEqual(bad[:8], [], '%d bytes wrong' % len(bad))
        self.assertEqual((txttab, vartab), (BASIC_START, r.layout.basic_end))
        est = estimate(r.layout, r.parts)
        self.assertLess(abs(est / cycles - 1), TIME_TOLERANCE, 'estimate %d, VICE %d cycles' % (est, cycles))
        return r

    def test_digi(self): self.check('digi')
    def test_digi_small(self): self.check('digi', 0.25, Options(paired=False))
    def test_superbox1(self): self.check('superbox1')
    def test_parallaxative(self): self.check('parallaxative')
    def test_superscroller(self): self.check('superscroller', 2)
    def test_staged(self): self.check('staged')
    def test_staged_colour(self): self.check('staged-col')
    def test_raw(self): self.check('raw')
    def test_bitmap(self): self.check('bitmap')
    def test_bitmap_hi(self): self.check('bitmap-hi')
    def test_superbox1_raw(self): self.check('superbox1', opts=Options(raw=True))
    def test_superscroller_bar(self): self.check('superscroller', opts=Options(bar=10))
    def test_parallaxative_raw(self): self.check('parallaxative', opts=Options(raw=True))

    def test_refinement_preserves_staging_raw_colour_and_bar(self):
        from sfxbas.refinement import refine_result
        for name, opts in (('staged-col', Options()), ('raw', Options()),
                           ('superbox1', Options(raw=True)), ('superscroller', Options(bar=10))):
            with self.subTest(name=name, opts=opts):
                t = Path(self.paths[name]).read_bytes()[2:]
                original = packer.build(t, 10, opts)
                result = refine_result(t, original, opts)
                self.assertLessEqual(packer.rank(result, t, opts), packer.rank(original, t, opts))
                bad, pointers, cycles = unpack_on_vice(result, t)
                self.assertEqual(bad[:8], [])
                self.assertEqual(pointers, (BASIC_START, result.layout.basic_end))

    def test_loader_title_position(self):
        t = Path(self.paths['digi']).read_bytes()[2:]
        r = packer.build(t, 10)
        row = (BAR_ADDR, BAR_ADDR + BAR_WIDTH - 1)
        res = vice.run(None, raw_prg=b'\x01\x08' + r.prg, stop_at=RUN, stop_hit=2, dumps=[row])
        self.assertEqual(res[row][-len(TITLE):], TITLE_SCREEN)

    def test_exit_cursor_bottom(self):
        # After the unpack, the next CR lands on the bottom row, below the bar and the decruncher text.
        t = tokenize('10 print:print"ab";:sys53240\n', BASIC_START)
        r = packer.build(t, 10)
        res = vice.run(None, raw_prg=b'\x01\x08' + r.prg, stop_at=0xCFF8, dumps=[(0x0400, 0x07E7)])
        scr = res[(0x0400, 0x07E7)]
        self.assertEqual(scr[24 * 40:24 * 40 + 3], b'\x01\x02 ')          # "ab" in screen codes, row 24

    def test_progress_bar_repeated_endpoints(self):
        # Run the generated bar statement with empty ranges as well as advances.
        # FOR must leave W at the next column without touching column 41.
        for poke in (False, True):
            with self.subTest(poke=poke):
                v = Layout(0xE000 if poke else 0x9000, (), [], (True, False), 0x0803)
                code = [x for x in part_loop(v, 30, {'bar': 'v', 'lo': '1', 'hi': '2'}) if not x.startswith(';')]
                bar = code[-2].split(' ', 1)[1]
                start = 1025 if poke else 1
                source = ('10 v=3:o=1:b=%d:w=%d:dimn%%(3):poke1064,65:print"{home}{swlc}%s";\n'
                          '20 forz=1to5:readn%%(v)\n'
                          '30 %s\n'
                          '40 next:poke200,w-%d:%s\n'
                          '50 data %d,%d,%d,%d,%d\n'
                          % (BAR_CHAR, start, BAR_PRINT_CHAR, bar, start - 1, vice.END,
                             start - 1, start, start, start + 38, start + 38))   # end columns minus one
                res = vice.run(source, dumps=[(200, 200), (1024, 1064)])
                self.assertEqual(res[(200, 200)], bytes([40]))
                self.assertEqual(res[(1024, 1064)], bytes([BAR_CHAR]) * 40 + b'A')

    def test_fill_partial_pages(self):
        # The fill's low length byte lives in $71 across PEEK and arithmetic.
        for tail in (0, 1, 255):
            with self.subTest(tail=tail):
                t = b'\0\0' + b'V' * (2048 + tail) + bytes(range(256))
                # Force the fill variant: the size search prefers string copies for this tiny input.
                r = packer.build_once(t, 10, Options(), (), lambda _: None)
                self.assertTrue(r.layout.fill_runs)
                bad, pointers, _ = unpack_on_vice(r, t)
                self.assertEqual(bad, [])
                self.assertEqual(pointers, (BASIC_START, r.layout.basic_end))

    @unittest.skipUnless(CHARGEN, 'no VICE chargen ROM')
    def test_superbox1_charrom(self):
        self.check('superbox1', opts=Options(char_rom=Path(CHARGEN[0]).read_bytes()))

    @unittest.skipUnless(CHARGEN, 'no VICE chargen ROM')
    def test_superbox1_charrom_speed04(self):
        r = self.check('superbox1', opts=Options(char_rom=Path(CHARGEN[0]).read_bytes()), load=0.4)
        self.assertTrue(any(p.kind == 'rom' for p in r.parts))


def ram_on_vice(prg, hit, size):
    """RAM $0801.. (size bytes), TXTTAB and VARTAB at the hit-th RUN of prg."""
    rng = (BASIC_START, min(BASIC_START + size, 0x10000) - 1, 'ram')
    res = vice.run(None, raw_prg=prg, stop_at=RUN, stop_hit=hit, cycles=2_000_000_000, dumps=[rng, (0x2B, 0x2E)])
    zp = res[(0x2B, 0x2E)]
    return res[rng], zp[0] + 256 * zp[1], zp[2] + 256 * zp[3]


@unittest.skipUnless(HAVE_VICE, 'needs x64sc and petcat')
class Examples(unittest.TestCase):
    """examples/*.prg: RAM after loading the original must equal RAM after the sfx version has unpacked.

    The original is caught at autostart's RUN, the sfx version at the decruncher's (the third RUN)."""
    def test_examples(self):
        """RAM must also hold the .prg byte for byte, zero blocks included: a program can show its own zero blocks
        (a blank bitmap)."""
        with tempfile.TemporaryDirectory() as d:
            for src in sorted(glob.glob(os.path.join(inputs.EXAMPLES, '*.prg'))):
                with self.subTest(os.path.basename(src)):
                    orig = Path(src).read_bytes()
                    out = os.path.join(d, 'sfx.prg')
                    subprocess.run([sys.executable, '-m', 'sfxbas', src, '-o', out], cwd=ROOT,
                                   check=True, capture_output=True)
                    size = len(orig) - 2
                    want, txttab, vartab = ram_on_vice(orig, 1, size)
                    got, txttab2, vartab2 = ram_on_vice(Path(out).read_bytes(), 3, size)
                    bad = [BASIC_START + i for i in range(len(want)) if got[i] != want[i]]
                    self.assertEqual(['$%04X' % a for a in bad[:8]], [], '%d bytes differ' % len(bad))
                    bad = [BASIC_START + i for i in range(size) if got[i] != orig[2 + i]]
                    self.assertEqual(['$%04X' % a for a in bad[:8]], [], '%d bytes differ from the .prg' % len(bad))
                    self.assertEqual(txttab2, txttab)
                    self.assertEqual(vartab2, vartab)          # where LOAD leaves it

if __name__ == '__main__':
    unittest.main()
