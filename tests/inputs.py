"""Test inputs: the programs in examples/ plus synthetic programs that reach staging, colour RAM and raw mode."""
import os
from pathlib import Path

ROOT = os.path.join(os.path.dirname(__file__), '..')
EXAMPLES = os.path.join(ROOT, 'examples')
DIGI = os.path.join(EXAMPLES, 'digi.prg')


def lcg(seed):
    while True:
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        yield seed >> 16


def synthetic(end, noise=()):
    """digi.prg padded up to end: compressible filler, a zero block at $6000, random bytes in noise."""
    rnd = lcg(1)
    t = bytearray(Path(DIGI).read_bytes()[2:])
    while 0x0801 + len(t) < end:
        x = 0x0801 + len(t)
        if any(a <= x < b for a, b in noise): t.append(next(rnd) & 255)
        elif x >> 12 == 6: t.append(0)
        else: t += bytes([x >> 8 & 255, next(rnd) & 3])
    return b'\x01\x08' + bytes(t[:end - 0x0801])


def repetitive(end):
    """digi.prg padded with a repeating block and some noise: compresses well enough to stage $D000+."""
    rnd = lcg(7)
    t = bytearray(Path(DIGI).read_bytes()[2:])
    block = bytes(next(rnd) & 255 for _ in range(200))
    while 0x0801 + len(t) < end:
        t += block if next(rnd) % 8 else bytes(next(rnd) & 255 for _ in range(16))
    return b'\x01\x08' + bytes(t[:end - 0x0801])


def bitmap(end):
    """digi.prg repeated up to $A000, then a bitmap at $A000: a few 320-byte rows that repeat, some patched. The
    rows occur nowhere else, so only staging $A000-$BFFF lets them copy."""
    rnd = lcg(3)
    digi = Path(DIGI).read_bytes()[2:]
    t = bytearray((digi * 8)[:0xA000 - 0x0801])
    rows = [bytes(next(rnd) & 255 for _ in range(320)) for _ in range(4)]
    while 0x0801 + len(t) < end:
        row = bytearray(rows[next(rnd) % len(rows)])
        if next(rnd) % 4 == 0:
            x = next(rnd) % 300
            row[x:x + 16] = bytes(next(rnd) & 255 for _ in range(16))
        t += row
    return b'\x01\x08' + bytes(t[:end - 0x0801])


def bitmap_hi():
    """bitmap(), an all-zero $C000 block (a fill), then $D000-$F7FF."""
    rnd = lcg(5)
    t = bytearray(bitmap(0xC000)[2:]) + bytes(0x1000)
    block = bytes(next(rnd) & 255 for _ in range(150))
    while 0x0801 + len(t) < 0xF800:
        t += block if next(rnd) % 4 else bytes(next(rnd) & 255 for _ in range(24))
    return b'\x01\x08' + bytes(t[:0xF800 - 0x0801])


def write_all(d):
    """Write the test inputs to d; returns {name: path}."""
    paths = {'digi': DIGI, 'superbox1': os.path.join(EXAMPLES, 'superbox1.prg'),
             'parallaxative': os.path.join(EXAMPLES, 'parallaxative.prg'),
             'superscroller': os.path.join(EXAMPLES, 'superscroller.prg')}
    for name, data in (('stage', synthetic(0xF800)), ('col', synthetic(0x10400)),
                       ('raw', synthetic(0xC000, [(0x9000, 0xC000)])),
                       ('staged', repetitive(0xF800)), ('staged-col', repetitive(0x10400)),
                       ('bitmap', bitmap(0xC000)), ('bitmap-hi', bitmap_hi())):
        paths[name] = os.path.join(d, name + '.prg')
        Path(paths[name]).write_bytes(data)
    paths['rom'] = os.path.join(d, 'rom.bin')                # a stand-in char ROM: 4 KB of digi.prg
    Path(paths['rom']).write_bytes(Path(DIGI).read_bytes()[3000:3000 + 4096])
    return paths
