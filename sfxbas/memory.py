"""C64 memory map as the decruncher sees it.

I/O is banked out while unpacking, so writes to $D000-$FFFF land in RAM. Reads are another story: BASIC's string
copy sees ROM at $A000-$BFFF and $E000-$FFFF, and char ROM at $D000. Only RAM below $A000 and at $C000 can be a
copy source.

Colour RAM: input past $FFFF ($10000-$103FF) goes to $D800-$DBFF. It is the top of the output, so it is written
first, with I/O still on. A zero-width hole at $10000 then banks I/O out.

Staging: output at $D000+ can never be read back, so it is unpacked into readable windows ($C000, then free RAM
above the packed file) and copied up when done. Until then any staged byte can be a source. $A000-$BFFF can be
staged too, in a second phase: by then $C000 holds its own output, so only the RAM above the packed file is left,
reused from its top. Unless $C000 is free for good: when the output ends below $D000, [max(end, $C000), $D000) is
never written. Such a window is free from the start, and what is staged there stays readable for all output below.

Fills: a long run of one byte is a hole that BASIC's DIM clear loop fills ($B2C2, entered with the byte in A).
It writes RAM under ROM and I/O too, so such runs are never staged; where staging happens, only whole pages.
"""
import functools


BASIC_START = 0x0801
VIRTUAL_COLOR_RAM = 0x10000                     # input from here goes to colour RAM
COLOR_RAM = 0xD800
COLOR_RAM_SIZE = 0x400
INPUT_END = VIRTUAL_COLOR_RAM + COLOR_RAM_SIZE
HIGH_RAM = 0xD000                               # unreadable from here up
FIXED_WINDOW = (0xC000, 0xD000)                 # always free for staging
BASIC_ROM = (0xA000, 0xC000)                    # unreadable, staged in the second phase
CHAR_ROM_ADDR = 0xD000                          # with $01 = 51


def stage_plan(t, windows, low=(), fills=()):
    """Segments (virtual base, virtual top, window base), top first. windows: in fill order; () = no staging.
    low: windows for $A000-$BFFF, filled once $D000+ is copied up. Pages the first phase left alone go first: the
    rest still hold $D000+ bytes that output below can copy from.

    Each phase goes into its windows page by page, top down. A new segment starts wherever
    either side stops being contiguous. Whatever does not fit is written in place."""
    end = min(BASIC_START + len(t), VIRTUAL_COLOR_RAM)
    def phase(v0, v1, wins, taken=()):
        if not wins or end <= v0: return []
        vpages = [a for a in range((min(end, v1) - 1) & ~255, v0 - 1, -256)
                  if not any(f0 <= a and a + 256 <= f1 for f0, f1, b in fills)]
        wpages = [a for w0, w1 in wins for a in range(w1 - 256, w0 - 1, -256)]
        wpages.sort(key=lambda a: any(w0 <= a < w1 for w0, w1 in taken))     # stable: keeps top down
        segs = []
        for v, w in zip(vpages, wpages):
            if segs and segs[-1][0] == v + 256 and segs[-1][2] == w + 256:
                segs[-1] = (v, segs[-1][1], w)
            else:
                segs.append((v, min(v + 256, end, v1), w))
        return segs
    high = phase(HIGH_RAM, VIRTUAL_COLOR_RAM, windows)
    return tuple(high + phase(*BASIC_ROM, low, windows_used(high)))


def free_windows(t):
    """Windows for $A000-$BFFF that nothing writes afterwards: [max(end, $C000), $D000) when the output ends below
    $D000."""
    end = BASIC_START + len(t)
    if end <= BASIC_ROM[0]: return []
    w0 = max(FIXED_WINDOW[0], (end + 255) & ~255)
    return [(w0, HIGH_RAM)] if w0 < HIGH_RAM else []


FILL_MIN = 2048                                 # shorter runs are cheaper as parts


def fill_runs(t, skip=(), min_len=FILL_MIN):
    """[(start, end, byte)]: runs of one byte value worth a fill, outside skip (holes anyway) and colour RAM.
    An end in a region that can be staged is a page boundary. Fills never touch: one hole holds one fill."""
    return list(cached_fill_runs(bytes(t), tuple(skip), min_len))


@functools.lru_cache(maxsize=16)
def cached_fill_runs(t, skip, min_len):
    """fill_runs(): every build of one input looks for the same runs."""
    end = min(BASIC_START + len(t), VIRTUAL_COLOR_RAM)
    stageable = lambda x: BASIC_ROM[0] <= x < BASIC_ROM[1] or x >= HIGH_RAM
    out, a = [], BASIC_START
    while a < end:
        b, v = a, t[a - BASIC_START]
        while b < end and t[b - BASIC_START] == v: b += 1
        if min_len and b - a >= min_len:
            for s0, s1 in subtract_ranges([(a, b)], list(skip)):
                if out and s0 <= out[-1][1]: s0 = out[-1][1] + 1
                if stageable(s0): s0 = (s0 + 255) & ~255
                if stageable(s1 - 1): s1 &= ~255
                if s1 - s0 >= min_len: out.append((s0, s1, v))
        a = b
    return tuple(out)


def low_staged(plan): return [s for s in plan if s[0] < HIGH_RAM]


def windows_used(plan):
    return [(w, w + ((vt - vb + 255) & ~255)) for vb, vt, w in plan]


def real_address(a, plan=()):
    """Where virtual address a is actually written."""
    if a >= VIRTUAL_COLOR_RAM: return a - (VIRTUAL_COLOR_RAM - COLOR_RAM)
    for vb, vt, w in plan:
        if vb <= a < vt: return w + a - vb
    return a


def real_fretop(a, stage=()):
    """FRETOP for output ending at a. $10000 comes out as $0000 through lo/hi, which is what we want."""
    return real_address(a - 1, stage) + 1


def has_color_ram(end): return end > VIRTUAL_COLOR_RAM


def fretop_jumps(t, plan):
    """Zero-width holes: where FRETOP has to jump (colour RAM done, staging segment edges)."""
    end = BASIC_START + len(t)
    return (([(VIRTUAL_COLOR_RAM, VIRTUAL_COLOR_RAM)] if has_color_ram(end) else [])
            + [(x, x) for seg in plan for x in seg[:2]])


def readable_at(a):
    """Can BASIC's string copy read RAM at a?"""
    return a < 0xA000 or 0xC000 <= a < 0xD000


def readable(start, size):
    if not (0 <= start and start + size <= 65536): return False
    return size <= 0 or start + size <= 0xA000 or 0xC000 <= start and start + size <= 0xD000


def subtract_ranges(holes, cut):
    out = []
    for h0, h1 in holes:
        pieces = [(h0, h1)]
        for c0, c1 in cut:
            pieces = [q for p0, p1 in pieces for q in ((p0, min(p1, c0)), (max(p0, c1), p1)) if q[0] < q[1]]
        out += pieces
    return out


def merge_ranges(holes):
    out = []
    for h0, h1 in sorted(holes):
        if out and h0 <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], h1))
        else: out.append((h0, h1))
    return out


def mask_color_ram(t):
    """Colour RAM is 4 bits wide: drop the high nibbles."""
    return bytes(t[:VIRTUAL_COLOR_RAM - BASIC_START]) + bytes(b & 15 for b in t[VIRTUAL_COLOR_RAM - BASIC_START:])


def window_images(t, plan, drop=()):
    """[(address, bytes)]: what the windows hold once the plan is done (later segments win). Output below can
    copy from them. drop: [(start, end)] to leave out."""
    img = {}
    for vb, vt, w in plan:
        for i in range(vt - vb):
            if not any(d0 <= w + i < d1 for d0, d1 in drop): img[w + i] = t[vb - BASIC_START + i]
    runs = []
    for a in sorted(img):
        if runs and runs[-1][0] + len(runs[-1][1]) == a: runs[-1][1].append(img[a])
        else: runs.append((a, bytearray([img[a]])))
    return [(a, bytes(b)) for a, b in runs]
