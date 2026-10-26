"""Choose copy parts for each C64 memory range and split them at staging edges and holes."""
import bisect
import functools
from . import matcher
from .matcher import Part
from .memory import (BASIC_ROM, BASIC_START, HIGH_RAM, INPUT_END, VIRTUAL_COLOR_RAM,
                     has_color_ram, low_staged, readable_at, window_images, windows_used)


def char_rom_regions(t, a, b, rom, min_len):
    """[(start, end, ROM offset)] in t[a:b] that equal the char ROM.

    Glyph-aligned, min_len bytes or more, and at least 4 distinct values, so blank runs don't count."""
    if not rom: return []
    return list(cached_char_rom_regions(bytes(t), a, b, bytes(rom), min_len))


@functools.lru_cache(maxsize=16)
def cached_char_rom_regions(t, a, b, rom, min_len):
    """Staging plans and lambda searches repeatedly inspect the same ROM range."""
    idx, out = {}, []
    for o in range(0, len(rom) - 7, 8):
        idx.setdefault(rom[o:o + 8], []).append(o)
    k, n = a - BASIC_START, b - a
    i = -a % 8
    while i + 8 <= n:
        best = (0, 0)
        for o in idx.get(bytes(t[k + i:k + i + 8]), ()):
            L = 8
            while i + L < n and o + L < len(rom) and t[k + i + L] == rom[o + L]: L += 1
            best = max(best, (L, -o))
        L, o = best[0], -best[1]
        if L >= min_len and len(set(t[k + i:k + i + L])) >= 4:
            out.append((i, i + L, o)); i += (L + 7) // 8 * 8
        else:
            i += 8
    return tuple(out)


def cut_at_holes(parts, holes=()):
    """Cut parts at the holes and add one 'skip' part per hole (source = hole index). holes must be merged."""
    bounds = sorted({a - BASIC_START for h in holes for a in h})
    starts = [h0 for h0, h1 in holes]
    def in_hole(a):
        i = bisect.bisect_right(starts, a) - 1
        return i >= 0 and a < holes[i][1]
    result = []
    for part in parts:
        s, length, kind, src = part
        first, last = bisect.bisect_right(bounds, s), bisect.bisect_left(bounds, s + length)
        if first == last:
            if not in_hole(BASIC_START + s): result.append(part)
            continue
        cuts = [s] + bounds[first:last] + [s + length]
        for a, b in zip(cuts, cuts[1:]):
            if not in_hole(BASIC_START + a):
                result.append(Part(a, b - a, kind, src + a - s))
    for i, (h0, h1) in enumerate(holes):
        result.append(Part(h0 - BASIC_START, h1 - h0, 'skip', i))
    return sorted(result)


READABLE = bytes(readable_at(a) for a in range(INPUT_END))


def parse(t, lam, plan, opts, can_read=readable_at, lasting=None):
    """Parse colour RAM, staged $D000+ and the rest separately. Each copies only from itself: by the time a
    range is written, the ones above are unreadable or overwritten.

    With $A000-$BFFF staged, [$A000, $D000) is its own range: it reads its staged bytes, and the $D000+ windows
    where the second phase does not reuse them. Below $A000 the staged bytes are gone, but $C000 still counts:
    that parse runs up to $D000 and keeps only the parts below $A000. lasting: [(start, end)] nothing writes
    once unpacking starts (default: above the output); a window there keeps its staged bytes, so the parse sees
    them in place."""
    end = BASIC_START + len(t)
    staged = lambda x: any(vb <= x < vt for vb, vt, w in plan)
    low = low_staged(plan)
    top = min(end, HIGH_RAM if plan else VIRTUAL_COLOR_RAM)
    imgs = window_images(t, plan)
    lasts = lambda x: any(l0 <= x < l1 for l0, l1 in (lasting or [(end, 0x10000)]))
    kept = {a + i: b for a, d in imgs for i, b in enumerate(d) if a + i < HIGH_RAM and lasts(a + i)}
    # Each range's readable bytes, the same as rd for every address. With the default can_read, they come from a
    # table instead of a call per byte.
    fast = can_read is readable_at
    def mask(a, b, upto=None, ons=(), keys=()):          # readable_at below upto (default b), and ons and keys
        if not fast: return None
        m = bytearray(READABLE[a:min(b, b if upto is None else upto)])
        m += bytes(b - a - len(m))
        for o0, o1 in ons:
            o0, o1 = max(o0, a), min(o1, b)
            if o0 < o1: m[o0 - a:o1 - a] = bytes([1]) * (o1 - o0)
        for x in keys:
            if a <= x < b: m[x - a] = 1
        return bytes(m)
    segs = [(vb, vt) for vb, vt, w in plan]
    if kept:
        data = bytearray(t[:top - BASIC_START]) + bytes(HIGH_RAM - top)
        for a, b in kept.items(): data[a - BASIC_START] = b
        ranges = [(BASIC_START, HIGH_RAM, BASIC_ROM[0] if low else top, lambda x: x in kept or x < top and can_read(x),
                   [(a, d) for a, d in imgs if not lasts(a)], data,
                   mask(BASIC_START, HIGH_RAM, top, keys=kept))]
    else:
        ranges = [(BASIC_START, top, BASIC_ROM[0] if low else None, can_read, imgs, t, mask(BASIC_START, top))]
    if low:
        high = [s for s in plan if s[0] >= HIGH_RAM]
        ranges.append((BASIC_ROM[0], top, None, lambda x: staged(x) or can_read(x),
                       window_images(t, high, windows_used(low)), t, mask(BASIC_ROM[0], top, ons=segs)))
    if plan and end > HIGH_RAM:
        b = min(end, VIRTUAL_COLOR_RAM)
        ranges.append((HIGH_RAM, b, None, staged, None, t, mask(HIGH_RAM, b, HIGH_RAM, segs)))
    if has_color_ram(end):
        ranges.append((VIRTUAL_COLOR_RAM, end, None, lambda x: True, None, t,
                       mask(VIRTUAL_COLOR_RAM, end, ons=[(VIRTUAL_COLOR_RAM, end)])))
    parts = []
    for a, b, stop, rd, extra, data, ok in ranges:
        k = a - BASIC_START
        rom = char_rom_regions(t, a, min(b, end), opts.char_rom, opts.char_min) if a == BASIC_START else ()
        seg = matcher.parse(data[k:b - BASIC_START], lam, base=a, can_read=rd, extra=extra, rom=rom, ok=ok,
                            stop=stop and stop - a, paired=opts.paired, pool_rounds=opts.pool_rounds, pool_seed=opts.pool_seed, cycle_ties=opts.cycle_ties,
                            source_choices=opts.source_choices)
        parts += [Part(s + k, n, kind, src + k if kind == 'match' else src) for s, n, kind, src in seg]
    return split_sources(parts, plan)


def split_sources(parts, plan):
    """Segments are not adjacent in the windows: split matches at segment edges."""
    seg_edges = sorted({x for seg in plan for x in seg[:2]})
    out = []
    for s, n, kind, src in parts:
        if kind == 'match':
            cuts = [e - BASIC_START - src for e in seg_edges if src < e - BASIC_START < src + n]
            for a, b in zip([0] + cuts, cuts + [n]):
                out.append(Part(s + a, b - a, kind, src + a))
        else:
            out.append(Part(s, n, kind, src))
    return out
