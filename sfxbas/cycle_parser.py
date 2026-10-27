"""Size-first DP tie-breaks using integer per-statement cycle estimates.

Primary costs use the exact integer ratio of lambda, with no epsilon.
Copy-length work is constant at each output endpoint. Single parts cost 5633
cycles each; paired statements cost half of a 14198-cycle loop iteration.
Chunk headers, odd paired tails and staging are ranked after layout, using the
full simulator estimate. Ordinary parses remain separate baseline candidates.
"""
import bisect
from collections import deque
import functools
from . import matcher as m

INF = float('inf')
MinTree = m.MinTree


def weights(lam, n):
    """Encode exact primary cost and statement count without changing order.

    Within each parser every statement costs the same secondary cycles. Counting
    statements preserves their order. A path or open statement has at most n+1
    statements, so its secondary cost fits below scale.
    """
    numerator, denominator = float(lam).as_integer_ratio()
    scale = 1 << (n + 1).bit_length()
    return denominator * scale, numerator * scale, scale


@functools.lru_cache(maxsize=32)
def single(t, lam, base, ok, extra, rom, stop, pool=()):
    n = len(t)
    unit, numerator, scale = weights(lam, n)
    lengths, sources, kinds = m.with_pool(m.matches(t, base, ok, extra, rom), pool)
    descriptor = 3*unit+numerator+1
    f, how = [0] + [INF]*n, [None]*(n+1)
    lit, win = deque(), deque()
    for p in range(1, n+1):
        j = p-1
        while lit and f[lit[-1]]-lit[-1]*unit >= f[j]-j*unit: lit.pop()
        lit.append(j)
        while win and f[win[-1]] >= f[j]: win.pop()
        win.append(j)
        while lit[0] < p-m.MAX_PART_LEN: lit.popleft()
        while win[0] < p-m.MAX_PART_LEN: win.popleft()
        b = lit[0]
        f[p], how[p] = f[b]+(p-b)*unit+descriptor, (p-b, 'lit', 0)
        if lengths[p]:
            b = win[bisect.bisect_left(win, p-lengths[p])]
            candidate = f[b]+descriptor
            if candidate < f[p]:
                f[p], how[p] = candidate, (p-b, kinds[p], sources[p]+lengths[p]-(p-b))
    parts, p = [], n if stop is None else stop
    while p:
        length, kind, src = how[p]
        parts.append(m.Part(p-length, length, kind, src)); p -= length
    return tuple(parts[::-1])


@functools.lru_cache(maxsize=32)
def paired(t, lam, base, ok, extra, rom, stop, pool=()):
    n = len(t)
    unit, numerator, scale = weights(lam, n)
    lengths, sources, kinds = m.with_pool(m.matches(t, base, ok, extra, rom), pool)
    far, far_sources = m.far_matches(t, ok)
    statement = 6*unit+2*numerator+1
    f, how = [0] + [INF]*n, [None]*(n+1)
    lit, win = deque(), deque()
    primary_lit, primary_win = deque(), deque()
    primary = [0]*(n+1)
    slots = 6
    open_c, open_l = MinTree(slots*(n+1), slots*m.MAX_PART_LEN), MinTree(slots*(n+1), slots*m.MAX_PART_LEN)
    reach, opened = {}, {}
    def closest(tree, lo, hi):
        while True:
            score, slot = tree.min(lo, hi)
            if slot is None or reach[slot] >= p: return score, slot
            open_c.clear(slot); open_l.clear(slot)
    def open_d(slot, a, kind, src, score, limit):
        opened[slot] = a, kind, src
        open_c.add(slot, (score, slot)); open_l.add(slot, (score-p*unit, slot))
        reach[slot] = limit
    for p in range(1, n+1):
        j = p-1
        while lit and f[lit[-1]]-lit[-1]*unit >= f[j]-j*unit: lit.pop()
        lit.append(j)
        while win and f[win[-1]] >= f[j]: win.pop()
        win.append(j)
        while lit[0] < p-m.MAX_PART_LEN: lit.popleft()
        while win[0] < p-m.MAX_PART_LEN: win.popleft()
        while primary_lit and primary[primary_lit[-1]]-primary_lit[-1]*unit >= primary[j]-j*unit: primary_lit.pop()
        primary_lit.append(j)
        while primary_win and primary[primary_win[-1]] >= primary[j]: primary_win.pop()
        primary_win.append(j)
        while primary_lit[0] < p-m.MAX_PART_LEN: primary_lit.popleft()
        while primary_win[0] < p-m.MAX_PART_LEN: primary_win.popleft()
        la = lit[0]
        best, h = f[la]+(p-la)*unit+statement, ('lone', p-la, 'lit', 0)
        if lengths[p]:
            ca = win[bisect.bisect_left(win, p-lengths[p])]
            candidate = f[ca]+statement
            if candidate < best:
                best, h = candidate, ('lone', p-ca, kinds[p], sources[p]+lengths[p]-(p-ca))
        score, slot = closest(open_l, slots*max(0,p-m.MAX_PART_LEN), slots*p)
        if score+p*unit < best:
            best, h = score+p*unit, ('pair', slot, 'lit', 0)
        if lengths[p]:
            score, slot = closest(open_c, slots*(p-lengths[p]), slots*p)
            if score < best:
                best, h = score, ('pair', slot, kinds[p], sources[p]+lengths[p])
        f[p], how[p] = best, h
        primary[p] = (best//scale)*scale
        if p == n: break
        open_d(slots*p, la, 'lit', 0, f[la]+(p-la)*unit+statement, la+m.MAX_PART_LEN)
        # Keep the latest equal-primary start too: its larger reach can
        # enable a future pair that the lower-cycle start cannot reach.
        legacy_la = primary_lit[0]
        open_d(slots*p+3, legacy_la, 'lit', 0, f[legacy_la]+(p-legacy_la)*unit+statement, legacy_la+m.MAX_PART_LEN)
        need_far = True
        if lengths[p]:
            top = sources[p]+lengths[p]
            offset = top-p if kinds[p]=='match' and top>p else m.MAX_PART_LEN
            open_d(slots*p+1, ca, kinds[p], top-(p-ca), f[ca]+statement, ca+min(offset,m.MAX_PART_LEN))
            legacy_ca = primary_win[bisect.bisect_left(primary_win,p-lengths[p])]
            open_d(slots*p+4, legacy_ca, kinds[p], top-(p-legacy_ca), f[legacy_ca]+statement, legacy_ca+min(offset,m.MAX_PART_LEN))
            need_far = offset < m.MAX_PART_LEN
        if need_far and far[p]:
            a = win[bisect.bisect_left(win,p-far[p])]
            open_d(slots*p+2, a, 'match', far_sources[p]+far[p]-(p-a), f[a]+statement, a+m.MAX_PART_LEN)
            legacy_a = primary_win[bisect.bisect_left(primary_win,p-far[p])]
            open_d(slots*p+5, legacy_a, 'match', far_sources[p]+far[p]-(p-legacy_a), f[legacy_a]+statement, legacy_a+m.MAX_PART_LEN)
    parts, p = [], n if stop is None else stop
    while p:
        h = how[p]
        if h[0] == 'lone':
            length, kind, src = h[1:]
            parts.append(m.Part(p-length,length,kind,src)); p -= length
        else:
            slot, kind, src = h[1:]
            b = slot//slots
            parts.append(m.Part(b,p-b,kind,src-(p-b) if kind!='lit' else 0))
            a, kind, src = opened[slot]
            parts.append(m.Part(a,b-a,kind,src)); p = a
    return tuple(parts[::-1])
