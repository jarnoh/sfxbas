"""Optimal parse of the output into string-copy parts.

The decruncher writes the output from the top down: each part is copied (MOVSTR, top-down) to just below what is
already there. A source above the destination is an LZ copy and may overlap; a source below is a plain move.
Literals live in the stream.

Cost = bytes + lambda * parts. lambda is what one part is worth in bytes: low = small and slow.
"""
import bisect
import functools
from collections import deque, namedtuple

MAX_PART_LEN = 255

# kind: 'lit' (src unused), 'match' (src = index in t), 'rom' (src = ROM offset), 'skip' (src = hole index),
# 'zero' ($0800, copied from the decruncher's end marker)
Part = namedtuple('Part', 'start length kind src')


@functools.lru_cache(maxsize=8)
def reversed_suffixes(t):
    """Suffix array of t reversed, cut at MAX_PART_LEN + 1 bytes, and the common prefix with each one's
    predecessor in it (at most MAX_PART_LEN)."""
    n = len(t)
    r = bytes(t[::-1])
    sa = sorted(range(n), key=lambda i: r[i:i + MAX_PART_LEN + 1])
    def lcp2(a, b):
        lo, hi = 0, min(MAX_PART_LEN, n - a, n - b)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if r[a:a + mid] == r[b:b + mid]: lo = mid
            else: hi = mid - 1
        return lo
    return sa, [0] + [lcp2(sa[k - 1], sa[k]) for k in range(1, n)]


FAR = MAX_PART_LEN   # a source this far above is past anything a pair writes
FAR_STEPS = 48       # suffix array neighbours tried on each side


@functools.lru_cache(maxsize=16)
def far_matches(t, ok):
    """fm[p], fsrc[p]: the longest readable part ending at p that copies from FAR or more bytes above its top, so
    it can sit below anything in a pair (paired_parse). Searched near p's suffix in the suffix array only."""
    n = len(t)
    sa, lcp = reversed_suffixes(t)
    rank = [0] * n
    for k, q in enumerate(sa): rank[q] = k
    rr = [0] * (n + 1)
    for x in range(1, n + 1):
        rr[x] = rr[x - 1] + 1 if ok[x - 1] else 0
    fm, fsrc = [0] * (n + 1), [0] * (n + 1)
    for p in range(3, n + 1):
        q, best, bx = n - p, 2, 0                        # reversed position of t[..p]; 3 bytes or more
        for step in (-1, 1):
            k, run = rank[q], MAX_PART_LEN
            for _ in range(FAR_STEPS):
                run = min(run, lcp[k] if step < 0 else lcp[k + 1] if k + 1 < n else 0)
                k += step
                if run <= best or not 0 <= k < n: break
                x = n - sa[k]                            # the source's top
                if x >= p + FAR:
                    L = min(run, rr[x], p)
                    if L > best: best, bx = L, x
        if bx: fm[p], fsrc[p] = best, bx - best
    return tuple(fm), tuple(fsrc)


@functools.lru_cache(maxsize=8)
def longest_matches(t):
    """m[p], src[p]: the longest part ending at p that can copy from the output above it.

    Reverse the data and a top-down copy becomes a plain LZ77 copy from earlier on, so this is the longest
    previous factor: suffix array plus the nearest smaller position on each side (Crochemore-Ilie).
    t must be bytes. Cached: build() parses the same input many times. Do not modify the result."""
    n = len(t)
    sa, lcp = reversed_suffixes(t)
    best, bsrc = [0] * n, [0] * n
    for order in (range(n), range(n - 1, -1, -1)):
        st = []                                          # [position, min lcp from it to here]
        prev = None
        for k in order:
            if prev is not None and st:
                st[-1][1] = min(st[-1][1], lcp[max(k, prev)])
            while st and st[-1][0] > sa[k]:
                v, g = st.pop()
                if st: st[-1][1] = min(st[-1][1], g)
            if st and st[-1][1] > best[sa[k]]:
                best[sa[k]], bsrc[sa[k]] = st[-1][1], st[-1][0]
            st.append([sa[k], MAX_PART_LEN])
            prev = k
    m, src = [0] * (n + 1), [0] * (n + 1)
    for q in range(n):                                   # back to forward indices
        m[n - q] = best[q]
        src[n - q] = n - bsrc[q] - best[q]
    return tuple(m), tuple(src)


def common_suffix(a, pa, b, pb, lo, hi):
    """Longest L in [lo, hi] with a[pa-L:pa] == b[pb-L:pb]; the first lo bytes are known to match."""
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[pa - mid:pa] == b[pb - mid:pb]: lo = mid
        else: hi = mid - 1
    return lo


# Sources are indexed by their last 3 bytes. A source has to match one byte more than the best so far, so in a big
# bucket (runs of one byte) the longest key within that skips the sources that would fail anyway; as the best grows,
# the scan moves on to a longer key. Those keys are hashes: a collision only adds a source the byte compare rejects.
KEY_LENGTHS = (3, 4, 8, 16, 32, 64, 128, 255)
SPLIT_MIN = 64                                           # smaller buckets are scanned as they are


# KEY_FOR[need]: the longest key length for sources that must match need bytes (at most p: index with min(need, p))
KEY_FOR = [max([3] + [k for k in KEY_LENGTHS if k <= need]) for need in range(MAX_PART_LEN + 2)]


def readable_index(t, rr):
    """3-byte suffix -> (readable lengths longest first, {readable length: sorted source tops}, source count)."""
    index = {}
    for x in range(3, len(t) + 1):
        if rr[x] >= 3:
            index.setdefault(bytes(t[x - 3:x]), {}).setdefault(min(MAX_PART_LEN, rr[x]), []).append(x)
    return {key: (sorted(g, reverse=True), g, sum(map(len, g.values()))) for key, g in index.items()}


def parse(t, lam, base, can_read, extra=None, rom=(), stop=None, paired=False, pool_rounds=1, pool_seed=None, cycle_ties=False, source_choices=False, ok=None):
    """Cheapest parts for t at address base, bottom up.

    A literal costs 3 + L + lam, a copy 3 + lam. paired: priced as pair statements instead (paired_parse).
    Sources must pass can_read(), since ROM and I/O shadow RAM.
    extra: [(address, bytes)], sorted and disjoint, of memory holding something else while t is written (staging
    windows); usable while it lies wholly below the part.
    rom: [(start, end, offset)] stretches of t that equal a constant image (the char ROM), copied as kind 'rom'.
    stop: return the parts for t[:stop] only; t above still serves as a source.

    Discover a pool from an initial single parse (at pool_seed or lam), then
    reparse pool_rounds times. Bytes inside an earlier literal below cost no bytes
    (pool_matches); stream.share_literals points there if the owner survives.
    Different rounds are candidates: their final stored size need not decrease.
    cycle_ties selects the exact-primary/integer-cycle DP; build() retains the
    legacy parse as a complete candidate because layout can change final size."""
    ok = bytes(can_read(base + i) for i in range(len(t))) if ok is None else ok   # ok: precomputed, 0/1 per byte
    extra = tuple((a, bytes(d)) for a, d in extra) if extra else None
    run = paired_parse if paired else cached_parse
    seed_run = cached_parse
    if cycle_ties:
        from .cycle_parser import single, paired as cycle_paired
        run = cycle_paired if paired else single
        seed_run = single
    if paired and source_choices:
        from .source_parser import paired as source_paired, legacy_paired
        run = source_paired if cycle_ties else legacy_paired
    args = bytes(t), lam, base, ok, extra, tuple(rom), stop
    seed_args = args if pool_seed is None else (args[0], pool_seed, *args[2:])
    seed = seed_run(*seed_args)
    for _ in range(pool_rounds):
        result = run(*args, pool_matches(args[0], seed))
        if result == seed: break
        seed = result
    return list(result)


def pool_matches(t, parts, min_len=2):
    """Longest suffix at each literal endpoint found inside a lower literal.

    Only literal boundaries matter, so different parses with the same literal
    owners share this immutable result. Sources must fit wholly in one owner.
    """
    literals = tuple(sorted((s, n) for s, n, kind, src in parts if kind == 'lit'))
    return literal_pool_matches(bytes(t), literals, min_len)


class ActiveRanks:
    """Fenwick index of active suffix ranks; supports nearest active neighbours."""
    def __init__(self, n):
        self.tree = [0] * (n + 1)
        self.total = 0
        self.step = 1 << max(0, n.bit_length() - 1)

    def add(self, rank):
        self.total += 1
        i, tree = rank + 1, self.tree
        while i < len(tree):
            tree[i] += 1
            i += i & -i

    def count(self, end):
        total, tree = 0, self.tree
        while end:
            total += tree[end]
            end -= end & -end
        return total

    def kth(self, k):
        """Zero-based rank of the k-th active entry (k is one-based)."""
        i, step, tree = 0, self.step, self.tree
        while step:
            j = i + step
            if j < len(tree) and tree[j] < k:
                i, k = j, k - tree[j]
            step >>= 1
        return i


@functools.lru_cache(maxsize=32)
def literal_pool_matches(t, literals, min_len):
    """Index reversed suffixes truncated at their owning literal's left edge.

    Process owners bottom up, activating their suffixes only after querying the
    whole owner. Its longest common prefix with an active key occurs at a nearest
    active neighbour in sorted order. Truncation enforces owner boundaries; plain
    suffixes of the whole input could falsely cross from one literal to another.
    """
    if min_len > MAX_PART_LEN: return ()
    prefix_len = max(0, min_len)
    entries = sorted((t[max(a, p - MAX_PART_LEN):p][::-1], i, p)
                     for i, (a, n) in enumerate(literals) for p in range(a + max(1, min_len), a + n + 1))
    owners = [[] for _ in literals]
    keys = []
    for rank, (key, owner, p) in enumerate(entries):
        keys.append(key)
        owners[owner].append((p, rank))
    active, out = ActiveRanks(len(keys)), []
    for queries in owners:
        for p, rank in sorted(queries):
            count = active.count(rank + 1)
            neighbours = ([active.kth(count)] if count else [])
            if count < active.total: neighbours.append(active.kth(count + 1))
            key, length = keys[rank], 0
            for source in neighbours:
                other = keys[source]
                if key[:prefix_len] == other[:prefix_len]:
                    length = max(length, common_prefix(key, other, prefix_len))
                    if length == len(key): break
            if length >= min_len: out.append((p, length))
        for p, rank in queries: active.add(rank)
    return tuple(out)


def with_pool(found, pool):
    """matches() with pool_matches() where those are longer: kind 'lit', no source."""
    if not pool: return found
    m, msrc, kind = (list(x) for x in found)
    for p, L in pool:
        if L > m[p]: m[p], msrc[p], kind[p] = L, 0, 'lit'
    return m, msrc, kind


@functools.lru_cache(maxsize=32)
def cached_parse(t, lam, base, ok, extra, rom, stop, pool=()):
    """parse() with hashable arguments: build() parses the same ranges for several staging plans."""
    n = len(t)
    m, msrc, kind = with_pool(matches(t, base, ok, extra, rom), pool)
    # f[p] = cheapest cost of t[:p]. Two sliding-window minima make it O(n).
    INF = float('inf')
    f, how = [0.0] + [INF] * n, [None] * (n + 1)
    lit = deque()                                        # argmin f[j] - j: best literal start
    win = deque()                                        # argmin f[j], monotone: best copy start
    for p in range(1, n + 1):
        j = p - 1
        while lit and f[lit[-1]] - lit[-1] >= f[j] - j: lit.pop()
        lit.append(j)
        while win and f[win[-1]] >= f[j]: win.pop()
        win.append(j)
        while lit[0] < p - MAX_PART_LEN: lit.popleft()
        while win[0] < p - MAX_PART_LEN: win.popleft()
        b = lit[0]
        f[p], how[p] = f[b] + (p - b) + 3 + lam, (p - b, 'lit', 0)
        if m[p]:
            lo, hi = 0, len(win) - 1                     # first start the match can reach
            while lo < hi:
                mid = (lo + hi) // 2
                if win[mid] >= p - m[p]: hi = mid
                else: lo = mid + 1
            b = win[lo]
            if f[b] + 3 + lam < f[p]:
                f[p], how[p] = f[b] + 3 + lam, (p - b, kind[p], msrc[p] + (m[p] - (p - b)))
    parts, p = [], n if stop is None else stop
    while p:
        L, kind, sidx = how[p]
        parts.append(Part(p - L, L, kind, sidx)); p -= L
    return tuple(parts[::-1])


class MinTree:
    """Range minimum over a fixed number of slots, with (value, payload) entries. Ties go to the entry a node
    already holds (add) or to the left child (clear), and min() takes the first it meets.

    span: no query covers more slots. min() reads only nodes inside its range, so none over span slots, and a
    node depends only on what happened to its own slots: add() and clear() leave the bigger ones stale.
    Values and payloads are kept in two lists, which is faster than a list of pairs."""
    def __init__(self, n, span=None):
        self.size = 1 << max(1, (n - 1).bit_length())
        self.val = [float('inf')] * (2 * self.size)
        self.pay = [None] * (2 * self.size)
        self.top = max(1, self.size >> (span.bit_length() - 1)) if span else 1   # nodes below index top: never read

    def add(self, i, x):
        """Fill an empty slot: only the parents it improves change."""
        val, pay, top = self.val, self.pay, self.top
        c, s = x
        i += self.size
        val[i] = c; pay[i] = s
        i >>= 1
        while i >= top and c < val[i]:
            val[i] = c; pay[i] = s
            i >>= 1

    def clear(self, i):
        val, pay, top = self.val, self.pay, self.top
        i += self.size
        val[i] = float('inf'); pay[i] = None
        i >>= 1
        while i >= top:
            l = 2 * i
            if val[l] <= val[l + 1]: val[i] = val[l]; pay[i] = pay[l]
            else: val[i] = val[l + 1]; pay[i] = pay[l + 1]
            i >>= 1

    def min(self, lo, hi):
        val = self.val
        best, at = float('inf'), 0
        lo += self.size; hi += self.size
        while lo < hi:
            if lo & 1:
                if val[lo] < best: best, at = val[lo], lo
                lo += 1
            if hi & 1:
                hi -= 1
                if val[hi] < best: best, at = val[hi], hi
            lo >>= 1; hi >>= 1
        return best, self.pay[at] if at else None


@functools.lru_cache(maxsize=32)
def paired_parse(t, lam, base, ok, extra, rom, stop, pool=()):
    """cached_parse() priced as the paired decruncher runs it: one statement copies an upper part u and the
    part d just below it, or one part alone, for 6 bytes + 2 lam either way. Together at most 255 bytes, and d
    must not read what u writes (stream.can_pair): its source starts at or above u's top.

    f[p]: cheapest t[:p] that ends with a whole statement. An open d = t[a:b] waits for its u; it can reach up
    to a + its offset (a copy's offset is the same for every start a), at most a + 255. For each b, one open d
    of each kind is kept, from its cheapest start: a literal, a copy, and a copy from a far source.
    The first two are the lone parts ending at b."""
    n = len(t)
    m, msrc, kind = with_pool(matches(t, base, ok, extra, rom), pool)
    fm, fsrc = far_matches(t, ok)
    INF = float('inf')
    S = 6 + 2 * lam
    f, how = [0.0] + [INF] * n, [None] * (n + 1)
    lit, win = deque(), deque()
    W = 3                                                # slots per b: literal, copy, far copy
    open_c, open_l = (MinTree(W * (n + 1), W * MAX_PART_LEN) for _ in range(2))   # cost; cost - b for a literal u
    reach = {}                                           # slot: the highest top its u can have
    opened = {}                                          # slot: (a, kind, src)
    def closest(tree, lo, hi):                           # cheapest open d in slots [lo, hi) that reaches p
        while True:
            c, slot = tree.min(lo, hi)
            if slot is None or reach[slot] >= p: return c, slot
            open_c.clear(slot); open_l.clear(slot)
    def open_d(slot, a, k, src, cost, r):
        opened[slot] = (a, k, src)
        open_c.add(slot, (cost, slot)); open_l.add(slot, (cost - p, slot))
        reach[slot] = r
    for p in range(1, n + 1):
        j = p - 1
        while lit and f[lit[-1]] - lit[-1] >= f[j] - j: lit.pop()
        lit.append(j)
        while win and f[win[-1]] >= f[j]: win.pop()
        win.append(j)
        while lit[0] < p - MAX_PART_LEN: lit.popleft()
        while win[0] < p - MAX_PART_LEN: win.popleft()
        # Lone part, as cached_parse.
        la = lit[0]
        best, h = f[la] + (p - la) + S, ('lone', p - la, 'lit', 0)
        if m[p]:
            ca = win[bisect.bisect_left(win, p - m[p])]
            if f[ca] + S < best:
                best, h = f[ca] + S, ('lone', p - ca, kind[p], msrc[p] + (m[p] - (p - ca)))
        # Close a pair: u = t[b:p] over an open d ending at b.
        c, slot = closest(open_l, W * max(0, p - MAX_PART_LEN), W * p)
        if c + p < best:
            best, h = c + p, ('pair', slot, 'lit', 0)
        if m[p]:
            c, slot = closest(open_c, W * (p - m[p]), W * p)
            if c < best:
                best, h = c, ('pair', slot, kind[p], msrc[p] + m[p])
        f[p], how[p] = best, h
        # Open a d = t[a:p]. Its u starts at p.
        if p == n: break
        open_d(W * p, la, 'lit', 0, f[la] + (p - la) + S, la + MAX_PART_LEN)
        far = True
        if m[p]:
            top = msrc[p] + m[p]
            off = top - p if kind[p] == 'match' and top > p else MAX_PART_LEN
            open_d(W * p + 1, ca, kind[p], top - (p - ca), f[ca] + S, ca + min(off, MAX_PART_LEN))
            far = off < MAX_PART_LEN
        if far and fm[p]:                                # a farther source reaches the whole 255 bytes
            a = win[bisect.bisect_left(win, p - fm[p])]
            open_d(W * p + 2, a, 'match', fsrc[p] + fm[p] - (p - a), f[a] + S, a + MAX_PART_LEN)
    parts, p = [], n if stop is None else stop
    while p:
        h = how[p]
        if h[0] == 'lone':
            L, k, src = h[1:]
            parts.append(Part(p - L, L, k, src)); p -= L
            continue
        slot, k, src = h[1:]
        b = slot // W
        parts.append(Part(b, p - b, k, src - (p - b) if k != 'lit' else 0))
        a, dk, dsrc = opened[slot]
        parts.append(Part(a, b - a, dk, dsrc)); p = a
    return tuple(parts[::-1])


@functools.lru_cache(maxsize=16)
def readable_matches(t, ok):
    """Output-source matches shared by plans with the same bytes and readability."""
    n = len(t)
    m, msrc = (list(x) for x in longest_matches(t))
    rr = [0] * (n + 1)                               # readable bytes ending at x-1
    for x in range(1, n + 1):
        rr[x] = rr[x - 1] + 1 if ok[x - 1] else 0
    index, split, short_index = None, {}, {}
    def readable_split(key, k, p):
        """index[key] cut down to the sources whose last k bytes may equal t[p-k:p]."""
        if k == 3: return index[key]
        if (key, k) not in split:
            sp = {}
            for rl, xs in index[key][1].items():
                for x in xs:
                    if x >= k: sp.setdefault(hash(bytes(t[x - k:x])), {}).setdefault(rl, []).append(x)
            split[key, k] = {h: (sorted(g, reverse=True), g, 0) for h, g in sp.items()}
        return split[key, k].get(hash(bytes(t[p - k:p])), ((), {}, 0))
    for p in range(1, n + 1):
        # Keep the top of the match and cut it where the source becomes unreadable.
        top = msrc[p] + m[p]
        L = min(m[p], rr[top])
        best, bsrc = L, top - L
        if L < m[p] and p >= 3:
            # Cut short: a readable copy of the same bytes further up may be longer.
            if index is None: index = readable_index(t, rr)
            key = bytes(t[p - 3:p])
            rls, groups, count = index.get(key, ((), {}, 0))
            big, k = count >= SPLIT_MIN, 3
            if big:
                k = KEY_FOR[min(best + 1, p)]
                rls, groups, count = readable_split(key, k, p)
            for readable_len in rls:
                # Restricting sources cannot beat the unrestricted longest match.
                cap = min(readable_len, p, m[p])
                # Remaining sources cannot improve the result. No candidate limit:
                # the only long readable match may be far beyond the nearest ones.
                if cap <= best: break
                if big and cap == MAX_PART_LEN:
                    # The scan would end at the first source matching all 255 bytes: look it up directly.
                    full = readable_split(key, cap, p)[1].get(readable_len, ())
                    i = bisect.bisect_right(full, p)
                    while i < len(full) and t[p - cap:p] != t[full[i] - cap:full[i]]: i += 1   # a hash collision
                    if i < len(full):
                        best, bsrc = cap, full[i] - cap
                        break
                sources = groups.get(readable_len, ())
                i = bisect.bisect_right(sources, p)
                while i < len(sources) and best < cap:
                    x = sources[i]; i += 1
                    if t[p - best - 1:p] != t[x - best - 1:x]: continue
                    best = common_suffix(t, p, t, x, best + 1, cap)
                    bsrc = x - best
                    if big and KEY_FOR[min(best + 1, p)] != k:
                        k = KEY_FOR[min(best + 1, p)]
                        groups = readable_split(key, k, p)[1]
                        sources = groups.get(readable_len, ()); i = bisect.bisect_right(sources, x)
        # The 3-byte index cannot recover shorter readable alternatives. They can still save
        # bytes between other copies. For each suffix only the farthest readable top is needed.
        for length in range(min(2, m[p]), best, -1):
            if length not in short_index:
                short_index[length] = {bytes(t[x - length:x]): x for x in range(length, n + 1)
                                       if rr[x] >= length}
            x = short_index[length].get(bytes(t[p - length:p]), 0)
            if x > p:
                best, bsrc = length, x - length
                break
        m[p], msrc[p] = best, bsrc
    return tuple(m), tuple(msrc)


def common_prefix(a, b, lo=0):
    """Length of the common prefix of a and b; the first lo bytes are known to match."""
    hi = min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]: lo = mid
        else: hi = mid - 1
    return lo


@functools.lru_cache(maxsize=16)
def window_index(extra):
    """The staging window sources, sorted by the bytes before them read backwards (a part's length at most).

    Sources sharing a long suffix are then neighbours, as in a suffix array. Returns
      keys    distinct reversed suffixes, sorted
      tops    for each key, the top addresses of its sources, ascending
      near    near[i]: common prefix of keys[i - 1] and keys[i]
      bucket  {first 3 key bytes: (start, end) range in keys}
      short   {1 or 2 bytes: ascending tops of sources ending with them}"""
    entries = sorted((data[max(0, x - MAX_PART_LEN):x][::-1], addr + x)
                     for addr, data in extra for x in range(3, len(data) + 1))
    keys, tops = [], []
    for key, top in entries:
        if keys and keys[-1] == key: tops[-1].append(top)
        else: keys.append(key); tops.append([top])
    near = [0] + [common_prefix(a, b) for a, b in zip(keys, keys[1:])]
    bucket = {}
    for i, key in enumerate(keys):
        bucket[key[:3]] = (bucket.get(key[:3], (i,))[0], i + 1)
    short = {}
    for length in (1, 2):
        for addr, data in extra:
            for x in range(length, len(data) + 1):
                short.setdefault(data[x - length:x], []).append(addr + x)
    for v in short.values(): v.sort()
    return keys, tops, near, bucket, short


def window_match(t, p, end, need, index):
    """(length, top) of the best window source for the part t[..p] whose top address is end: the longest that
    stays wholly below the part, the highest of equals. (0, 0) if none reaches need (3 or more).

    Walk out from where t's reversed suffix sorts among the keys. The common prefix only shrinks on the way, so
    each side stops once it falls below need, which rises to the best length so far."""
    keys, tops, near, bucket, _ = index
    lo, hi = bucket.get(t[p - 3:p][::-1], (0, 0))
    if lo == hi: return 0, 0
    q = t[max(0, p - MAX_PART_LEN):p][::-1]
    i = bisect.bisect_left(keys, q, lo, hi)
    best = (0, 0)
    for j, step in ((i - 1, -1), (i, 1)):
        run = None                                       # common prefix of q and keys[j]
        while lo <= j < hi:
            if run is None:
                if q[:need] != keys[j][:need]: break
                run = common_prefix(q, keys[j], need)
            else:
                run = min(run, near[j + 1] if step < 0 else near[j])
                if run < need: break
            ts = tops[j]                                 # the highest top a whole run fits below, else the lowest
            k = bisect.bisect_right(ts, end - run) - 1
            cand = (run, ts[k]) if k >= 0 else (end - ts[0], ts[0]) if ts[0] <= end - 3 else (0, 0)
            if cand > best: best, need = cand, max(need, cand[0])
            j += step
    return best


@functools.lru_cache(maxsize=16)
def matches(t, base, ok, extra, rom):
    """Copy choices, adding plan-specific staging and ROM to shared output matches."""
    n = len(t)
    m, msrc = (list(x) for x in readable_matches(t, ok))
    if extra:
        index = window_index(extra)
        for p in range(3, n + 1):
            if m[p] >= min(MAX_PART_LEN, p): continue
            L, top = window_match(t, p, base + p, max(m[p] + 1, 3), index)
            if L > m[p]: m[p], msrc[p] = L, top - L - base
        # One- and two-byte copies: the suffix itself is the whole match.
        short = index[4]
        for length in (2, 1):
            for p in range(length, n + 1):
                if m[p] >= length: continue
                sources = short.get(t[p - length:p], ())
                i = bisect.bisect_right(sources, base + p - length) - 1
                if i >= 0: m[p], msrc[p] = length, sources[i] - length - base
    kind = ['match'] * (n + 1)
    for a, b, off in rom:
        for p in range(a + 1, b + 1):
            L = min(p - a, MAX_PART_LEN)
            if L > m[p]: m[p], msrc[p], kind[p] = L, off + p - a - L, 'rom'
    return tuple(m), tuple(msrc), tuple(kind)
