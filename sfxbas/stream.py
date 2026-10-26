"""Pair parts, form chunks, and lay out the packed BASIC arrays and literals."""
import bisect
import functools
import itertools
import math
from collections import namedtuple
from .basic import BAR_WIDTH, Vars, header_fields, hi, integer_array, lo, poke_bar, string_array
from .matcher import MAX_PART_LEN, Part
from .memory import BASIC_START, CHAR_ROM_ADDR, real_address, real_fretop, windows_used
from .simulator import chunk_cycles


def stale(part):
    """Reads a staging window from below, before the output gets there."""
    return part.kind == 'match' and part.src + part.length <= part.start


def can_pair(u, d):
    """Can u and d share one statement? CAT copies d first, so d must not clobber what u reads, and the two
    must fit in one 255-byte string."""
    if u.length + d.length > 255: return False
    if stale(u) and u.src + u.length > d.start: return False
    return d.kind in ('lit', 'rom', 'zero') or not (d.src < u.start + u.length and d.src + d.length > u.start)


def literal_needs(u, d):
    """The pair's literals, in reading order, each with how much output may already be written over it.
    u's literal has to survive d's copy."""
    ld = d.length if d else 0
    return (([(d, u.length)] if d and d.kind == 'lit' else [])
            + ([(u, u.length + ld if ld else 0)] if u.kind == 'lit' else []))


def resourcer(t, parts, plan):
    """cached_resourcer() for this parse: the same function for the same parse, so paired_runs() can cache."""
    return cached_resourcer(bytes(t), tuple(parts), tuple(plan))


@functools.lru_cache(maxsize=64)
def cached_resourcer(t, parts, plan):
    """f(u, d): d copying the same bytes from elsewhere so it can pair with u, or None.

    The parse keeps one source per part. When d reads what u writes, another copy of its bytes above u will do.
    The new source is where the old one is: plain RAM below $A000 that nothing else holds (no staging window),
    or the same staged segment, whose window holds it in one piece. Never in a hole."""
    t = bytes(t)
    holes = [(p.start, p.start + p.length) for p in parts if p.kind == 'skip']
    busy = sorted([(a - BASIC_START, b - BASIC_START) for a, b in windows_used(plan)] + holes)
    segs = [(vb - BASIC_START, vt - BASIC_START) for vb, vt, w in plan]
    def region(a, n):
        """(start, end) of the region t[a:a+n] lies in, or None."""
        if any(h0 < a + n and a < h1 for h0, h1 in holes): return None
        if a + n <= 0xA000 - BASIC_START:
            return None if any(b0 < a + n and a < b1 for b0, b1 in busy) else (0, 0xA000 - BASIC_START)
        return next((s for s in segs if s[0] <= a and a + n <= s[1]), None)
    def f(u, d):
        if d.kind != 'match' or stale(d): return None
        r = region(d.src, d.length)
        if not r: return None
        sub = t[d.start:d.start + d.length]
        x = t.find(sub, u.start + u.length, r[1])
        while x >= 0:
            if region(x, d.length) == r: return d._replace(src=x)
            x = t.find(sub, x + 1, r[1])
        return None
    return f


MAX_PER_CHUNK = Vars.MAX_PER_CHUNK
SMALL_TAIL_PAIRS = 4  # tiny bottom chunks go single: 2 array headers fewer where space is tight
SINGLE_RUN_MIN = 12   # unpairable runs this long get their own single chunk (a chunk head costs ~30k cycles)


# pairs: [(upper part, lower part or None)]; skip: 1 + the hole jumped first, or 0; single: t$=p$(k)+"" chunk
Chunk = namedtuple('Chunk', 'pairs skip single')


def chunks_for(parts, paired=False, tail=None, per=None, small=None, resource=None):
    """Chunks, top down. resource: resourcer(), to pair parts the parse's sources keep apart.

    A chunk is read until its last part is done, and the last ones are read when the output is nearly
    complete. So the bottom run is cut into doubling sizes: tail, 2 tail, 4 tail, ... up to per."""
    per = per or Vars.MAX_PER_CHUNK
    tail = per if tail is None else min(tail, per)
    runs = paired_runs(tuple(parts), paired, resource)
    chunks = []
    for r, (pairs, single, skip) in enumerate(runs):
        if not pairs: continue
        segs = {i for i in range(1, len(pairs)) if single[i] != single[i - 1]}
        if r == len(runs) - 1:                           # bottom run
            cuts, pos, size = {len(pairs)}, len(pairs), tail
            while pos > 0:
                pos -= size; cuts.add(max(pos, 0)); size = min(2 * size, per)
        else:
            cuts = {len(pairs)}
            bounds = sorted(segs | {0, len(pairs)})
            for a, b in zip(bounds, bounds[1:]):
                cuts |= set(range(a + per, b, per))
        cuts |= segs
        prev = 0
        for c in sorted(x for x in cuts if x > 0):
            ch, sg = list(pairs[prev:c]), single[prev]
            if paired and not sg and r == len(runs) - 1 and len(ch) <= (SMALL_TAIL_PAIRS if small is None else small):
                ch, sg = [(x, None) for u, d in ch for x in (u, d) if x], True
            chunks.append(Chunk(ch, skip if prev == 0 else 0, sg)); prev = c
    return chunks


@functools.lru_cache(maxsize=64)
def paired_runs(parts, paired, resource):
    """chunks_for() before cutting: per run between holes, top down, (pairs, single per pair, skip). The same for
    every tail, per and small, so the layout search pairs each parse once. resource is held, so its id stays."""
    runs, skips = [[]], [0]                              # runs between holes
    for part in reversed(parts):
        if part.kind == 'skip':
            runs.append([]); skips.append(part.src + 1)
        else:
            runs[-1].append(part)
    out = []
    for run, skip in zip(runs, skips):
        pairs, i = [], 0
        while i < len(run):
            u = run[i]
            d = run[i + 1] if paired and i + 1 < len(run) and can_pair(u, run[i + 1]) else None
            if paired and not d and resource and i + 1 < len(run) and u.length + run[i + 1].length <= 255:
                d = resource(u, run[i + 1])
                if d and not can_pair(u, d): d = None
            if paired and not d and i + 1 < len(run) and u.length < 255 < u.length + run[i + 1].length:
                n = run[i + 1]                           # too long together: split n, pair its top
                k = 255 - u.length
                top = Part(n.start + n.length - k, k, n.kind, n.src + n.length - k)
                if not can_pair(u, top) and resource: top = resource(u, top) or top
                if can_pair(u, top):
                    pairs.append((u, top)); run[i + 1] = n._replace(length=n.length - k); i += 1
                    continue
            pairs.append((u, d)); i += 2 if d else 1
        single = [not paired] * len(pairs)
        if paired:
            i = 0
            while i < len(pairs):
                j = i
                while j < len(pairs) and pairs[j][1] is None: j += 1
                if j - i >= SINGLE_RUN_MIN: single[i:j] = [True] * (j - i)
                i = max(j, i + 1)
        out.append((tuple(pairs), tuple(single), skip))
    return tuple(out)


def split_long(chunks, cost, n):
    """Cut every chunk that takes over 1/n of the estimated cycles into pieces of about equal time, so the bar
    (drawn once per chunk) moves at least that often. Each piece costs a chunk header and its cycles."""
    limit = sum(cost(*c) for c in chunks) / n
    out = []
    for ch, skip, single in chunks:
        k = min(len(ch), math.ceil(cost(ch, skip, single) / limit))
        w = [cost([x], 0, single) for x in ch]           # per statement; the overhead in each is the same
        step, acc, start, pieces = sum(w) / k, 0, 0, []
        for i, x in enumerate(w[:-1]):
            acc += x
            if acc >= step * (len(pieces) + 1) and len(pieces) < k - 1:
                pieces.append(ch[start:i + 1]); start = i + 1
        pieces.append(ch[start:])
        out += [Chunk(p, skip if j == 0 else 0, single) for j, p in enumerate(pieces)]   # the hole jump goes first
    return out


def share_literals(t, needs):
    """{literal start: (start of the literal holding its bytes, offset in it)} for literals whose bytes are
    wholly inside one read later. That one is still in the stream when the first is read, below it, so the
    first needs no bytes of its own. needs: Stream.needs, in reading order."""
    return dict(shared_in(bytes(t), tuple((key[1], L) for key, L, y in needs if key[0] == 'l')))


@functools.lru_cache(maxsize=256)
def shared_in(t, lits):
    """share_literals() for literals (start, length) in reading order. Most layouts of one parse read the same
    literals in the same order, only cut into other chunks."""
    shared, pool, starts, owners = {}, bytearray(), [], []
    for a, L in reversed(lits):                          # last read first: the pool holds those read later
        s = bytes(t[a:a + L])
        x = pool.find(s)
        while x >= 0:
            i = bisect.bisect_right(starts, x) - 1
            if x + L <= starts[i] + owners[i][1]:
                b = owners[i][0]
                shared[a] = (b, x - starts[i])
                break
            x = pool.find(s, x + 1)
        else:
            starts.append(len(pool)); owners.append((a, L)); pool += s
    return shared


OVERLAP_MAX = 64   # longest literal overlap overlap_literals() looks for
OVERLAP_REACH = 512  # ... with groups whose owner is at most this many literal bytes later in the stream


def overlap_literals(t, needs):
    """share_literals(), but a literal that is no part of a later one can still overlap one: their bytes are
    stored once, as one group, in the slot of the literal read last. Every other one reads its bytes from
    lower than its own slot would be, where the output gets later.
    Returns (shared, store): shared as in share_literals, offsets into the group; store: {group owner:
    (group bytes, the owner's offset in them)}."""
    return overlapped_in(bytes(t), tuple((key[1], L) for key, L, y in needs if key[0] == 'l'))


@functools.lru_cache(maxsize=256)
def overlapped_in(t, lits):
    """overlap_literals() for literals (start, length) in reading order."""
    groups, owner, members, at = [], [], [], []          # per group: bytes, owner start, {start: offset}, where
    done = 0                                             # literal bytes read after this one
    heads, tails = {}, {}                                # k first / last bytes -> groups
    def index(g):
        b = groups[g]
        for k in range(1, min(len(b), OVERLAP_MAX) + 1):
            heads.setdefault(bytes(b[:k]), set()).add(g)
            tails.setdefault(bytes(b[-k:]), set()).add(g)
    pool, starts = b'', []                               # all groups, for searching; rebuilt after changes
    def inside(s):                                       # (group, offset) of s wholly in one group, or None
        nonlocal pool, starts
        if len(starts) != len(groups) or sum(map(len, groups)) != len(pool):
            starts = list(itertools.accumulate((len(b) for b in groups[:-1]), initial=0))
            pool = b''.join(groups)
        x = pool.find(s)
        while x >= 0:
            g = bisect.bisect_right(starts, x) - 1
            if x + len(s) <= starts[g] + len(groups[g]): return g, x - starts[g]
            x = pool.find(s, x + 1)
        return None
    for a, L in reversed(lits):                          # last read first: groups hold those read later
        s = bytes(t[a:a + L])
        found = inside(s)
        if found:
            g, members[g][a] = found
            continue
        best = None                                      # (overlap, group, s goes after the group)
        near = lambda g: done - at[g] <= OVERLAP_REACH  # everything in between moves up this much
        for k in range(min(L - 1, OVERLAP_MAX), 0, -1):
            after = [g for g in tails.get(s[:k], ()) if near(g) and groups[g].endswith(s[:k])]
            before = [g for g in heads.get(s[-k:], ()) if near(g) and groups[g].startswith(s[-k:])]
            if after or before:
                best = (k, after[0], True) if after else (k, before[0], False)
                break
        if best is None:
            groups.append(bytearray(s)); owner.append(a); members.append({a: 0}); at.append(done)
            index(len(groups) - 1); done += L
            continue
        k, g, after = best
        if after:
            members[g][a] = len(groups[g]) - k
            groups[g] += s[k:]
        else:
            shift = L - k
            members[g] = {x: o + shift for x, o in members[g].items()}
            members[g][a] = 0
            groups[g][:0] = s[:shift]
        index(g); done += L - k
    shared = {x: (owner[g], o) for g in range(len(groups)) for x, o in members[g].items() if x != owner[g]}
    store = {owner[g]: (bytes(groups[g]), members[g][owner[g]]) for g in range(len(groups))}
    return shared, store


def merge_literals(t, chunks):
    """Join neighbouring literals of a single chunk that both keep their own bytes: one part instead of two.

    A literal the second parse cut out for sharing (matcher.pool_matches) can end up with nowhere to share,
    next to the rest of its literal. Read one after the other, the two are stored one after the other anyway.
    Sharing only grows: the joined literal holds the bytes of both."""
    shared = None                                        # only needed once two literals could join
    out = []
    for ch, skip, single in chunks:
        if single:
            joined = []
            for u, d in ch:
                if joined:
                    a = joined[-1][0]                    # the part above u
                    if a.kind == u.kind == 'lit' and u.start + u.length == a.start and shared is None:
                        shared = share_literals(t, [(('l', x.start), x.length, 0) for c in chunks
                                                    for p, q in c.pairs for x, need in literal_needs(p, q)])
                    if (a.kind == u.kind == 'lit' and u.start + u.length == a.start
                            and a.length + u.length <= MAX_PART_LEN and not {a.start, u.start} & shared.keys()):
                        joined[-1] = (Part(u.start, a.length + u.length, 'lit', 0), None)
                        continue
                joined.append((u, d))
            ch = joined
        out.append(Chunk(ch, skip, single))
    return out


Placed = namedtuple('Placed', 'off at size reads')


def chunk_size(count, skip, single, fields, fill=False):
    """Array headers and descriptors, excluding literals."""
    return 7 + 2 * (fields + (2 if skip else 0) + (3 if fill else 0)) + (
        7 + 3 * count if single else 4 * (7 + 3 * ((count + 1) // 2)))


class Stream:
    """Chunks and literals in reading order. Placing them for a given B is cheap; building is done once."""
    def __init__(self, t, parts, plan, paired=False, tail=None, per=None, small=None, fills=None, bar=0, overlap=True):
        """fills: {hole index: (start, end, byte)} for the holes that are filled. bar: Options.bar.
        overlap: share literal bytes with overlap_literals(), else only with share_literals()."""
        self.t, self.stage, self.fills = t, plan, fills or {}
        self.hole_start = {p.src: p.start + BASIC_START for p in parts if p.kind == 'skip'}
        self.hole_end = {p.src: p.start + p.length + BASIC_START for p in parts if p.kind == 'skip'}
        seg_at = {vb: vt for vb, vt, w in plan}
        def staged(skip):                                # bytes the chunk copies up first (see build())
            above = self.hole_end[skip - 1] if skip else None
            return seg_at[above] - above if above in seg_at else 0
        def cost(ch, skip, single):
            return chunk_cycles(ch, single, self.fills.get(skip - 1) if skip else None, staged(skip))
        chunks = chunks_for(parts, paired, tail, per, small, resourcer(t, parts, plan))
        if bar: chunks = split_long(chunks, cost, bar)
        chunks = merge_literals(t, chunks)
        self.chunks = chunks
        self.kinds = kinds = (any(not c.single for c in chunks), any(c.single for c in chunks))
        self.fields = header_fields(kinds, any(p.kind == 'skip' for p in parts), bool(plan))
        self.csize = [chunk_size(len(ch), skip, single, len(self.fields), skip - 1 in self.fills)
                      for ch, skip, single in chunks]
        # Each chunk's bar end: its share of the statements, or with bar, of the estimated cycles (even in time).
        weight = [cost(*c) if bar else len(c.pairs) for c in chunks]
        total, done, self.bar = sum(weight), 0, []
        screen = 1024 if poke_bar(BASIC_START + len(t)) else 0
        for w in weight:
            done += w
            self.bar.append(screen + round(BAR_WIDTH * done / total))
        # Reading order: (key, bytes past its start, output written by the time it is done)
        self.needs, y = [], 0
        for c, (ch, skip, single) in enumerate(chunks):
            if skip: y = BASIC_START + len(t) - self.hole_start[skip - 1]
            for u, d in ch:
                for part, need in literal_needs(u, d):
                    self.needs.append((('l', part.start), part.length, y + need))
                y += u.length + (d.length if d else 0)
            self.needs.append((('c', c), self.csize[c], y))
        if overlap:
            self.shared, self.store = overlap_literals(t, self.needs)
        else:
            self.shared = share_literals(t, self.needs)
            self.store = {key[1]: (bytes(t[key[1]:key[1] + L]), 0) for key, L, y in self.needs
                          if key[0] == 'l' and key[1] not in self.shared}
        self.shared_reads = [(self.shared[key[1]], L, y) for key, L, y in self.needs
                             if key[0] == 'l' and key[1] in self.shared]
        # A group owner's slot holds the whole group, but its own read ends with its own bytes.
        self.reach = {key: self.store[key[1]][1] + L for key, L, y in self.needs if key[0] == 'l'
                      and key[1] in self.store}
        self.needs = [(key, len(self.store[key[1]][0]) if key in self.reach else L, y) for key, L, y in self.needs
                      if key[1] not in self.shared or key[0] != 'l']
        # For fits(): offsets at B=0 in file order, and per item the largest offset + length + output written of
        # the reads that move with it (its own, and those of the literals sharing its bytes).
        self._off, self._len, index, cur = [], [], {}, 0
        for key, L, y in reversed(self.needs):
            index[key] = len(self._off); self._off.append(cur); self._len.append(L); cur += L
        w = [-math.inf] * len(self._off)
        for key, L, y in self.needs:
            i = index[key]; w[i] = max(w[i], self._off[i] + self.reach.get(key, L) + y)
        for (b, o), L, y in self.shared_reads:
            i = index[('l', b)]; w[i] = max(w[i], self._off[i] + o + L + y)
        self._pre, self._suf = [-math.inf], [-math.inf] * (len(w) + 1)
        for x in w: self._pre.append(max(self._pre[-1], x))
        for i in range(len(w) - 1, -1, -1): self._suf[i] = max(self._suf[i + 1], w[i])

    def fits(self, sd, ss, B, end):
        """Same as all(top <= end - y for top, y in self.place(sd, ss, B).reads), without placing. Items wholly
        below B stay where place(0) puts them, at sd; the rest are at ss, all shifted alike if one crosses B."""
        i, shift = bisect.bisect_left(self._off, B), 0
        if i and self._off[i - 1] + self._len[i - 1] > B:
            i -= 1; shift = B - self._off[i]
        return self._pre[i] <= end - sd and self._suf[i] <= end - ss - shift

    def place(self, sd, ss=None, B=0):
        """The first B bytes go to sd, the rest stays at ss; nothing crosses B. ss=None: everything at sd.
        Returns Placed: file offsets, addresses, size, and reads = [(top address, output written by then)]."""
        off, cur = {}, 0
        for key, L, y in reversed(self.needs):
            if cur < B < cur + L: cur = B
            off[key] = cur; cur += L
        at = lambda key: (sd if ss is None or off[key] < B else ss) + off[key]
        return Placed(off, at, cur, [(at(key) + self.reach.get(key, L), y) for key, L, y in self.needs]
                      + [(at(('l', b)) + o + L, y) for (b, o), L, y in self.shared_reads])

    def build(self, placed, code_end):
        """The stream bytes. The last chunk links to page 0, which ends the decruncher."""
        t, chunks, stage = self.t, self.chunks, self.stage
        off, at = placed.off, placed.at
        def desc(part):
            if part is None: return bytes(3)
            src = (code_end - 1 if part.kind == 'zero' else
                   lit_at(part.start) if part.kind == 'lit' else CHAR_ROM_ADDR + part.src if part.kind == 'rom'
                   else real_address(BASIC_START + part.src, stage))
            return bytes([part.length, lo(src), hi(src)])
        def lit_at(start):
            b, o = self.shared.get(start, (start, self.store.get(start, (None, 0))[1]))
            return at(('l', b)) + o
        mem = bytearray(placed.size)
        seg_at = {vb: vt for vb, vt, w in stage}
        for c, (ch, skip, single) in enumerate(chunks):
            for u, d in ch:
                for part in (u, d):
                    if part and part.kind == 'lit' and part.start not in self.shared:
                        o, group = off[('l', part.start)], self.store[part.start][0]
                        mem[o:o + len(group)] = group
            nary = at(('c', c + 1)) if c + 1 < len(chunks) else 0
            target = real_fretop(self.hole_start[skip - 1], stage) if skip else 0
            # The segment right above this hole is done: where to copy it.
            above = self.hole_end[skip - 1] if skip else None
            copy = hi(above + ((seg_at[above] - above - 1) & ~255)) if above in seg_at else 0
            # Pair chunk: iteration k does pairs 2k (p$/q$) and 2k+1 (r$/s$). P$ goes first: BASIC searches
            # arrays linearly, and it is looked up most.
            if single:
                b = string_array('P', b''.join(desc(u) for u, d in ch))
            else:
                ch2 = list(ch) + [(None, None)] * (len(ch) % 2)
                A, Bp = ch2[0::2], ch2[1::2]
                b = b''.join(string_array(name, b''.join(desc(x[i]) for x in half))
                             for name, half, i in (('P', A, 0), ('Q', A, 1), ('R', Bp, 0), ('S', Bp, 1)))
            fill = self.fills.get(skip - 1) if skip else None
            # Count and bar end minus one: the loops run to them inclusive.
            val = dict(m=(len(ch) if single else (len(ch) + 1) // 2) - 1, lo=lo(nary), hi=hi(nary),
                       pair=0 if single else 1, hole=2 if fill else 1 if skip else 0, copy=copy, bar=self.bar[c] - 1)
            extra = [lo(target), hi(target)] if skip else []
            if fill:                                     # from the new FRETOP up: byte, L, N (basic.decruncher)
                f0, f1, byte = fill
                assert f0 == target
                extra += [byte, (f1 - f0) & 255, (f1 - f0) >> 8]
            b += integer_array('N', [val[f] for f in self.fields] + extra)
            assert len(b) == self.csize[c]
            o = off[('c', c)]
            mem[o:o + len(b)] = b
        return bytes(mem)


def stream_top(t, parts, plan, paired=False, tail=4, bar=0, overlap=True):
    """(stream size, highest address the whole stream can start at and still stay ahead of the output).
    Stream.place(0) without building the chunks: the same reads, in the same reading order."""
    if bar:
        placed = Stream(t, parts, plan, paired, tail, bar=bar, overlap=overlap).place(0)
        return placed.size, min(BASIC_START + len(t) - y - top for top, y in placed.reads)
    chunks = merge_literals(t, chunks_for(parts, paired, tail, resource=resourcer(t, parts, plan)))
    kinds = any(not c.single for c in chunks), any(c.single for c in chunks)
    fields = len(header_fields(kinds, any(p.kind == 'skip' for p in parts), bool(plan)))
    holes = {p.src + 1: p.start + BASIC_START for p in parts if p.kind == 'skip'}
    end = BASIC_START + len(t)
    needs, y = [], 0
    for c, (pairs, skip, single) in enumerate(chunks):
        if skip: y = end - holes[skip]
        for u, d in pairs:
            for part, need in literal_needs(u, d):
                needs.append((('l', part.start), part.length, y + need))
            y += u.length + (d.length if d else 0)
        needs.append((('c', c), chunk_size(len(pairs), skip, single, fields), y))
    if overlap:
        shared, store = overlap_literals(t, needs)
    else:
        shared = share_literals(t, needs)
        store = {key[1]: (None, 0) for key, L, y in needs if key[0] == 'l' and key[1] not in shared}
    # Reading order runs down from the stream's top. An item consumed bytes before it has its slot top at
    # size - consumed: it is read up to size - consumed - slot + reach, by when y bytes of output are written.
    consumed, limit, slot_of = 0, math.inf, {}
    for key, L, y in needs:
        if key[0] == 'l' and key[1] in shared: continue
        group = store[key[1]][0] if key[0] == 'l' else None
        n, reach = (len(group), store[key[1]][1] + L) if group is not None else (L, L)
        consumed += n
        slot_of[key[1] if key[0] == 'l' else key] = consumed
        limit = min(limit, end - y + consumed - reach)
    for key, L, y in needs:                              # literals reading another's group
        if key[0] == 'l' and key[1] in shared:
            b, o = shared[key[1]]
            limit = min(limit, end - y + slot_of[b] - o - L)
    return consumed, limit - consumed
