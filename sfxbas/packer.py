"""Build search and final layout. parsing.py chooses parts; stream.py forms the packed arrays.
The output is written in place, top down, and must never overtake stream bytes still to be read. So the
decruncher moves the bottom of the stream (read last) down into screen RAM; the rest stays where LOAD put it.
The bottom chunks are kept small for the same reason.

The packed PRG must end below $A000: BASIC cannot read the RAM under its ROM.

Raw mode, when nothing else fits or with -raw: the PRG carries [x, $C000) as is (LOAD can write under the ROM), and
the stream only produces what is below x, and what is above the raw block. With -raw it reaches $D000 and nothing
is staged. The fallback stops at $C000: a block up to $D000 would let a high lambda fit where a lower one with
staging gives a smaller file.
"""
import bisect
import copy
from dataclasses import dataclass, replace
from functools import partial, lru_cache
from .basic import poke_bar, DECRUNCHER_START, Layout, Vars, loader, tokenize, tokenized_program
from .matcher import Part
from .memory import (FILL_MIN, BASIC_ROM, BASIC_START, FIXED_WINDOW, HIGH_RAM, INPUT_END, fill_runs, free_windows,
                     fretop_jumps, low_staged, mask_color_ram, merge_ranges, readable, real_fretop, stage_plan,
                     windows_used)
from .parsing import cut_at_holes, parse
from .simulator import PAL_HZ, estimate, simulate
from .stream import MAX_PER_CHUNK, SMALL_TAIL_PAIRS, Stream, stream_top

CHUNK_SIZES = (MAX_PER_CHUNK, 1024, 253, 128, 64, 32, 16)  # place() tries these, largest first


@dataclass(frozen=True)
class Options:
    char_rom: bytes = None       # copy fonts from this char ROM image; the target needs the same ROM
    char_min: int = 32           # shortest char ROM copy
    paired: bool = True          # pair chunks (faster); False = single chunks only (smaller)
    raw: bool = False            # -raw: no staging, the largest raw block that fits (fastest unpack, bigger file)
    fill_min: int = FILL_MIN     # runs of one byte this long are filled by the ROM, not copied (0: off)
    speed: object = None         # how build() ranks its variants: None = size, 'max' or load KB/s as in -speed
    bar: int = 0                 # -bar=N: the bar advances at least every 1/N of the unpack, evenly in time (0: off)
    pool_seed: float = None      # alternate seed parse lambda for literal-pool discovery
    pool_rounds: int = 1         # repeated literal-pool discovery; -size compares each round
    search_layout: bool = False # -size compares fitting chunk layouts by final PRG size
    cycle_ties: bool = True      # compare legacy and cycle-aware builds; False reproduces the legacy path
    source_choices: bool = False # optional paired-source candidate, selected after the CLI search
    time_limit: float = None     # -time: prefer meeting the deadline, then size; otherwise fastest


@dataclass
class Result:
    prg: bytes                   # without the load address
    layout: Layout
    parts: list
    code_size: int               # loader + decruncher
    stream_size: int
    lam: float = None
    cycle_ties: bool = False


@dataclass
class Job:
    """One build attempt: the input with colour RAM masked and its staging plan."""
    t: bytes
    plan: tuple
    basic_end: int
    opts: Options
    fills: list = ()

    @property
    def end(self): return BASIC_START + len(self.t)


def run_vartab(t):
    """VARTAB for RUN: where LOAD leaves it, after the last byte loaded (16 bits). Past $A000 too: a program
    that pokes only its high byte (poke46,50) must find its variables where the original does."""
    return min(BASIC_START + len(t), 0x10000) & 0xFFFF


class NoFit(RuntimeError): pass
class NotAhead(NoFit): pass                             # output overtook the stream


def fitting(calls):
    """([results], [NoFit errors]) of calling each of calls."""
    results, errors = [], []
    for call in calls:
        try:
            results.append(call())
        except NoFit as e:
            errors.append(e)
    return results, errors


def build(t, lam, opts=Options(), log=None, run=None):
    """Pack t (the PRG without its load address) at lambda lam. Returns a Result or raises NoFit.

    Compare complete legacy and cycle-aware builds, including fill and staging
    variants. Rank final verified files by size (cycles break ties), speed, or
    deadline. A size build never trades bytes for a cycle saving.
    run: an ordered map, such as the CLI's worker pool, for the independent build_staged() calls."""
    plan = build_plan(t, opts)
    jobs = [(t, lam, o) for policy, variants in plan for o in variants]
    if run is None:
        outcomes = [staged_outcome(*job, log) for job in jobs]
    else:
        outcomes = []
        for outcome, messages in run(staged_job, jobs):   # their messages, in the order build_staged() says them
            for msg in messages:
                if log: log(msg)
            outcomes.append(outcome)
    return combine(t, opts, plan, outcomes)


def build_plan(t, opts):
    """build()'s build_staged() options: [(parser policy, [fill variants])]. Parsing cost omits final literal
    sharing, chunk headers and relocation, so a complete legacy build is retained next to the cycle-aware one."""
    policies = [replace(opts, cycle_ties=False), opts] if opts.cycle_ties else [opts]
    return [(p, fill_variants(t, p)) for p in policies]


def fill_variants(t, opts):
    """With and without fills, when there are any."""
    if opts.raw or not opts.fill_min or not fill_runs(mask_color_ram(t), min_len=opts.fill_min):
        return [opts]
    return [opts, replace(opts, fill_min=0)]


def staged_outcome(t, lam, opts, log=None):
    """build_staged()'s Result, or its NoFit."""
    try:
        return build_staged(t, lam, opts, log)
    except NoFit as e:
        return e


def staged_job(args):
    """(staged_outcome(), its log messages): for a worker process."""
    messages = []
    return staged_outcome(*args, messages.append), messages


def combine(t, opts, plan, outcomes):
    """build()'s result from the outcomes of build_plan()'s calls, in order: each policy ranks its fill variants
    by its own options, then the policies are ranked by opts. NoFit: the last policy's last error."""
    outcomes, results, errors = iter(outcomes), [], []
    for policy, variants in plan:
        got = [next(outcomes) for _ in variants]
        fit = [x for x in got if not isinstance(x, NoFit)]
        if fit: results.append(min(fit, key=lambda r: rank(r, t, policy)))
        else: errors.append(got[-1])
    if not results: raise errors[-1]
    return min(results, key=lambda r: rank(r, t, opts))


def build_staged(t, lam, opts, log=None):
    """build() for one fill setting.

    Grows the staging window into the RAM freed above the packed file and rebuilds while that keeps shrinking
    it, with and without staging $A000-$BFFF there too. If staging does not fit at all, tries without: its code
    costs screen RAM."""
    log = log or (lambda msg: None)
    if opts.raw: return build_once(t, lam, opts, (), log)
    has_low = BASIC_START + len(t) > BASIC_ROM[0]
    free = free_windows(t)
    key = lambda r: rank(r, t, opts)
    lows = [(), free] if free else [()]
    rs, errs = fitting(partial(build_once, t, lam, opts, [FIXED_WINDOW], log, low) for low in lows)
    best = min(rs, key=key, default=None)
    if has_low and (best is None or best.layout.raw):
        r = first_low(t, lam, opts, log, free)
        if r and (best is None or key(r) < key(best)): best = r
    if best is None:
        if not stage_plan(mask_color_ram(t), [FIXED_WINDOW]): raise errs[0]
        log('with the staging window: %s; trying without it' % errs[0])
        return build_once(t, lam, opts, (), log)
    if not (best.layout.stage or has_low) or best.layout.raw: return best
    top = BASIC_START + len(best.prg)
    for _ in range(4):
        w1 = (top + 255) & ~255
        if w1 >= 0xA000: break
        win = (w1, 0xA000)
        lows = ([free + [win], [win], ()] if free else [[win], ()]) if has_low else [()]
        rs, _ = fitting(partial(build_once, t, lam, opts, [FIXED_WINDOW, win], log, low) for low in lows)
        if not rs: break
        r = min(rs, key=key)
        if key(r) >= key(best): break
        best, top = r, BASIC_START + len(r.prg)
    return best


def rank(r, t, opts):
    """Final size/time ranking for every candidate; lower is better."""
    if opts.time_limit is not None:
        cycles = estimate(r.layout, r.parts)
        sec = cycles / PAL_HZ
        return (0, len(r.prg), cycles) if sec <= opts.time_limit else (1, cycles, len(r.prg))
    if opts.speed is None:
        return (len(r.prg), round(10 * estimate(r.layout, r.parts))) if opts.cycle_ties else len(r.prg)
    sec = estimate(r.layout, r.parts) / PAL_HZ
    if opts.speed == 'max': return -(len(t) - len(r.prg)) / sec
    return (len(r.prg) + 2) / 1024 / opts.speed + sec


def first_low(t, lam, opts, log, free=()):
    """Staging $A000-$BFFF when the plain build did not fit (or only raw): unread, $A000 data cost as literals.
    The window starts just big enough for both phases, then shrinks from below until the packed file fits.
    free: windows at $C000 nothing writes afterwards, used first."""
    m = mask_color_ram(t)
    whole = stage_plan(m, [FIXED_WINDOW, (BASIC_START, 0xA000)], [(BASIC_START, 0xA000)])
    pages = sum((vt - vb + 255) // 256 for vb, vt, w in whole if w < FIXED_WINDOW[0])
    for w1 in range(max(0xA000 - 256 * pages, 0x1000), 0xA000, 0x800):
        try:
            return build_once(t, lam, opts, [FIXED_WINDOW, (w1, 0xA000)], log, list(free) + [(w1, 0xA000)])
        except NoFit as e:
            log('staging $A000-$BFFF at $%04X: %s' % (w1, e))
    return None


def build_once(t, lam, opts, windows, log, low=()):
    if not t or BASIC_START + len(t) > INPUT_END:
        raise ValueError('input must contain bytes within $0801-$103FF ($10000-$103FF = colour RAM)')
    t = mask_color_ram(t)
    fills = [] if opts.raw else fill_runs(t, min_len=opts.fill_min)
    plan = stage_plan(t, windows, low, fills)
    job = Job(t, plan, run_vartab(t), opts, fills)
    parsed = parse(t, lam, plan, opts)
    col = fretop_jumps(t, plan)
    while True:     # a fill starts at its hole's bottom, the new FRETOP (not when that jumps into a window)
        holes = merge_ranges(col + [(f0, f1) for f0, f1, b in fills])
        ok = [f for f in fills if any(h0 == f[0] < f[1] <= h1 for h0, h1 in holes) and real_fretop(f[0], plan) == f[0]]
        if ok == fills: break
        fills = job.fills = ok
    if opts.raw: return raw_results(job, parsed, col, opts, lam, None)
    try:
        r = place(job, cut_at_holes(parsed, holes), holes, opts.paired)
        r.lam = lam
        return r
    except NoFit as e:
        err = e
        if low_staged(plan): raise                       # raw mode keeps $A000-$BFFF as is: build() tries without
    # Raw mode is bigger, so say why we fell back.
    log('normal layout: %s; trying raw mode' % err)
    return raw_results(job, parsed, col, opts, lam, err)


def raw_results(job, parsed, fixed, opts, lam, err):
    """The best raw build. err: why the normal layout did not fit (None with -raw)."""
    # Try single chunks too: a slightly smaller stream can lower the raw start a lot. Lowest start wins.
    modes = (True, False) if opts.paired else (False,)
    results, errs = fitting(partial(build_raw, job, parsed, fixed, f) for f in modes)
    if not results:
        raise NoFit('%sraw mode: %s' % ('%s; ' % err if err else '', '; '.join(sorted(set(map(str, errs))))))
    r = min(results, key=lambda r: (r.layout.raw[0], not r.layout.paired))
    r.lam = lam
    return r


def build_raw(job, parsed, fixed, paired):
    """Raw mode: find the lowest x where the stream for [$0801, x) fits below x. fixed: the other holes."""
    t, e = job.t, min(job.end, HIGH_RAM if job.opts.raw else 0xC000)
    holes = lambda x: merge_ranges(fixed + [(x, e)])
    # Fixed holes and the output above the raw block do not change as x moves.
    # Only the final part of the prefix can need another cut.
    kept = [p for p in cut_at_holes(parsed, fixed) if p.kind != 'skip']
    starts = [p.start for p in kept]
    ei = bisect.bisect_left(starts, e - BASIC_START)
    suffix = kept[ei:]
    if ei and kept[ei - 1].start + kept[ei - 1].length > e - BASIC_START:
        p = kept[ei - 1]
        cut = e - BASIC_START - p.start
        suffix = [Part(p.start + cut, p.length - cut, p.kind, p.src + cut), *suffix]
    def raw_parts(x):
        hs = holes(x)
        stop = x - BASIC_START
        prefix = kept[:bisect.bisect_left(starts, stop)]
        if prefix and prefix[-1].start + prefix[-1].length > stop:
            prefix[-1] = prefix[-1]._replace(length=stop - prefix[-1].start)
        skips = [Part(a - BASIC_START, b - a, 'skip', i) for i, (a, b) in enumerate(hs)]
        return sorted(prefix + suffix + skips), hs
    v = Layout(job.end, job.plan, holes(0xFFFF), (True, True), job.basic_end, paired)
    code = len(tokenized_program(v))                          # upper bound
    low = Vars(bool(job.plan), DECRUNCHER_START + code - len(tokenize(loader(v), BASIC_START)) + 2,
               poke_bar(job.end)).end
    def need(x):                                         # where the PRG ends for raw start x
        parts, _ = raw_parts(x)
        size, top = stream_top(t, parts, job.plan, paired, bar=job.opts.bar, overlap=False)   # fits both
        return BASIC_START + code + size + 255 if top >= low else 0x10000
    # Not monotone in x, so scan coarse, then fine below the first hit.
    top = min(e, 0xA000)
    x = next((x for x in range(BASIC_START + code, top + 256, 256) if need(min(x, top)) <= min(x, top)), None)
    if x is None: raise NoFit('no raw start fits')
    x = next(y for y in range(max(BASIC_START + code, x - 255), min(x, top) + 1) if need(y) <= y)
    reasons = set()
    for x in range(x, min(x + 64, top) + 1):
        try:
            return place(job, *raw_parts(x), paired, raw=(x, e))
        except NoFit as raw_err:
            reasons.add(str(raw_err))
    raise NoFit('; '.join(sorted(reasons)))


def place(job, parts, holes, paired, raw=None):
    """Reuse verified placement when another parse has identical parts and policy.

    Pool-discovery settings no longer matter after parsing. Return fresh mutable
    containers: callers set lam and may inspect or modify the returned layout.
    """
    opts = replace(job.opts, pool_seed=None, pool_rounds=1,
                   char_rom=bytes(job.opts.char_rom) if job.opts.char_rom is not None else None)
    result = cached_place(bytes(job.t), tuple(job.plan), job.basic_end, opts,
                          tuple(job.fills), tuple(parts), tuple(holes), paired, raw)
    layout = copy.deepcopy(replace(result.layout, chunks=None))
    layout.chunks = [chunk._replace(pairs=list(chunk.pairs)) for chunk in result.layout.chunks]
    return replace(result, layout=layout, parts=list(result.parts))


@lru_cache(maxsize=32)
def cached_place(t, plan, basic_end, opts, fills, parts, holes, paired, raw):
    """Immutable cache key; the result remains private to place()."""
    job = Job(t, plan, basic_end, opts, list(fills))
    return place_uncached(job, list(parts), list(holes), paired, raw)


def place_uncached(job, parts, holes, paired, raw=None):
    """Try ever smaller chunks until the stream stays ahead of the output: the bottom ones first, then single
    chunks at the bottom, then everywhere."""
    # Restore $0800 from the decruncher's end marker. The progress bar now overwrites
    # the zero before the relocated program, so it cannot be used as the source.
    parts = [Part(-1, 1, 'zero', 0), *parts]
    key_of = (lambda r: rank(r, job.t, job.opts)) if job.opts.cycle_ties or job.opts.time_limit is not None \
        else (lambda r: len(r.prg))
    def both(*layout):
        """The better of overlapping and contained literal sharing for one chunk layout. Overlap groups store
        less, but move the bytes between a group and its members up: a layout can stop fitting."""
        best, errors = None, {}
        for overlap in (True, False):
            try:
                best = try_layout(job, parts, holes, paired, raw, *layout, overlap=overlap,
                                  beat=(key_of, key_of(best)) if best else None)
            except NotBetter:
                pass
            except NoFit as e:
                errors[overlap] = e
        if best is None: raise errors[False]                 # as without overlap groups
        return best
    def first():
        for per in CHUNK_SIZES:
            for small in (SMALL_TAIL_PAIRS, MAX_PER_CHUNK):
                for tail in (per, 32, 16, 8, 4, 2, 1):
                    if tail > per: continue
                    try:
                        return both(tail, per, small)
                    except NotAhead as e:
                        err = e
        raise err
    best = first()
    if not job.opts.search_layout: return best
    # A fitting layout is not necessarily the smallest: headers, padding and
    # literal sharing depend on chunk boundaries. Keep the original candidate.
    def signature(chunks):
        return tuple((tuple(ch), skip, single) for ch, skip, single in chunks)
    seen = {(signature(best.layout.chunks), overlap) for overlap in (True, False)}   # first() tried both
    fills = {} if raw else {i: f for i, (h0, h1) in enumerate(holes)
                           for f in job.fills if h0 <= f[0] and f[1] <= h1}
    for per in (MAX_PER_CHUNK, 253, 128):
        for small in ((SMALL_TAIL_PAIRS, MAX_PER_CHUNK) if paired else (SMALL_TAIL_PAIRS,)):
            for tail in dict.fromkeys((per, 128, 64, 32, 4, 1)):
                for overlap in (True, False):
                    try:
                        stream = Stream(job.t, parts, job.plan, paired, tail, per, small, fills, job.opts.bar,
                                        overlap)
                        key = signature(stream.chunks), overlap
                        if key in seen: continue
                        seen.add(key)
                        best = try_layout(job, parts, holes, paired, raw, tail, per, small, _stream=stream,
                                          beat=(key_of, key_of(best)))
                    except NoFit:
                        continue
    return best


class NotBetter(NoFit): pass                            # try_layout(beat=...): would lose, so not simulated


def try_layout(job, parts, holes, paired, raw, tail, per, small, _stream=None, beat=None, overlap=True):
    """Lay out loader, decruncher and stream, and simulate it. NoFit if it does not work.
    beat: (key, value): NotBetter unless key(result) < value. The ranking needs only sizes and the layout, so a
    loser is neither built nor simulated."""
    t, end = job.t, job.end
    # A filled hole holds exactly one fill. Raw mode has none: its block may cover them.
    fills = {} if raw else {i: f for i, (h0, h1) in enumerate(holes) for f in job.fills if h0 <= f[0] and f[1] <= h1}
    assert raw or len(fills) == len(job.fills), 'two fills in one hole'
    plan = _stream if _stream is not None else Stream(t, parts, job.plan, paired, tail, per, small, fills, job.opts.bar,
                                                      overlap)
    v = Layout(end, job.plan, holes, plan.kinds, job.basic_end, paired, len(plan.chunks),
               stream_dst=0, fill=bool(fills))
    unplaced = plan.place(0)
    pages = (unplaced.size + 255) // 256 + 2
    highest = min(end - y - top for top, y in unplaced.reads)
    minimum_vars = Vars(bool(job.plan), DECRUNCHER_START, poke_bar(end), len(plan.fields),
                        len(plan.chunks)).end
    if highest < minimum_vars:
        raise NotAhead('the stream cannot stay ahead of the output')
    def lay(sd, ss):                                     # move as few pages as possible
        for B in range(256, 256 * pages, 256):
            if plan.fits(sd, ss, B, end):
                return B, plan.place(sd, ss, B)
        raise NotAhead('the stream cannot stay ahead of the output')
    # The addresses are in the code, and the code size moves the addresses. Iterate. A digit count can flip
    # back and forth forever, so code never shrinks: pad it with colons nobody executes.
    lmax = dmax = 0
    for i in range(40):
        code = tokenized_program(replace(v, loader_pad=0, decruncher_pad=0))
        l0 = code[0] + 256 * code[1] - BASIC_START; d0 = len(code) - l0
        if i < 2: lmax, dmax = l0, d0                    # pass 0 has placeholders
        lmax, dmax = max(lmax, l0), max(dmax, d0)
        # K is the first scalar and is read for every string-array subscript.
        # Keep its seven-byte record within one page to avoid extra cycles in
        # BASIC's indirect variable reads. Padding is never executed.
        var_lo = (DECRUNCHER_START + dmax) & 255
        if var_lo > 249:
            dmax += 256 - var_lo
        pads = dict(loader_pad=lmax - l0, decruncher_pad=dmax - d0)
        vt = DECRUNCHER_START + dmax
        f = Vars(bool(job.plan), vt, poke_bar(end), len(plan.fields), len(plan.chunks))
        sd = max(v.stream_dst, f.end)                     # never moves down: converges
        if i >= 1:                                      # pass 0's placeholder size may shrink
            if highest < sd:
                raise NotAhead('the stream cannot stay ahead of the output')
        code = tokenized_program(replace(v, **pads))
        dsrc = code[0] + 256 * code[1]                    # the decruncher's first line
        ld, dc = code[:dsrc - BASIC_START], code[dsrc - BASIC_START:]
        assert (len(ld), len(dc)) == (lmax, dmax)
        ss = BASIC_START + len(code)
        B, placed = lay(sd, ss)
        cn = (len(dc) + 1 + 255) // 256                  # pages to move, with the $00 before line 1
        nv = replace(v, code_src=dsrc - 1 + 256 * (cn - 1), code_dst=DECRUNCHER_START - 1 + 256 * (cn - 1),
                     code_pages=cn, stream_src=ss, stream_dst=sd, moved_pages=B // 256, stream_len=placed.size,
                     vartab=vt, first_chunk=placed.at(('c', 0)), **pads)
        if nv == v: break
        v = nv
    else:
        raise NoFit('layout numbers do not converge')
    if v.stream_src - v.stream_dst < 256:
        raise NoFit('stream move shift below one page')
    packed_size = len(ld) + len(dc) + placed.size
    if not (readable(BASIC_START, packed_size)
            and readable(v.code_src - 256 * (v.code_pages - 1), 256 * v.code_pages)):
        raise NoFit('packed stream/decruncher does not fit in BASIC-readable RAM below $A000')
    if raw and BASIC_START + packed_size > raw[0]: raise NoFit('stream overlaps the raw block')
    file_end = raw[1] if raw else BASIC_START + packed_size
    if any(w0 < file_end for w0, w1 in windows_used(job.plan)):
        raise NoFit('a staging window overlaps the packed file')
    v = replace(v, raw=raw, chunks=plan.chunks, fill_runs=tuple(fills.values()))
    if beat:
        size = raw[1] - BASIC_START if raw else packed_size
        key, value = beat
        if not key(Result(bytes(size), v, parts, len(ld) + len(dc), placed.size)) < value:
            raise NotBetter('not smaller')
    st = plan.build(placed, vt)
    prg = ld + dc + st
    if raw:
        prg += bytes(raw[0] - BASIC_START - len(prg)) + t[raw[0] - BASIC_START:raw[1] - BASIC_START]
    if not simulate(prg, v, t, job.opts.char_rom): raise NoFit('simulation failed')
    return Result(prg, v, parts, len(ld) + len(dc), len(st), cycle_ties=job.opts.cycle_ties)


MAX_PRG_SIZE = 0xA000 - BASIC_START


def why_not(t, opts=Options()):
    """Why nothing fits, for the error message."""
    t = mask_color_ram(t)
    plan = stage_plan(t, [FIXED_WINDOW])
    col = fretop_jumps(t, plan)
    parts = cut_at_holes(parse(t, 0, plan, opts), merge_ranges(col))
    stream_size = Stream(t, parts, plan, paired=True).place(0).size
    v = Layout(BASIC_START + len(t), plan, col, (True, True), BASIC_START + 2)
    need = len(tokenized_program(v)) + stream_size
    msg = ('the packed PRG must end below $A000, i.e. at most %d bytes ($0801-$9FFF): BASIC cannot read the RAM '
           'under its ROM. ' % MAX_PRG_SIZE)
    if need > MAX_PRG_SIZE:
        msg += ('Even at lambda 0 this input needs about %d bytes (stream %d = %.0f%% of the %d input bytes): '
                'it does not compress enough (already crunched?).'
                % (need, stream_size, 100 * stream_size / len(t), len(t)))
        if BASIC_START + len(t) > 0xA000:
            msg += (' Raw mode (keeping the output at $A000-$BFFF uncompressed in place) did not fit either: '
                    'everything else must still come from the stream.')
    else:
        msg += ('It would fit (about %d bytes at lambda 0), but not in place: the output under the loader always '
                'comes from the stream, and the data there does not compress enough to stay ahead of the output.'
                % need)
        return msg
