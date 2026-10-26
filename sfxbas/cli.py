"""sfxbas: pack a C64 BASIC program into a self-extracting BASIC program.

python3 -m sfxbas in.prg [lambda | -time=S | -size | -speed[=L] | -raw] [-bar=N] [-charrom=FILE [-charmin=N]]
                  [-o out.prg] [-d]

  lambda      bytes one part is worth (default 10). Low = small and slow.
  -time=S     smallest file that unpacks in S seconds
  -size       smallest file, parse and layout search; -size=full searches every lambda breakpoint
  -speed      most bytes saved per second of unpacking; -speed=L: fastest load + unpack at L kB/s
  -raw        fastest unpack: keep as much as fits uncompressed up to $D000 (LOAD puts it in place), no staging
  -bar=N      even progress bar: it advances at least every 1/N of the unpack time (extra chunks: bigger, slower)
  -charrom    copy fonts from this 4096-byte char ROM image; the target needs the same ROM
  -o          output file (default <in>_sfx.prg)
  -d          also write the commented BASIC source next to it (<out>.bas)

cli.py runs the command line and lambda searches. packer.py searches layouts,
parsing.py plans copy parts, stream.py builds packed arrays, and basic.py writes the BASIC.
matcher.py finds optimal parses, memory.py knows the C64, simulator.py checks the result.
"""
import multiprocessing, os, sys
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import replace
from pathlib import Path
from .basic import BAR_WIDTH, PetcatError, program
from .packer import NoFit, Options, build, why_not
from .simulator import PAL_HZ, estimate
from .refinement import refine_result

DEFAULT_LAMBDA = 10
LAMBDA_STEPS = (0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 200, 256)


def say(msg):
    print(msg, file=sys.stderr)


def file_size(r): return len(r.prg) + 2                     # with the load address
def unpack_seconds(r): return estimate(r.layout, r.parts) / PAL_HZ


def build_time(t, target, opts):
    """Smallest file that unpacks in time, else the fastest. Time is not monotone in lambda, so build them all."""
    opts = replace(opts, time_limit=target)
    done = []
    for lam, r in build_many(t, LAMBDA_STEPS, opts).items():
        if r is None: continue
        done.append(r)
        say('  lambda %-4g %6d bytes  ~%5.2f s' % (lam, file_size(r), unpack_seconds(r)))
    if not done: raise NoFit(why_not(t, opts))
    ok = [r for r in done if unpack_seconds(r) <= target]
    r = min(ok, key=lambda r: (len(r.prg), unpack_seconds(r))) if ok else min(done, key=unpack_seconds)
    r = refine_result(t, r, opts, run_jobs)
    say('time target %.2f s: lambda %g, %d bytes, ~%.2f s%s' % (
        target, r.lam, file_size(r), unpack_seconds(r),
        '' if unpack_seconds(r) <= target else ' (the fastest; the target cannot be met)'))
    return r


def try_build(t, lam, opts, run=None):
    """build(), None if nothing fits."""
    try:
        return build(t, lam, opts, run=run)
    except NoFit:
        return None


def try_build_job(args): return try_build(*args)          # picklable for ProcessPoolExecutor.map


JOBS = int(os.environ.get('SFXBAS_JOBS', 0)) or os.cpu_count() or 1
_pool = None


def build_many(t, keys, opts):
    """{key: result or None} for several builds in parallel. A key is a lambda or (lambda, paired).

    Processes, not threads: builds are pure Python. One pool serves the whole run, so its workers keep their match
    caches from one batch of builds to the next. Forked, not spawned: no new interpreter to start. One job per
    build, not per build step (build(run=)): the steps of one build share parse caches."""
    keys = list(keys)
    jobs = [(t, k[0], replace(opts, paired=k[1])) if isinstance(k, tuple) else (t, k, opts) for k in keys]
    return dict(zip(keys, run_jobs(try_build_job, jobs)))


def run_jobs(function, jobs):
    """Stable ordered map; one pool keeps indexes hot across search batches."""
    global _pool
    if JOBS < 2 or len(jobs) < 2: return list(map(function, jobs))
    if _pool is None:
        fork = 'fork' in multiprocessing.get_all_start_methods()
        _pool = ProcessPoolExecutor(JOBS, mp_context=multiprocessing.get_context('fork') if fork else None)
    return list(_pool.map(function, jobs))


def try_size_job(args):
    t, key, variants = args
    lam, paired = key
    # Keep all rounds for one lambda in one worker: DP, pool and placement
    # caches can then be reused before the next lambda displaces them.
    return [try_build(t, lam, replace(opts, paired=paired)) for tag, opts in variants]


def build_size_many(t, keys, variants):
    return dict(zip(keys, run_jobs(try_size_job, [(t, key, variants) for key in keys])))


# -size tries these, single chunks only. Close enough to the full search.
SIZE_GRID = (0, 1 / 12, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 1, 1.25)
SIZE_MAX_DENOMINATOR = 6  # -size=full
SIZE_MAX_LAMBDA = 3


def size_lambdas(maxq=SIZE_MAX_DENOMINATOR, top=SIZE_MAX_LAMBDA):
    """Every lambda where the parse can change, plus one between each pair.

    Bytes and parts are integers, so the parse only changes where lambda crosses a small fraction p/q."""
    from fractions import Fraction
    fr = sorted({Fraction(p, q) for q in range(1, maxq + 1) for p in range(0, top * q + 1)})
    mids = [(x + y) / 2 for x, y in zip(fr, fr[1:])]
    return sorted(round(float(x), 6) for x in fr + mids)


def build_size(t, opts, full=False):
    """Smallest file. lambda 0 is not it: the parse ignores chunk overhead, so a small lambda often wins.
    Ties go to the faster build. Falls back to the full search if the grid finds nothing."""
    # The full search includes the grid exactly: ties at a breakpoint go either way, so 2/3 and 0.666667 can differ.
    keys = ([(l, p) for l in sorted(set(size_lambdas()) | set(SIZE_GRID)) for p in (True, False)] if full
            else [(l, False) for l in SIZE_GRID])
    # Keep baseline builds: better local layout choices or new literal pools can
    # change the staging search, so only final verified PRG sizes decide.
    variants = [((1, False), opts)]
    for rounds in range(1, 5):
        candidates = replace(opts, pool_rounds=rounds, search_layout=True)
        variants.append(((rounds, True), candidates))
        if rounds > 1: variants.append(((rounds, False), replace(candidates, search_layout=False)))
    for rounds in range(1, 5):
        variants.append(((rounds, 2), replace(opts, pool_seed=SIZE_GRID[-1], pool_rounds=rounds, search_layout=True)))
    batches = build_size_many(t, keys, variants)
    # Preserve the original variant/lambda insertion order, including exact ties.
    done = {(l, p, *tag): batches[l, p][i] for i, (tag, policy) in enumerate(variants)
            for l, p in keys if batches[l, p][i]}
    if not done and not full: return build_size(t, opts, True)
    if not done: raise NoFit(why_not(t, opts))
    key = lambda k: (len(done[k].prg), unpack_seconds(done[k]))
    for k in sorted(done, key=key)[:8]:
        method = ('seed %g, ' % SIZE_GRID[-1] if k[3] == 2 else '') + 'pool %d' % k[2]
        if k[3]: method += ', layouts'
        say('  lambda %-8g %-6s %6d bytes  ~%5.2f s (%s)' % (
            k[0], 'pairs' if k[1] else 'single', file_size(done[k]), unpack_seconds(done[k]), method))
    b = min(done, key=key)
    result = refine_result(t, done[b], replace(opts, search_layout=True), run_jobs)
    say('smallest: lambda %g, %s chunks, %d bytes (%d builds)' % (b[0], 'pair' if b[1] else 'single',
                                                                   file_size(result), len(done)))
    return result


SPEED_TIE_MARGIN = 0.01  # within 1% is a tie; the lower lambda wins
SPEED_PATIENCE = 2       # keep searching until the best is this far from the edge
SPEED_START_LAMBDA = 20  # usually near the peak
SPEED_LOAD_LAMBDA = 2.5  # with a load rate L kB/s the peak is near this * L: a byte costs more the slower the load


def build_speed(t, opts, load=None):
    """Best bytes saved per second of unpacking, or with load=L kB/s, the fastest load + unpack.

    Hill climb in parallel batches from SPEED_START_LAMBDA. Low lambdas build slowly, so only go there when
    they can win."""
    lams = sorted(set(LAMBDA_STEPS) | ({0.25, 0.5, 0.75} if load is not None else set()))
    opts = replace(opts, speed='max' if load is None else load)    # build() ranks its variants the same way
    n, B = len(lams), max(JOBS, 3)
    done = {}                                            # index -> (result, score) or None
    def score(r):
        sec = unpack_seconds(r)
        if load is None:
            rate = (len(t) - len(r.prg)) / 1024 / sec
            return rate, '~%5.2f s  saved %5.2f kB/s' % (sec, rate)
        tot = file_size(r) / 1024 / load + sec
        return -tot, 'load %5.1f s + unpack ~%5.2f s = %5.1f s' % (file_size(r) / 1024 / load, sec, tot)
    def build_idx(idx):
        idx = [i for i in idx if 0 <= i < n and i not in done]
        for i, r in sorted(zip(idx, build_many(t, [lams[i] for i in idx], opts).values()), key=lambda x: x[0]):
            done[i] = (r, score(r)[0]) if r else None
            if r: say('  lambda %-4g %6d bytes  %s' % (lams[i], file_size(r), score(r)[1]))
    def best():
        ok = {i: d for i, d in done.items() if d}
        if not ok: return None
        top = max(d[1] for d in ok.values())
        margin = abs(top) * SPEED_TIE_MARGIN
        return min(i for i, d in ok.items() if d[1] >= top - margin)   # lowest lambda within the margin
    start = SPEED_START_LAMBDA if load is None else SPEED_LOAD_LAMBDA * load
    c = min(range(n), key=lambda i: abs(lams[i] - start))
    build_idx(range(c - B // 2, c - B // 2 + B))
    while True:
        b = best()
        lo, hi = min(done), max(done)
        if b is None:                                    # nothing built fits: try lower lambdas
            if lo == 0: break
            build_idx(range(lo - B, lo)); continue
        down, up = b - lo < SPEED_PATIENCE and lo > 0, hi - b < SPEED_PATIENCE and hi < n - 1
        if not (down or up): break
        if down: build_idx(range(lo - B, lo))
        if up: build_idx(range(hi + 1, hi + 1 + B))
    b = best()
    if b is None: raise NoFit(why_not(t, opts))
    r, sc = done[b]
    r = refine_result(t, r, opts, run_jobs)
    sc = score(r)[0]
    say('best: lambda %g, %d bytes, %s' % (lams[b], file_size(r), '%.2f kB/s saved' % sc if load is None else
                                           '%.1f s load + unpack at %g kB/s' % (-sc, load)))
    return r


def build_fitting(t, lam, opts):
    """build() at lam, or the largest lambda below it that fits. A high lambda means more literals, and the
    stream can stop keeping ahead of the output."""
    try:
        return refine_result(t, build(t, lam, opts, log=say, run=run_jobs), opts, run_jobs)
    except NoFit as e:
        say('lambda %g: %s' % (lam, e))
    lo_, hi_, best = 0.0, lam, try_build(t, 0.0, opts, run_jobs)
    if best is None: raise NoFit(why_not(t, opts))
    while hi_ - lo_ > max(0.01, lo_ / 200):              # bisect to ~0.5%
        mid = (lo_ + hi_) / 2
        r = try_build(t, mid, opts, run_jobs)
        if r: lo_, best = mid, r
        else: hi_ = mid
    say('using lambda %.3g, the largest that fits' % lo_)
    return refine_result(t, best, opts, run_jobs)


def parse_args(argv=None):
    """in.prg [lambda] plus single-dash options; values go with '='."""
    import argparse
    ap = argparse.ArgumentParser(prog='python3 -m sfxbas', allow_abbrev=False,
                                 description='Self-extracting BASIC packer (see the module docstring).')
    ap.add_argument('input', help='BASIC PRG loading at $0801')
    ap.add_argument('lam', nargs='?', type=float, metavar='lambda',
                    help='parse cost per part in bytes (default %g)' % DEFAULT_LAMBDA)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('-size', nargs='?', choices=('grid', 'full'), metavar='=full',
                      help='smallest file: lambda grid, literal-pool and layout search; -size=full: more lambdas and paired parses')
    mode.add_argument('-speed', nargs='?', metavar='=L',
                      help='most bytes saved per second; -speed=L: shortest load + unpack at L kB/s')
    mode.add_argument('-time', type=float, metavar='S', help='smallest file unpacking in at most S seconds')
    mode.add_argument('-raw', action='store_true',
                      help='fastest unpack: the largest uncompressed block that fits, up to $D000; no staging')
    ap.add_argument('-keep0', action='store_true', help=argparse.SUPPRESS)   # the only behaviour now; still accepted
    ap.add_argument('-bar', type=int, metavar='N', default=0,
                    help='even progress bar: advance at least every 1/N of the unpack (1-%d; more chunks)' % BAR_WIDTH)
    ap.add_argument('-charrom', metavar='FILE', help='copy obvious font copies from this 4096-byte character ROM')
    ap.add_argument('-charmin', type=int, metavar='N', default=Options.char_min,
                    help='minimum char ROM stretch (default %d)' % Options.char_min)
    ap.add_argument('-o', metavar='FILE', dest='out', help='output file (default <input>_sfx.prg)')
    ap.add_argument('-d', action='store_true', dest='source', help='also write the BASIC source as <output>.bas')
    # Bare -size / -speed must not eat the next word (the input file).
    bare = {'-size': '-size=grid', '-speed': '-speed=max'}
    args = ap.parse_intermixed_args([bare.get(a, a) for a in (sys.argv[1:] if argv is None else argv)])
    if args.lam is not None and (args.size or args.speed or args.time is not None):
        ap.error('a lambda cannot be combined with -size, -speed or -time')
    if not 0 <= args.bar <= BAR_WIDTH:
        ap.error('-bar=N: N must be 1-%d' % BAR_WIDTH)
    if args.speed not in (None, 'max'):
        try: args.speed = float(args.speed)
        except ValueError: ap.error('-speed=L: L must be a number (kB/s)')
    return args


def main():
    args = parse_args()
    rom = Path(args.charrom).read_bytes() if args.charrom else None
    if rom is not None and len(rom) != 4096:
        sys.exit('-charrom: expects a 4096-byte character ROM image')
    opts = Options(char_rom=rom, char_min=args.charmin, raw=args.raw, bar=args.bar)
    name = args.input
    prg = Path(name).read_bytes()
    if prg[:2] != b'\x01\x08':
        sys.exit('%s: expects a BASIC PRG at $0801' % name)
    t = prg[2:]
    try:
        if args.size:
            r = build_size(t, opts, args.size == 'full')
        elif args.speed:
            r = build_speed(t, opts, None if args.speed == 'max' else args.speed)
        elif args.time is not None:
            r = build_time(t, args.time, opts)
        else:
            r = build_fitting(t, DEFAULT_LAMBDA if args.lam is None else args.lam, opts)
    except (NoFit, PetcatError) as e:
        sys.exit('%s: cannot pack: %s' % (name, e))
    dst = args.out or os.path.splitext(name)[0] + '_sfx.prg'
    Path(dst).write_bytes(b'\x01\x08' + r.prg)
    if args.source:
        Path(os.path.splitext(dst)[0] + '.bas').write_text(program(r.layout))
    if len(r.prg) >= len(t):
        say('warning: packed file (%d bytes) is not smaller than the input (%d); use a lower lambda'
            % (len(r.prg), len(t)))
    v, parts = r.layout, r.parts
    kinds = Counter(p.kind for p in parts)
    print('%s: %d -> %d bytes (BASIC %d, stream %d, bottom %d pages moved to $%04X, %s), %d parts (%d literal%s), '
          '%d chunks%s, lambda %g, unpack ~%.1f s%s; simulation ok; wrote %s' % (
              name, len(t), len(r.prg), r.code_size, r.stream_size, v.moved_pages, v.stream_dst,
              'raw $%04X-$%04X' % (v.raw[0], v.raw[1] - 1) if v.raw else 'no raw block',
              len(parts) - kinds['skip'], kinds['lit'],
              ', %d char ROM = %d bytes' % (kinds['rom'], sum(p.length for p in parts if p.kind == 'rom'))
              if rom else '',
              v.n_chunks, '' if v.paired else ' (single chunks)', r.lam, unpack_seconds(r),
              ' (cycle-aware)' if r.cycle_ties else '', dst))


if __name__ == '__main__':
    main()
