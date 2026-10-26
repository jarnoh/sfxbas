"""The BASIC side: loader and decruncher source, tokenized with petcat.

Layout at $0801: line 0 is the loader, then the decruncher (10 title, 12 setup, 14 on the
chunk loop), then the stream.

The loader moves the decruncher to DECRUNCHER_START in screen RAM, relinks it and RUNs
it. That frees all of BASIC RAM for the output.

The decruncher never moves a chunk. A chunk is a set of ready-made arrays, and ARYTAB is pointed at it, so BASIC
just finds them: p$ (or p$ q$ r$ s$ for pairs), then the n% header. Each `t$=p$(k)+""` makes BASIC copy one part
to FRETOP-len, top down. That string copy is the whole decompressor. The last chunk links to page 0: then we are
done, restore the pointers and RUN the program.

Holding STOP only breaks once the program is ready to LIST.

The REM title uses inverse screen codes to show capitals in the uppercase charset.

The source carries `;` comment lines on the major structures. petcat skips them and tokenize() drops them, so they
never reach the PRG; `-d` keeps the source next to it.
"""
import functools, re, subprocess, tempfile
from dataclasses import dataclass
from pathlib import Path
from .memory import BASIC_START, COLOR_RAM, has_color_ram, real_fretop


@dataclass
class Layout:
    """Everything the generated code needs to know. The defaults have the most digits, which makes the code
    they produce an upper bound on its size."""
    end: int                     # output end, virtual
    stage: tuple                 # staging plan (memory.stage_plan)
    holes: list                  # merged holes, zero-width ones included
    kinds: tuple                 # (any pair chunk, any single chunk)
    basic_end: int               # VARTAB for the unpacked program
    paired: bool = True
    n_chunks: int = 1
    code_src: int = 255          # loader: top page of the decruncher in the PRG
    code_dst: int = 255          # ... and where it goes
    code_pages: int = 9
    stream_src: int = 0xFFFF     # stream start in the PRG
    stream_dst: int = 0xFFFF     # where its bottom is moved to
    moved_pages: int = 255
    stream_len: int = 0xFFFF
    vartab: int = 0xFFFF         # decruncher VARTAB
    first_chunk: int = 0xFFFF
    loader_pad: int = 0          # dead colons, see packer.try_layout()
    decruncher_pad: int = 0
    raw: tuple = None            # raw mode: (x, e) carried as is
    fill: bool = False           # some holes are filled by the ROM (memory.fill_runs)
    fill_runs: tuple = ()        # ... these: (start, end, byte)
    chunks: list = None


class Vars:
    """Decruncher variables and chunk limits.

    All scalars are created up front, in order of use: BASIC looks them up linearly, and nothing may be created
    later because ARYTAB points into the stream. The moved stream bottom follows them.

    The header holds counts and bar ends minus one, so the loops need no -1.

    Pair chunks do t$=q$(k)+p$(k), two parts per statement: ~25% faster, ~1% bigger.

    Progress bar: each chunk holds its cumulative end column (minus one) and advances the bar once after copying.
    PRINT is safe until output reaches colour RAM; then POKE avoids changing its colours. Both paths draw the same
    glyph and finish at column 40."""
    MAX_PER_CHUNK = 4096                       # n%(0) is 16 bits
    LEAN_CHUNKS = 8                            # this few chunks: header indices are literals, see below
    def __init__(self, staged, vt, poke, n_fields=None, n_chunks=None):
        # K also holds the next chunk's high byte and the fill page count, between copy loops.
        # Put the copy and bar variables first: BASIC searches scalars linearly.
        # Header index 1 is a literal; Y holds index 4 only when the header has one (None: assume it does).
        # U V Y E only make the once-per-chunk header reads faster. With few chunks (n_chunks, None: many)
        # their setup costs more bytes than the time is worth, so the reads use literals.
        index = [] if n_chunks is not None and n_chunks <= self.LEAN_CHUNKS else ['u', 'v', 'y', 'e']
        self.scalars = (['k', 't$'] + (['b'] if poke else []) + ['w']
                        + [n for n in index if n in 'uv' or n == 'y' and (n_fields is None or n_fields > 4)]
                        + ['f', 'g'] + (['e'] if staged and 'e' in index else []) + (['h'] if staged else []))
        self.end = vt + 7 * len(self.scalars)


def header_fields(kinds, holes, staged):
    """The n% header: count, next chunk, bar end, and only the flags this build needs. Chunks after a hole
    append the new FRETOP."""
    return (['m', 'lo', 'hi', 'bar'] + (['pair'] if kinds == (True, True) else []) + (['hole'] if holes else [])
            + (['copy'] if staged else []))


# Scalars holding n% indices; a literal where the scalar does not exist.
HEADER_INDEX_VARS = {0: '.', 1: 'o', 2: 'u', 3: 'v', 4: 'y', 5: 'e'}


TITLE = '   sfxbas'                       # leading space, then the name; rendered inverse
TITLE_SCREEN = bytes((b - 64 if 65 <= b <= 90 else b) | 0x80
                     for b in TITLE.upper().encode('ascii'))
BAR_ADDR = 0x0400                          # screen row 0, over the title
BAR_CHAR = 0x23                            # screen code
BAR_PRINT_CHAR = '{$23}'
BAR_WIDTH = 40
TITLE_LINE_ADDR = BAR_ADDR + BAR_WIDTH - len(TITLE) - 5  # link, number, REM; title ends at column 40
DECRUNCHER_START = TITLE_LINE_ADDR         # title first, then setup; no leading spaces or alignment colons


def poke_bar(end):
    """POKE the bar when the output reaches $D800, where PRINT would store its colour."""
    return end > COLOR_RAM


def lo(v): return v & 255
def hi(v): return (v >> 8) & 255


class PetcatError(RuntimeError):
    """The external BASIC tokenizer could not run successfully."""


def petcat(lines):
    """Tokenized bodies (link, line number, tokens, $00) of BASIC lines, in one petcat run: it takes many
    source files and with -nh writes their PRGs back to back."""
    with tempfile.TemporaryDirectory() as w:
        paths = []
        for i, line in enumerate(lines):
            paths.append(Path(w, '%d.bas' % i))
            paths[-1].write_text(line + '\n')
        try:
            raw = subprocess.run(['petcat', '-w2', '-nh', '--', *map(str, paths)],
                                 check=True, capture_output=True).stdout
        except FileNotFoundError as e:
            raise PetcatError('petcat not found; install VICE and put petcat on PATH') from e
        except OSError as e:
            raise PetcatError('cannot execute petcat: %s' % e) from e
        except subprocess.CalledProcessError as e:
            detail = e.stderr.decode(errors='replace').strip()
            raise PetcatError('petcat failed (exit %d)%s' % (e.returncode, ': ' + detail if detail else '')) from e
    bodies, p = [], 0
    for line in lines:
        end = raw.index(0, p + 6) + 1                   # load address, link, line number, body
        assert raw[end:end + 2] == b'\0\0', 'missing petcat program terminator'
        bodies.append(raw[p + 2:end])
        p = end + 2
    assert p == len(raw), 'unexpected trailing petcat output'
    return bodies


_LINES = {}                                         # BASIC line -> petcat body
LINE_CACHE_SIZE = 4096
_SHAPES = {}                                        # line with its numbers masked -> petcat body pieces, or None
_NUMBER = re.compile(r'\{[^}]*\}|(\d+)')            # a number outside {control codes}


def shape(line):
    """(line with its numbers masked, line number, the other numbers). Layouts differ mostly in numbers."""
    runs = []
    def mask(m):
        if m.group(1) is None: return m.group(0)
        runs.append(m.group(1))
        return '\0'
    masked = _NUMBER.sub(mask, line)
    return masked, int(runs[0]), runs[1:]


def learn_shape(line, body):
    """Remember body's pieces between the numbers, if they come out of petcat as the same digits, in order."""
    masked, n, runs = shape(line)
    if masked in _SHAPES: return
    tail = body[4:]                                 # after link and line number
    digits = list(re.finditer(rb'[0-9]+', tail))
    ok = body[2:4] == bytes([lo(n), hi(n)]) and [m.group().decode() for m in digits] == runs
    cuts = [0] + [x for m in digits for x in m.span()] + [len(tail)]
    _SHAPES[masked] = [body[:2]] + [tail[a:b] for a, b in zip(cuts[0::2], cuts[1::2])] if ok else None


def from_shape(line):
    """line's petcat body from a learned line of the same shape, or None."""
    masked, n, runs = shape(line)
    pieces = _SHAPES.get(masked)
    if pieces is None: return None
    out = [pieces[0], bytes([lo(n), hi(n)]), pieces[1]]
    for run, piece in zip(runs, pieces[2:]):
        out += [run.encode(), piece]
    return b''.join(out)


@functools.lru_cache(maxsize=256)
def tokenize(text, base):
    """Tokenize text and relink it for base. Lines are cached: most layout changes touch only a few, and those
    mostly change numbers, so a line shaped like one petcat already did is put together without it."""
    lines = [line for line in text.split('\n') if line.strip() and not line.startswith(';')]
    if len(_LINES) > LINE_CACHE_SIZE: _LINES.clear()
    missing = []
    for line in dict.fromkeys(lines):
        if line in _LINES: continue
        body = from_shape(line)
        if body is None: missing.append(line)
        else: _LINES[line] = body
    if missing:
        _LINES.update(zip(missing, petcat(missing)))
        for line in missing: learn_shape(line, _LINES[line])
    code = bytearray(b''.join(_LINES[line] for line in lines) + b'\0\0')
    a = 0
    while code[a] | code[a + 1]:
        e = code.index(0, a + 4) + 1
        code[a], code[a + 1] = lo(base + e), hi(base + e)
        a = e
    return bytes(code)


def loader(v):
    # 809: STOP vector $F6ED -> $FFED, which returns "not pressed".
    # Move the decruncher into screen RAM ($A3EC), relink it ($A533) and RUN it. I/O is still banked in here,
    # even for colour output. Trailing colons pad, see try_layout().
    return ('; disable STOP, copy the decruncher to screen memory, relink it there and run it\n'
            '0 poke809,255:poke90,%d:poke91,%d:poke88,%d:poke89,%d:poke781,%d:poke782,.:sys41964'
            ':poke43,%d:poke44,%d:sys42291:poke782,.:poke45,%d:poke46,%d:print"{home}";:run%s\n'
            % (lo(v.code_src), hi(v.code_src), lo(v.code_dst), hi(v.code_dst), v.code_pages,
               lo(DECRUNCHER_START), hi(DECRUNCHER_START), lo(v.vartab), hi(v.vartab), ':' * v.loader_pad))


def decruncher(v):
    staged = bool(v.stage)
    poke = poke_bar(v.end)
    fields = header_fields(v.kinds, bool(v.holes), staged)
    f = Vars(staged, v.vartab, poke, len(fields), v.n_chunks)
    ft = real_fretop(v.end, v.stage)
    # restore basic registers and enable runstop, then run the extracted program
    finish = ('poke43,1:poke44,8:poke45,%d:poke46,%d:poke1,55:%spoke56333,129:poke214,23:poke809,246:run'
              % (lo(v.basic_end), hi(v.basic_end), 'poke3,170:' if v.fill else ''))
    assert v.n_chunks, 'no chunks: the stream always copies the $0800 zero'
    g, ng = move_groups(v)
    ix = {n: HEADER_INDEX_VARS[i] if i == 0 or HEADER_INDEX_VARS.get(i) in f.scalars else str(i)
          for i, n in enumerate(fields)}
    th = len(fields)                                     # hole chunks: new FRETOP at n%(th), n%(th+1)
    # Stream move constants are scalars only if the loop runs more than twice: BASIC parses a literal every time.
    # Scalars start with their move value and get their chunk value afterwards.
    # Create them before poking $58/$5A: creating a variable runs BASIC's block move, which uses those.
    col = has_color_ram(v.end)
    chunk = dict(o=1, u=2, v=3, y=4, f=47, g=48, b=BAR_CHAR, w=BAR_ADDR if poke else 0)
    if len(fields) > 5: chunk['e'] = 5
    if staged: chunk['h'] = hi(ft) + (lo(ft) > 0)
    move_vals = dict(o=41964, u=91, v=89, y=781, f=hi(v.stream_src) - hi(v.stream_dst), g=g)
    mv = {} if ng <= 2 else {n: x for n, x in move_vals.items() if n in f.scalars}
    ref = lambda n: n if n in mv else basic_number(move_vals[n])
    lit = lambda x: basic_number(x) if x else '.'                 # '.' is the fastest zero
    create = ':'.join(n + '=' + ('""' if n == 't$' else lit(mv.get(n, chunk.get(n, 0)))) for n in f.scalars)
    if ng == 1:                                          # one group: no loop
        top = hi(v.stream_src) + g - 1
        move = 'poke90,%d:poke88,%d:poke91,%d:poke89,%d:poke781,%d:sys41964' % (
            lo(v.stream_src), lo(v.stream_dst), top, top - move_vals['f'], g)
    else:
        move = 'poke90,%d:poke88,%d:fork=%dto%d%s:poke%s,k:poke%s,k-%s:poke%s,%s:sys%s:next' % (
            lo(v.stream_src), lo(v.stream_dst), hi(v.stream_src) + g - 1, hi(v.stream_src) + g * ng - 1,
            'step' + ref('g') if g != 1 else '', ref('u'), ref('v'), ref('f'), ref('y'), ref('g'), ref('o'))
    consts = ''.join('%s=%s:' % (n, basic_number(chunk[n])) for n in f.scalars if n in mv)
    # Mask the CIA1 IRQ: it is dead time, and its keyboard scan would read char ROM. Bank I/O out unless
    # colour RAM comes first. STREND hi = 0 keeps garbage collection off.
    setup = [x for x in ['poke56333,127', '' if col else 'poke1,51'] + create.split(':') + move.split(':')
             + consts.split(':') +
             ['poke47,%d' % lo(v.first_chunk), 'poke48,%d' % hi(v.first_chunk),
              'poke51,%d' % lo(ft), 'poke52,%d' % hi(ft),
              'poke50,.'] if x]
    lines = [
        # The title comes first: its header sits in row 0, setup starts below it. The bar may later overwrite
        # the title's links, so the chunk loop uses NEXT rather than walking them again with GOTO.
        '10 rem' + TITLE,
        '; setup: create the variables, move the stream bottom up, point ARYTAB at the first chunk',
        '12 ' + ':'.join(setup),
        # A zero-step loop never ends and preserves its variable: G stays 48 for the link.
        '; chunk loop: one pass per chunk, n% is its header',
        '14 forg=48to.step.',
        # Staging segment done: copy it up from the window. Read n% first, since the array lookup itself
        # clobbers $58/$59.
        *(['; n%%(%d) copy: staging segment done, copy it up from the window' % fields.index('copy'),
           '15 ifn%%(%s)thenpoke89,n%%(%s):poke88,.:poke90,.:poke91,h-1:poke781,h-peek(52):poke782,.:sys41964'
           % (ix.get('copy'), ix.get('copy'))] if staged else []),
        # Hole: jump FRETOP. h remembers the window top for the copy above.
        # Filled hole (n% = 2): DIM's clear loop at $B2C2 fills [($58), ($58) + 256 * ($72) + ($71)) top down with A,
        # here from the new FRETOP up. $58-$5B and $5C-$60 are float work registers: array lookups overwrite
        # $58/$59 and $71/$72, so read N into k (free here: the loop lines start it again) and L into $71 after
        # the final array lookup. PEEK does not overwrite $71/$72.
        # It returns only with $0C set (any variable lookup clears it), then writes 2 bytes at ($5F)+2:
        # the SYS argument leaves $5F = 0, so $02/$03.
        *(['; n%%(%d) hole: set the new FRETOP from n%%(%d), n%%(%d)%s'
           % (fields.index('hole'), th, th + 1, '; n%%(%d)>1 fills it' % fields.index('hole') if v.fill else ''),
           '%d ifn%%(%s)thenpoke51,n%%(%d):poke52,n%%(%d)' % (15 + staged, ix.get('hole'), th, th + 1)
           + (':poke1,51' if col else '') + (':h=peek(52)-(peek(51)>.)' if staged else '')
           + (':ifn%%(%s)>1thenk=n%%(%d):poke780,n%%(%d):poke113,n%%(%d):poke88,peek(51):poke89,peek(52)+k'
              ':poke114,k:poke12,1:sys45762' % (ix.get('hole'), th + 4, th + 2, th + 3) if v.fill else '')]
          if v.holes else []),
        # NEXT, not GOTO: GOTO walks the line links, and the bar overwrites line 10's.
        *part_loop(v, 16 + staged, ix)]
    # The last chunk links to page 0: a false IF skips the outer NEXT and falls through to the finish line.
    last = int(lines[-1].split(' ', 1)[0])
    lines.append('; done: restore the BASIC pointers, memory banks and IRQ, cursor to the bottom, restore STOP, run the program')
    lines.append('%d %s%s' % (last + 1, finish, ':' * v.decruncher_pad))
    return '\n'.join(lines) + '\n'


def move_groups(v):
    """(g, ng): move the stream bottom in ng $A3EC calls of g pages.

    $A3EC copies top page first, so g pages must not overlap their own destination. The last group may copy a
    few pages too many; they land in dead space."""
    g = max(1, min(v.moved_pages, (v.stream_src - v.stream_dst) // 256))
    return g, -(-v.moved_pages // g)


def part_loop(v, n, ix):
    """Copy a chunk, update its bar, then link. Mixed chunks select one copy
    path before the common bar/link lines, so both paths share the exit."""
    kinds = v.kinds
    at = {f: i for i, f in enumerate(header_fields(kinds, bool(v.holes), bool(v.stage)))}   # for the comments
    pair, single = 'q$(k)+p$(k):t$=s$(k)+r$(k)', 'p$(k)+""'
    def copy(stmt, test=''):
        return test + 'fork=.ton%(.):t$=' + stmt + ':next'
    if kinds == (True, True):
        flag = 'n%%(%s)' % ix['pair']
        lines = ['; n%%(%d) pair: copy two parts per statement, else one' % at['pair'],
                 copy(pair, 'if' + flag + 'then'), copy(single, 'if' + flag + '=.then')]
    else:
        lines = ['; copy the chunk: n%%(0)+1 statements of %s' % ('two parts' if kinds[0] else 'one part'),
                 copy(pair if kinds[0] else single)]
    end = 'n%%(%s)' % ix['bar']
    draw = 'pokew,b' if poke_bar(v.end) else 'print"%s";' % BAR_PRINT_CHAR
    # NEXT leaves W at the first undrawn column. Keep IF: BASIC executes even an empty FOR range once.
    lines += ['; n%%(%d) bar: advance the progress bar to this column' % at['bar'],
              'if%swthenforw=wto%s:%s:next' % (end + '>=', end, draw),
              # K is free after the copy/bar loops; read both bytes first. Page 0 ends: ARYTAB then points there,
              # but only K is looked up, and it is the first scalar, so BASIC's search finds it before ARYTAB.
              '; n%%(%d), n%%(%d) next chunk: point ARYTAB at it; page 0 ends the loop' % (at['lo'], at['hi']),
              'k=n%%(%s):pokef,n%%(%s):pokeg,k:ifkthen next' % (ix['hi'], ix['lo'])]
    out = []
    for x in lines:
        if x.startswith(';'): out.append(x)
        else: out.append('%d %s' % (n, x)); n += 1
    return out


def basic_number(x):
    """Shortest BASIC literal for x."""
    return ('%.8g' % x).lstrip('0') or '0'


def string_array(name, elems):
    """String array, laid out the way DIM does it."""
    n = len(elems) // 3
    return bytes([ord(name), 0x80, lo(7 + 3 * n), hi(7 + 3 * n), 1, hi(n), lo(n)]) + elems


def integer_array(name, values):
    """Integer array, laid out the way DIM does it. Elements are big-endian."""
    n = len(values)
    return bytes([ord(name) | 0x80, 0x80, lo(7 + 2 * n), hi(7 + 2 * n), 1, hi(n), lo(n)]) + \
        b''.join(bytes([hi(x), lo(x)]) for x in values)


def program(v):
    return loader(v) + decruncher(v)


def tokenized_program(v):
    """Tokenized program. Line 10's REM text doubles as screen codes, so the moved copy shows the title."""
    code = bytearray(tokenize(program(v), BASIC_START))
    first = a = code[0] + 256 * code[1] - BASIC_START
    while code[a + 2] + 256 * code[a + 3] != 10:
        a = code[a] + 256 * code[a + 1] - BASIC_START
    assert DECRUNCHER_START + a - first == TITLE_LINE_ADDR, 'title line at $%04X' % (DECRUNCHER_START + a - first)
    a += 5                                               # link, line number, REM
    assert code[a - 1] == 0x8f and code.index(0, a) - a == len(TITLE), 'wrong title length'
    code[a:a + len(TITLE)] = TITLE_SCREEN
    return bytes(code)
