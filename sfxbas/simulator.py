"""Models of the decruncher in basic.py.

simulate() runs it on a 64K image and says whether the output comes out right. estimate() says how long it takes.
Change the decruncher and both need changing.
"""
from .basic import BAR_WIDTH, Vars, poke_bar, move_groups, header_fields, hi, lo
from .memory import (BASIC_START, CHAR_ROM_ADDR, COLOR_RAM, COLOR_RAM_SIZE, VIRTUAL_COLOR_RAM, has_color_ram,
                     readable, real_fretop, subtract_ranges, windows_used)


class Missing(Exception): pass                          # BASIC would not find the array element


POWER_ON = bytes((a * 37 + 11) & 255 for a in range(65536))   # RAM is not zero at power-on
_can_read = {}


def can_read_table(io, rom):
    """Byte per address (with room past $FFFF for a 255-byte read): 1 where the string copy can read."""
    if (io, rom) not in _can_read:
        tb = bytearray(65536 + 256)
        for a, b in ((0, 0xA000), (0xC000, 0xD000)) + (((COLOR_RAM, COLOR_RAM + COLOR_RAM_SIZE),) if io else
                                                        ((CHAR_ROM_ADDR, CHAR_ROM_ADDR + 4096),) if rom else ()):
            tb[a:b] = bytes([1]) * (b - a)
        _can_read[io, rom] = bytes(tb)
    return _can_read[io, rom]


def simulate(prg, v, t, charrom=None):
    """Run the loader and the chunk loop. True if the output is exact."""
    holes = v.holes
    no_write = subtract_ranges(holes, windows_used(v.stage))
    mem = bytearray(POWER_ON)
    if not readable(v.code_src - 256 * (v.code_pages - 1), 256 * v.code_pages): return False
    if not readable(v.stream_src, v.stream_len): return False
    mem[BASIC_START:BASIC_START + len(prg)] = prg
    def page(src, dst):                                  # $A3EC: one page, top byte first
        if readable(src, 256) and dst + 256 <= 65536 and (dst >= src or dst + 256 <= src):
            mem[dst:dst + 256] = mem[src:src + 256]      # same as top byte first: no overlap, or moving up
            return
        for y in range(255, -1, -1):
            a = src + y
            mem[dst + y] = mem[a] if readable(a, 1) else (a * 7 + 99) & 255
    for k in range(v.code_pages):                             # loader: move the decruncher
        page(v.code_src - 256 * k, v.code_dst - 256 * k)
    g, ng = move_groups(v)
    if not readable(v.stream_src, 256 * g * ng): return False
    for j in range(ng):                                  # setup: move the stream bottom
        for k in range(g * (j + 1) - 1, g * j - 1, -1):
            page(v.stream_src + 256 * k, v.stream_dst + 256 * k)
    fre, ary = real_fretop(v.end, v.stage), v.first_chunk
    h = hi(fre) + (lo(fre) > 0)
    io = has_color_ram(v.end)                              # I/O stays on for colour RAM
    colram = bytearray((a * 5 + 3) & 255 for a in range(COLOR_RAM_SIZE))
    def rd(a):
        return colram[a - COLOR_RAM] if io and COLOR_RAM <= a < COLOR_RAM + COLOR_RAM_SIZE else \
            charrom[a - CHAR_ROM_ADDR] if not io and charrom and CHAR_ROM_ADDR <= a < CHAR_ROM_ADDR + 4096 else mem[a]
    def can(a, n):
        return can_read_table(io, bool(charrom)).find(0, a, a + n) < 0
    def element(name, k):                                # BASIC's array search from ARYTAB
        ptr, key, size = ary, bytes([ord(name), 0x80]), 3
        if name == 'N': key, size = bytes([ord(name) | 0x80, 0x80]), 2
        for _ in range(5):
            if not readable(ptr, 7): return None
            if mem[ptr:ptr + 2] == key:
                e = ptr + 7 + size * k
                ok = mem[ptr + 4] == 1 and 256 * mem[ptr + 5] + mem[ptr + 6] > k and readable(e, size)
                return e if ok else None
            ptr += mem[ptr + 2] + 256 * mem[ptr + 3]
        return None
    def nk(k):                                            # n%(k)
        e = element('N', k)
        if e is None: raise Missing
        x = 256 * mem[e] + mem[e + 1]
        return x - 65536 if x >= 32768 else x
    fields = header_fields(v.kinds, bool(holes), bool(v.stage))
    fi = {n: i for i, n in enumerate(fields)}
    th = len(fields)                                     # hole chunks: new FRETOP, then the fill
    def word(k): return nk(k) + 256 * nk(k + 1)
    cols = 0
    try:
        while True:
            m = nk(fi['m']) + 1                          # stored minus one
            d = nk(fi['copy']) if 'copy' in fi else 0
            hole = nk(fi['hole']) if 'hole' in fi else 0
            pair_flag = nk(fi['pair']) if 'pair' in fi else 1
            bar = nk(fi['bar']) + 1 - (1024 if poke_bar(v.end) else 0)
            if not cols <= bar <= BAR_WIDTH: return False
            cols = bar
            if d:                                        # staging segment done: copy it up, top page first
                if not v.stage or lo(fre): return False
                n = h - hi(fre)
                if not 0 < n <= d + 1: return False
                for k in range(n):
                    src, dst = 256 * (h - 1 - k), 256 * (d - k)
                    if not readable(src, 256): return False
                    mem[dst:dst + 256] = mem[src:src + 256]
            if hole == 2:                                # $B2C2 from the new FRETOP up
                byte, l71, n72 = (nk(th + i) for i in range(2, 5))
                f0 = word(th)
                f1 = f0 + 256 * n72 + l71
                if not 0 <= byte < 256 or not any(h0 <= f0 and f1 <= h1 for h0, h1 in holes): return False
                mem[f0:f1] = bytes([byte]) * (f1 - f0)
            if hole:
                fre = word(th) or 0x10000
                io = False
                h = hi(fre) + (lo(fre) > 0)
            if not 1 <= m <= 2 * Vars.MAX_PER_CHUNK: return False
            pairs = v.kinds[0] and (not v.kinds[1] or pair_flag)
            stmts = [(k, names) for k in range(m) for names in ((('P', 'Q'), ('R', 'S')) if pairs else (('P',),))]
            for k, names in stmts:
                ds = []
                for name in names:
                    e = element(name, k)
                    if e is None or not readable(e, 3): return False
                    ds.append((mem[e], mem[e + 1] + 256 * mem[e + 2]))
                (Lu, su), (Ld, sd) = ds[0], ds[1] if pairs else (0, 0)
                L = Lu + Ld
                if not Lu and Ld or L > 255 or fre - L < BASIC_START - 1: return False
                if not can(su, Lu) or not can(sd, Ld): return False
                if io and not COLOR_RAM <= fre - L: return False
                if not io and any(fre - L < h1 and fre > h0 for h0, h1 in no_write): return False
                fre -= L
                for dst, src, n_ in ((fre, sd, Ld), (fre + Ld, su, Lu)):   # CAT: lower part first
                    if not io and (src >= dst + n_ or src + n_ <= dst) and \
                            not (charrom and src < CHAR_ROM_ADDR + 4096 and src + n_ > CHAR_ROM_ADDR):
                        mem[dst:dst + n_] = mem[src:src + n_]    # no overlap: same as byte by byte
                        continue
                    for i in range(n_ - 1, -1, -1):
                        if io: colram[dst + i - COLOR_RAM] = rd(src + i) & 15 | (dst * 3 & 240)
                        else: mem[dst + i] = rd(src + i)
            ary = word(fi['lo'])
            if not hi(ary): break                        # page 0: done
    except Missing:
        return False
    if v.chunks and cols != BAR_WIDTH: return False
    main, colour = t[:VIRTUAL_COLOR_RAM - BASIC_START], t[VIRTUAL_COLOR_RAM - BASIC_START:]
    return mem[BASIC_START:BASIC_START + len(main)] == main \
        and all(colram[i] & 15 == b & 15 for i, b in enumerate(colour)) \
        and fre == BASIC_START - 1 and mem[BASIC_START - 1] == 0


PAL_HZ = 985248
# Cycles per operation, fitted approximately to PAL emulator runs.
CYCLE_COST = dict(const=200000, fill=25000, fill_byte=11, iter=14198, single=5633,
                  chunk=28000, byte=21.1, page=4400, group=12930, staged=17.2,
                  seg=97553, bar_char=4100, print_char=2600)


def chunk_cycles(ch, single, fill=None, staged=0):
    """Cycles for one chunk at the rates estimate() uses: the staged copy and fill it starts with (staged: bytes
    copied up; fill: (start, end, byte)), then its parts."""
    c = CYCLE_COST
    return (c['chunk'] + (c['single'] * len(ch) if single else c['iter'] * ((len(ch) + 1) // 2))
            + c['byte'] * sum(u.length + (d.length if d else 0) for u, d in ch)
            + (c['fill'] + c['fill_byte'] * (fill[1] - fill[0]) if fill else 0)
            + (c['seg'] + c['staged'] * staged if staged else 0))


def estimate(v, parts):
    """Cycles from RUN to the unpacked program's RUN."""
    ch, c = v.chunks, CYCLE_COST
    fl = [f1 - f0 for f0, f1, b in v.fill_runs]
    return (c['const']
            + c['fill'] * len(fl) + c['fill_byte'] * sum(fl)
            + c['iter'] * sum((len(x) + 1) // 2 for x, skip, single in ch if not single)
            + c['single'] * sum(len(x) for x, skip, single in ch if single)
            + c['chunk'] * len(ch)
            + c['byte'] * sum(p.length for p in parts if p.kind != 'skip')
            + c['page'] * v.moved_pages + c['group'] * move_groups(v)[1]
            + c['staged'] * sum(vt - vb for vb, vt, w in v.stage) + c['seg'] * len(v.stage)
            + c['bar_char' if poke_bar(v.end) else 'print_char'] * (BAR_WIDTH if ch else 0))
