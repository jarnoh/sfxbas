"""Run a BASIC test program on real VICE (x64sc, stock ROMs) and dump memory / time it.

Markers: `sys 53232` = timer START (RTS placed at $CFF0 by the program via T0),
         `sys 53240` = END: dumps requested RAM ranges and quits VICE.
Cycle counts come from the monitor's register line printed at each break.
"""
import os, re, subprocess, tempfile, shutil
from pathlib import Path

X64SC = '/opt/homebrew/bin/x64sc'
# Extra x64sc arguments, e.g. another KERNAL: SFXBAS_VICE_ARGS='-kernal /abs/kernal.rom -parallel8 1'
EXTRA = os.environ.get('SFXBAS_VICE_ARGS', '').split()
M0, M2, M1 = 0xCFF0, 0xCFF4, 0xCFF8
INIT = 'poke 53232,96:poke 53236,96:poke 53240,96:'      # put RTS at both markers (put at program start)
T0 = 'sys 53232'
T1 = 'sys 53236'      # optional: stop the timer before post-processing
END = 'sys 53240'

class Result(dict):
    cycles = None      # cycles between T0 and END markers (incl. two SYS overheads)
    traces = None      # {address: [cycle counts]} for trace=

def run(basic, dumps=(), screen=False, cycles=400_000_000, stop_at=None, raw_prg=None, stop_hit=1, trace=()):
    """basic: lowercase BASIC (petcat syntax). Lines should call T0/END markers; INIT is prepended as line 1
    unless stop_at is used. dumps: [(start,end_incl)]. Returns Result {(s,e):bytes, 'screen':bytes}.
    stop_hit: dump at the n-th execution of stop_at. trace: addresses whose executions are timed."""
    w = tempfile.mkdtemp(prefix='c64t_')
    try:
        src, prg = os.path.join(w, 't.bas'), os.path.join(w, 't.prg')
        if raw_prg is not None:
            Path(prg).write_bytes(raw_prg)          # ready-made PRG (byte-exact token streams)
        else:
            if stop_at is None:
                basic = '1 ' + INIT.rstrip(':') + '\n' + basic
            Path(src).write_text(basic.rstrip('\n') + '\n')
            subprocess.run(['petcat', '-w2', '-o', prg, '--', src], check=True, capture_output=True)
        want = list(dumps) + ([(0x0400, 0x07E7)] if screen else [])
        end = M1 if stop_at is None else stop_at
        lines = []
        if stop_at is None:
            lines += ['trace exec %x' % M0, 'trace exec %x' % M2]
        lines += ['trace exec %x' % a for a in trace]
        n = len(lines)
        def cp(cmd):
            nonlocal n
            n += 1
            lines.extend(['break %x' % end, 'command %d "%s"' % (n, cmd)])
            if stop_hit > 1: lines.append('ignore %d %d' % (n, stop_hit - 1))
        for i, d in enumerate(want):
            a, b = d[0], d[1]
            # Each checkpoint resets the bank: select it in the SAVE command.
            save = 'save \\"%s/d%d.bin\\" 0 %x %x' % (w, i, a, b)
            if len(d) > 2: save = 'bank %s; %s; bank default' % (d[2], save)
            cp(save)
        cp('quit')
        Path(w, 'mon.txt').write_text('\n'.join(lines) + '\n')
        log = subprocess.run([X64SC, *EXTRA, '-console', '-warp', '+sound', '-autostartprgmode', '1',
                              '-limitcycles', str(cycles), '-moncommands', os.path.join(w, 'mon.txt'),
                              '-autostart', prg], capture_output=True, text=True, cwd=w, stdin=subprocess.DEVNULL, timeout=600)
        txt = log.stdout + log.stderr
        res = Result()
        for i, key in enumerate(want):
            p = os.path.join(w, 'd%d.bin' % i)
            if not os.path.exists(p):
                raise RuntimeError('no dump (did the test reach the end marker?)\n' + txt[-1800:])
            res[key] = Path(p).read_bytes()[2:]
        if screen: res['screen'] = res.pop((0x0400, 0x07E7))
        c = {}
        for m, name in ((M0, 't0'), (M2, 't1'), (M1, 't1e')):
            f = re.findall(r'\.C:%04x .*?(\d+)\s*$' % m, txt, re.M)
            if f: c[name] = int(f[-1])
        if 't1' not in c and 't1e' in c: c['t1'] = c['t1e']
        if 't0' in c and 't1' in c: res.cycles = c['t1'] - c['t0']
        res.traces = {a: sorted({int(x) for x in re.findall(r'\.C:%04x .*?(\d+)\s*$' % a, txt, re.M)})
                      for a in trace}
        return res
    finally:
        shutil.rmtree(w, ignore_errors=True)

SCR = '@abcdefghijklmnopqrstuvwxyz[£]↑←' + ' !"#$%&\'()*+,-./0123456789:;<=>?'
def screen_rows(scr):
    ch = lambda b: SCR[b & 0x7F] if (b & 0x7F) < 64 else '?'
    return [''.join(ch(b) for b in scr[r*40:(r+1)*40]).rstrip() for r in range(25)]
