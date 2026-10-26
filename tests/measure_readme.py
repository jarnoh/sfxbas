"""Rebuild and verify the README comparison on PAL VICE.

python3 tests/measure_readme.py                 # print the measured tables
python3 tests/measure_readme.py --update-readme # also replace the README section

Needs petcat, x64sc and exomizer on the host. Exomizer's superbox1 input is a
temporary copy with its last eight bytes removed; the original stays intact.
"""
import argparse
import contextlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vice
from test_vice import RUN, unpack_on_vice
from sfxbas.cli import build_speed
from sfxbas.packer import Options
from sfxbas.simulator import PAL_HZ

NAMES = ('amigabasic', 'digi', 'parallaxative', 'pubquest', 'superbox1', 'superscroller', 'yleradio')
LOAD_RATE = .4 * 1024
BASIC_EXECUTE = 0xA7AE


def measure(name, work, exomizer):
    source = ROOT / 'examples' / (name + '.prg')
    original = source.read_bytes()
    data = original[2:]
    print(f'{name}: building sfxbas -speed=0.4', file=sys.stderr, flush=True)
    with (work / (name + '-sfx.log')).open('w') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            packed = build_speed(data, Options(speed=.4), .4)
    bad, pointers, cycles = unpack_on_vice(packed, data)
    if bad or pointers != (0x0801, packed.layout.basic_end):
        raise RuntimeError(f'{name}: sfxbas output verification failed: {bad[:8]}, {pointers}')

    trim = 8 if name == 'superbox1' else 0
    native_input = original[:-trim] if trim else original
    native_source = work / (name + '.prg')
    native_source.write_bytes(native_input)
    native_output = work / (name + '-exo.prg')
    print(f'{name}: building and measuring Exomizer', file=sys.stderr, flush=True)
    result = subprocess.run([exomizer, 'sfx', 'basic', str(native_source), '-o', str(native_output)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{name}: Exomizer failed\n{result.stdout}\n{result.stderr}')
    expected = native_input[2:]
    memory = (0x0801, 0x0800 + len(expected), 'ram')
    # RUN enters BASIC_EXECUTE once to launch the native loader. Its second
    # entry is the original program's start, after native decompression.
    native = vice.run(None, raw_prg=native_output.read_bytes(), stop_at=BASIC_EXECUTE,
                      stop_hit=2, dumps=[memory], trace=[RUN, BASIC_EXECUTE],
                      cycles=100_000_000)
    if native[memory] != expected:
        raise RuntimeError(f'{name}: Exomizer output verification failed')
    if len(native.traces[RUN]) != 1 or len(native.traces[BASIC_EXECUTE]) != 2:
        raise RuntimeError(f'{name}: unexpected Exomizer BASIC entry trace')
    native_cycles = native.traces[BASIC_EXECUTE][1] - native.traces[RUN][0]
    row = dict(name=name, original=len(original), sfxbas=len(packed.prg) + 2,
               sfxbas_cycles=cycles, sfxbas_seconds=cycles / PAL_HZ, lambda_=packed.lam,
               exomizer=native_output.stat().st_size, exomizer_cycles=native_cycles,
               exomizer_seconds=native_cycles / PAL_HZ, exomizer_trim=trim)
    print(f'{name}: {row["sfxbas"]} bytes, {cycles} cycles; RAM checks passed',
          file=sys.stderr, flush=True)
    return row


def tables(rows, version):
    lines = [
        '## Comparison with native cruncher', '',
        'All examples, packed with `sfxbas -speed=0.4` (best load + unpack at stock 1541 speed)',
        f'and with Exomizer {version} (`exomizer sfx basic`). Sizes are file sizes in bytes, load address included.',
        'Exomizer is a machine code decruncher with bit-level coding: smaller and faster. sfxbas stays pure BASIC.', '',
        'Size and load time at about 0.4 kB/s (a stock 1541):', '',
        '| name | original | load | exomizer | load | sfxbas | load | factor |',
        '|---|--:|--:|--:|--:|--:|--:|--:|',
    ]
    for r in rows:
        high = '† ' if 0x0801 + r['original'] - 2 > 0xD000 else ''
        trimmed = '† ' if r['exomizer_trim'] else ''
        lines.append(f'| {r["name"]} | {r["original"]} | {high}{r["original"] / LOAD_RATE:.1f} s | '
                     f'{trimmed}{r["exomizer"]} | {r["exomizer"] / LOAD_RATE:.1f} s | '
                     f'{r["sfxbas"]} | {r["sfxbas"] / LOAD_RATE:.1f} s | {r["sfxbas"] / r["exomizer"]:.2f}× |')
    total = {key: sum(r[key] for r in rows) for key in
             ('original', 'exomizer', 'exomizer_cycles', 'exomizer_seconds', 'sfxbas', 'sfxbas_cycles', 'sfxbas_seconds')}
    lines.append(f'| **total** | {total["original"]} | {total["original"] / LOAD_RATE:.1f} s | '
                 f'{total["exomizer"]} | {total["exomizer"] / LOAD_RATE:.1f} s | '
                 f'{total["sfxbas"]} | {total["sfxbas"] / LOAD_RATE:.1f} s | {total["sfxbas"] / total["exomizer"]:.2f}× |')
    lines += ['', 'Size and decrunch time, measured from `RUN` until the unpacked program starts:', '',
              '| name | original | exomizer | decrunch | sfxbas | decrunch | factor |',
              '|---|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        trimmed = '† ' if r['exomizer_trim'] else ''
        lines.append(f'| {r["name"]} | {r["original"]} | {trimmed}{r["exomizer"]} | '
                     f'{trimmed}{r["exomizer_seconds"]:.1f} s | {r["sfxbas"]} | '
                     f'{r["sfxbas_seconds"]:.1f} s | {r["sfxbas_cycles"] / r["exomizer_cycles"]:.2f}× |')
    lines.append(f'| **total** | {total["original"]} | {total["exomizer"]} | {total["exomizer_seconds"]:.1f} s | '
                 f'{total["sfxbas"]} | {total["sfxbas_seconds"]:.1f} s | '
                 f'{total["sfxbas_cycles"] / total["exomizer_cycles"]:.2f}× |')
    lines += ['', 'factor: sfxbas divided by exomizer (size in the first table, decrunch time in the second).', '',
              '† Original load times for high-memory programs are theoretical. For superbox1, the Exomizer',
              'input omits the final eight bytes to avoid its $10000 end-address limitation; sfxbas uses the full input.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--update-readme', action='store_true')
    parser.add_argument('--json', type=Path, help='also save exact byte counts, cycles and seconds as JSON')
    parser.add_argument('--exomizer', default='exomizer', help='Exomizer executable (default: exomizer)')
    args = parser.parse_args()
    for executable in ('petcat', vice.X64SC, args.exomizer):
        if not shutil.which(executable):
            parser.error(f'missing executable: {executable}')
    result = subprocess.run([args.exomizer, '-v'], capture_output=True, text=True, check=True)
    version = re.search(r'Exomizer v([^\s]+)', result.stdout + result.stderr)
    if not version:
        parser.error('cannot read Exomizer version')
    vice.EXTRA = [*vice.EXTRA, '-pal']
    with tempfile.TemporaryDirectory(prefix='readme-measure-') as temp:
        rows = [measure(name, Path(temp), args.exomizer) for name in NAMES]
    report = tables(rows, version.group(1))
    if args.json:
        args.json.write_text(json.dumps(dict(exomizer_version=version.group(1), pal_hz=PAL_HZ,
                                            load_bytes_per_second=LOAD_RATE, results=rows), indent=2) + '\n')
    if args.update_readme:
        readme = ROOT / 'README.md'
        text = readme.read_text()
        start = text.index('## Comparison with native cruncher')
        end = text.index('## Further reading', start)
        readme.write_text(text[:start] + report + '\n' + text[end:])
    print(report)


if __name__ == '__main__':
    main()
