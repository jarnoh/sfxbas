"""Parallel unittest runner. --fast skips golden sweeps and large emulator inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
FAST_VICE = {'test_loader_title_position', 'test_progress_bar_repeated_endpoints',
             'test_digi_small', 'test_fill_partial_pages'}


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


def run_case(name):
    started = time.monotonic()
    env = dict(os.environ, TMPDIR='/tmp', SFXBAS_TEST_JOBS='1', SFXBAS_JOBS='1')
    result = subprocess.run([sys.executable, '-m', 'unittest', name], cwd=HERE,
                            env=env, capture_output=True, text=True)
    return name, result, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('SFXBAS_TEST_JOBS', '4')))
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(HERE))
    names = [test.id() for test in flatten(suite)
             if not args.fast or '.Matcher.' in test.id() or '.Simulator.' in test.id()
             or '.PlacementCache.' in test.id() or '.CycleParser.' in test.id() or '.SizeSearch.' in test.id()
             or '.Refinement.' in test.id() or '.SourceParser.' in test.id() or '.Tokenizer.' in test.id()
             or ('.Vice.' in test.id() and test._testMethodName in FAST_VICE)]
    started = time.monotonic()
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        pending = [pool.submit(run_case, name) for name in names]
        for task in as_completed(pending):
            name, result, elapsed = task.result()
            status = 'FAIL' if result.returncode else 'SKIP' if 'OK (skipped=' in result.stderr else 'PASS'
            print('%s %s (%.1fs)' % (status, name, elapsed), flush=True)
            if result.returncode:
                failed += 1
                print(result.stdout + result.stderr, flush=True)
    print('%d tests, %d failed, %.1fs' % (len(names), failed, time.monotonic() - started))
    return bool(failed)


if __name__ == '__main__':
    sys.exit(main())
