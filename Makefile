PYTHON   ?= python3
SFXFLAGS ?= -d -bar=7 2
MAKEFLAGS := --jobs=4


SOURCES := $(wildcard examples/*.prg)
TARGETS := $(patsubst examples/%.prg,build/%_sfx.prg,$(SOURCES))
DEPS    := $(wildcard sfxbas/*.py)

.PHONY: all clean test test-fast test-slow

all: $(TARGETS)

build/%_sfx.prg: examples/%.prg $(DEPS) | build
	$(PYTHON) -m sfxbas $< $(SFXFLAGS) -o $@ 

build:
	mkdir -p $@

# Unit and golden tests, plus the VICE tests when x64sc and petcat are installed.
test:
	$(PYTHON) tests/run_tests.py

# Unit tests and representative VICE checks; no CLI parameter sweeps.
test-fast:
	$(PYTHON) tests/run_tests.py --fast

# Also the big inputs.
test-slow:
	SFXBAS_SLOW=1 $(PYTHON) tests/run_tests.py

clean:
	rm -rf build
