# sfxbas

Packs a C64 program (.PRG) into a smaller, self-extracting **pure BASIC** program. The decruncher contains no machine
code: it is only BASIC that you can `LIST` plus BASIC variable data in memory. 
Use it to make a demo, game or intro load higher (RAM under $D000-$FFFF) or faster, 
or to fit it into a size-limited compo.  Once decrunched, the memory is byte-exact to the original.

You can also use it when the rules allow BASIC only (hello <a href=https://zooparty.org>ZOO Party</a>).  The decruncher uses only POKE, SYS41964 and standard BASIC instructions, so it matches the ZOOParty BASIC Effect competition rules.

```
python3 -m sfxbas demo.prg -o demo_sfx.prg      
```

## Why and How

A standard decruncher written in BASIC would maybe make `POKE`s one byte at a time and take minutes to run. Instead, we have ready-prepared string array and the decompression done by simply concatenating of existing strings in order chosen by the compressor.

- The packed stream is stored as **ready-made BASIC arrays of strings**. The decruncher points `ARYTAB` at each chunk, so BASIC finds the arrays without parsing or copying them.
- Each part is inserted by string concatenation: `t$=q$(k)+""`. BASIC's own string copy moves the bytes to the top of free memory (FRETOP).   Each source string is from a literal in the stream or a match from previous unpack. 
- In compression, an optimal parser selects the part, and you can adjust the part strategy with for smaller file or a faster unpack (options `-size`, `-speed`; `-speed=0.4` for 1541 like or `-speed=4` for disk turbo).
- BASIC programs cannot read whole memory of C64 (ROM covers $A000-$BFFF, $E000-$FFFF), so the packer trys to use staging area for these regions
- A simulator runs every result on a model of C64 before it is written. Then we know the file unpacks correctly, and approximately how long it would take.

Every build compares the original parse with a cycle-aware candidate that favors
fewer BASIC copy statements when parsing costs tie. Size builds keep bytes first
and use estimated unpack time to break ties. `-speed` compares its load/unpack
objective; `-time=S` prefers the smallest build estimated to meet the deadline
(and the fastest candidate if none does). The candidates are
simulator-verified before selection. This adds build time; the BASIC decoder and
packed format are unchanged.

Typical result: a 60 kB program becomes 20–25 kB and unpacks in 3–6 seconds.

## Searching for the smallest file

```sh
python3 -m sfxbas demo.prg -size -o demo_sfx.prg
```

`-size` compares a lambda grid, up to four rounds of literal sharing, an alternate
literal-pool seed, and several fitting chunk layouts. It keeps the original
candidates and selects the smallest verified PRG, preferring faster unpacking on
size ties. This takes more build time than choosing a single lambda.
`-size=full` adds more lambda values and paired parses. For shortest load plus
unpack time instead, use `-speed=0.4` (stock 1541) or your loader's kB/s rate.

## Comparison with native cruncher

All examples, packed with `sfxbas -speed=0.4` (best load + unpack at stock 1541 speed)
and with Exomizer 3.1.2 (`exomizer sfx basic`). Sizes are file sizes in bytes, load address included.
Exomizer is a machine code decruncher with bit-level coding: smaller and faster. sfxbas stays pure BASIC.

Size and load time at about 0.4 kB/s (a stock 1541):

| name | original | load | exomizer | load | sfxbas | load | factor |
|---|--:|--:|--:|--:|--:|--:|--:|
| amigabasic | 39915 | 97.4 s | 1702 | 4.2 s | 3469 | 8.5 s | 2.04× |
| digi | 37851 | 92.4 s | 11867 | 29.0 s | 18328 | 44.7 s | 1.54× |
| parallaxative | 58515 | † 142.9 s | 10873 | 26.5 s | 17995 | 43.9 s | 1.66× |
| pubquest | 33538 | 81.9 s | 16856 | 41.2 s | 26124 | 63.8 s | 1.55× |
| superbox1 | 63489 | † 155.0 s | † 10886 | 26.6 s | 17793 | 43.4 s | 1.63× |
| superscroller | 47105 | 115.0 s | 4250 | 10.4 s | 8433 | 20.6 s | 1.98× |
| yleradio | 38914 | 95.0 s | 8852 | 21.6 s | 12523 | 30.6 s | 1.41× |
| **total** | 319327 | 779.6 s | 65286 | 159.4 s | 104665 | 255.5 s | 1.60× |

Size and decrunch time, measured from `RUN` until the unpacked program starts:

| name | original | exomizer | decrunch | sfxbas | decrunch | factor |
|---|--:|--:|--:|--:|--:|--:|
| amigabasic | 39915 | 1702 | 1.2 s | 3469 | 2.4 s | 1.93× |
| digi | 37851 | 11867 | 3.3 s | 18328 | 17.1 s | 5.22× |
| parallaxative | 58515 | 10873 | 3.7 s | 17995 | 5.2 s | 1.42× |
| pubquest | 33538 | 16856 | 3.9 s | 26124 | 11.4 s | 2.92× |
| superbox1 | 63489 | † 10886 | † 3.5 s | 17793 | 12.8 s | 3.68× |
| superscroller | 47105 | 4250 | 1.9 s | 8433 | 7.3 s | 3.78× |
| yleradio | 38914 | 8852 | 2.5 s | 12523 | 9.8 s | 3.87× |
| **total** | 319327 | 65286 | 20.0 s | 104665 | 66.0 s | 3.30× |

factor: sfxbas divided by exomizer (size in the first table, decrunch time in the second).

† Original load times for high-memory programs are theoretical. For superbox1, the Exomizer
input omits the final eight bytes to avoid its $10000 end-address limitation; sfxbas uses the full input.

## Further reading

- https://www.masswerk.at/nowgobang/2020/commodore-basic-variables
