## Synopsis

This repository contains several tools for segmenting word-segmented text into subword units. Currently, the only fully-working script is `segment-by-derinet.py` (see below).

## Motivation

We want to explore machine translation text preprocessing options and pass NPFL087 Statistical Machine Translation.

## Installation

1. Install the Python MorphoDiTa bindings (ufal.morphodita package) from PyPI. Typically, you'd do this by typing `pip install --user ufal.morphodita` into your terminal.
2. Then, type `make download` to fetch the necessary models and data. It downloads MorphoDiTa 2016.11 models into czech-morfflex-pdt-161115/ and DeriNet 1.4 to derinet-1-4.tsv.gz

## Code Example

```shell-session
$ echo -e 'Ahoj\nsvěte\n!\n\nNáš\nzačátek\nbyl\npomalejší\n,\nlitoval\npo\nprohře\nBerdych\n.' | ./segment-by-derinet.py derinet-1-4.tsv.gz -a czech-morfflex-pdt-161115/czech-morfflex-161115-pos_only.dict
Ahoj
svět@@
e
!

Náš
zač@@
át@@
ek
b@@
yl
pomal@@
ejší
,
litoval
po
proh@@
ře
Berdych
```
