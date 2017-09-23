## Synopsis

This repository contains several tools for segmenting word-segmented text into subword units. Currently, the only fully-working script is `segment-by-derinet.py` (see below).

## Motivation

We want to explore machine translation text preprocessing options and pass NPFL087 Statistical Machine Translation.

## Installation

1. Install the Python MorphoDiTa bindings (ufal.morphodita package) from PyPI. Typically, you'd do this by typing `pip install --user ufal.morphodita` into your terminal.
2. Then, type `make download` to fetch the necessary models and data. It downloads MorphoDiTa 2016.11 models into czech-morfflex-pdt-161115/ and DeriNet 1.4 to derinet-1-4.tsv.gz

## Usage

```shell-session
$ ./segment-by-derinet.py --help
usage: segment-by-derinet.py [-h] [-a DICTIONARY.dict]
                             [-m MORFFLEX.tab.csv.xz]
                             DERINET.tsv.gz

Extract possible segmentations from dictionaries of derivations and
inflections.

positional arguments:
  DERINET.tsv.gz        a path to the compressed DeriNet dictionary.

optional arguments:
  -h, --help            show this help message and exit
  -a DICTIONARY.dict, --analyzer DICTIONARY.dict
                        a path to the MorphoDiTa morphological analyzer data.
                        When used, will lemmatize the input data before
                        segmenting, thus supporting segmentation of inflected
                        forms.
  -m MORFFLEX.tab.csv.xz, --morfflex MORFFLEX.tab.csv.xz
                        a path to the compressed MorfFlex dictionary. When
                        used, will enrich the dictionary with forms in
                        addition to lemmas, thus supporting segmentation of
                        inflected forms. Beware, this makes the program very
                        memory intensive.

By default, only lemmas from DeriNet are loaded. Since segmentation of lemmas
only is too limited for most applications, you can optionally enable support
for segmenting inflected forms by using the --analyzer or --morfflex options.
Loading MorfFlex produces the most detailed segmentation, but it is very
memory intensive. Using the MorphoDiTa analyzer is cheaper, but requires you
to install the 'ufal.morphodita' package prom PyPI and doesn't segment all
forms reliably.
```


## Code Example

```shell-session
$ echo -e 'Ahoj\nsvěte\n!\n\nNáš\nzačátek\nbyl\npomalejší\n,\nlitoval\npo\nprohře\nBerdych\n.' | \
> ./segment-by-derinet.py derinet-1-4.tsv.gz -a czech-morfflex-pdt-161115/czech-morfflex-161115-pos_only.dict
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
