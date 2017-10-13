## Synopsis

This repository contains several tools for segmenting word-segmented text into subword units and a statistics collection module that generates reports about their functionality. Currently, we test three systems:
- [Byte-Pair Encoding](https://github.com/rsennrich/subword-nmt) as per Sennrich et al., 2015 (Neural Machine Translation of Rare Words with Subword Units)
- [Morfessor 2.0](http://morpho.aalto.fi/projects/morpho/morfessor2.html) by Sami Virpioja et al., 2013 (Morfessor 2.0: Python Implementation and Extensions for Morfessor Baseline)
- [DeriNet](https://ufal.mff.cuni.cz/derinet) and [MorphoDita](https://ufal.mff.cuni.cz/morphodita) guided segmentation, developed by us and implemented in `segment-by-derinet.py` (see below).

## Motivation

We want to explore machine translation text preprocessing options and pass NPFL087 Statistical Machine Translation.

## Installation

1. Install the Python MorphoDiTa bindings (ufal.morphodita package) from PyPI. Typically, you'd do this by typing `pip3 install --user ufal.morphodita` into your terminal.
2. Install Morfessor 2.0 (Morfessor package) from PyPI. Type `pip3 install --user Morfessor`.
3. Optionally, type `make download` to fetch the necessary models and data. It downloads MorphoDiTa 2016.11 models into czech-morfflex-pdt-161115/ and DeriNet 1.4 to derinet-1-4.tsv.gz. You can then play with the tools yourself.
4. Or, proceed the automatized way by simply typing `make -rj8` (subsitute your own preferred thread count for 8). This downlads the data, runs the models on the WMT17 NMT training dataset and produces files `segments-*.txt` with the segmented texts and `stats-*.txt` with the measured statistics.

## Usage

```shell-session
$ ./segment-by-derinet.py --help
usage: segment-by-derinet.py [-h] [-a DICTIONARY.tagger]
                             [-m MORFFLEX.tab.csv.xz]
                             DERINET.tsv.gz

Extract possible segmentations from dictionaries of derivations and
inflections.

positional arguments:
  DERINET.tsv.gz        a path to the compressed DeriNet dictionary.

optional arguments:
  -h, --help            show this help message and exit
  -a DICTIONARY.tagger, --analyzer DICTIONARY.tagger
                        a path to the MorphoDiTa tagger data.
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
> ./segment-by-derinet.py derinet-1-4.tsv.gz -a czech-morfflex-pdt-161115/czech-morfflex-pdt-161115-pos_only.tagger
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
lit@@
ov@@
a@@
l
po
proh@@
ře
Berdych
.
```
