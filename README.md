## Synopsis

This repository contains a tool called `segment-by-derinet.py` (see below) for segmenting word-segmented text into subword units based on [DeriNet](https://ufal.mff.cuni.cz/derinet), a network of Czech lexical derivations.

Two other segmentation systems are present for comparison and a statistics collection module generates reports about their functionality. These systems are:
- [Byte-Pair Encoding](https://github.com/rsennrich/subword-nmt) as per Sennrich et al., 2015 (Neural Machine Translation of Rare Words with Subword Units)
- [Morfessor 2.0](http://morpho.aalto.fi/projects/morpho/morfessor2.html) by Sami Virpioja et al., 2013 (Morfessor 2.0: Python Implementation and Extensions for Morfessor Baseline)

## Motivation

We want to explore machine translation text preprocessing options and pass NPFL087 Statistical Machine Translation.

## Installation

If you want to segment some text with the DeriNet-based segmenter, do the following:
1. Install the Python MorphoDiTa bindings (ufal.morphodita package) from PyPI. Typically, you'd do this by typing `pip3 install --user ufal.morphodita` into your terminal.
2. Optionally, type `make download` to fetch the necessary models and data. It downloads MorphoDiTa 2016.11 models into czech-morfflex-pdt-161115/ and DeriNet 1.4 to derinet-1-4.tsv.gz. You can then play with the tools yourself.
3. Or, proceed the automatized way by simply typing `make`. This downlads the data, runs the DeriNet segmenter on the WMT17 NMT training dataset and produces files `segments-*.txt` with the segmented texts and `stats-*.txt` with the measured statistics.

Optionally, you can also compare the DeriNet-based method with BPE and Morfessor by:
1. Installing Morfessor 2.0 (Morfessor package) from PyPI. Type `pip3 install --user Morfessor`.
2. Running `make -rj8 compare` (subsitute your own preferred thread count for 8).
3. Inspecting the generated `segments-*.txt` and `stats-*.txt` files.

## Usage

```shell-session
$ ./segment-by-derinet.py --help
usage: segment-by-derinet.py [-h] [-a DICTIONARY.tagger]
                             [-m MORFFLEX.tab.csv.xz] [-f FORMAT] [-t FORMAT]
                             DERINET.tsv.gz

Extract possible segmentations from dictionaries of derivations and
inflections and segment corpora from STDIN.

positional arguments:
  DERINET.tsv.gz        a path to the compressed DeriNet dictionary.

optional arguments:
  -h, --help            show this help message and exit
  -a DICTIONARY.tagger, --analyzer DICTIONARY.tagger
                        a path to the MorphoDiTa tagger data. When used, will
                        lemmatize the input data before segmenting, thus
                        supporting segmentation of inflected forms.
  -m MORFFLEX.tab.csv.xz, --morfflex MORFFLEX.tab.csv.xz
                        a path to the compressed MorfFlex dictionary. When
                        used, will enrich the dictionary with forms in
                        addition to lemmas, thus supporting segmentation of
                        inflected forms. Beware, this makes the program very
                        memory intensive.
  -f FORMAT, --from FORMAT
                        the format to read. Available: vbpe, hbpe, spl,
                        hmorph. Default: spl.
  -t FORMAT, --to FORMAT
                        the format to write. Available: vbpe, hbpe, spl,
                        hmorph. Default: vbpe.

By default, only lemmas from DeriNet are loaded. Since segmentation of lemmas
only is too limited for most applications, you can optionally enable support
for segmenting inflected forms by using the --analyzer or --morfflex options.
Loading MorfFlex produces the most detailed segmentation, but it is very
memory intensive. Using the MorphoDiTa analyzer is cheaper, but requires you
to install the 'ufal.morphodita' package prom PyPI and doesn't segment all
forms reliably.
```

## Supported formats

The `segment-by-derinet.py` segmenter, `segmentshandler.py` converter and
`segmentation-statistics.py` stats calculator support several input and output
formats. They are:
- `vbpe` for BPE-like seg@@ ments separated by a newline. Non-final morphs in
  a word are marked by an "@@" suffix, the final morph is left unmarked.
  Sentences are separated by an empty line (two newlines).
- `hbpe` for BPE-like seg@@ ments separated by a space. Sentences are separated
  by a newline.
- `spl` for sentence-per-line without any morph separation. This is used for reading.
- `hmorph` for morphs separated by a space, words separated by a "◽" (white square)
  symbol and sentences separated by a newline.

The defaults are `spl` for reading and `vbpe` for writing.


## Code Example

```shell-session
$ echo -e 'Ahoj světe !\nNáš začátek byl pomalejší , litoval po prohře Berdych .' |
> ./segment-by-derinet.py -a czech-morfflex-pdt-161115/czech-morfflex-pdt-161115.tagger derinet-1-4.tsv.gz
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
