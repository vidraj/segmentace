#!/usr/bin/env python3

# Load segmented sentences from STDIN, output various statistics.

import sys
from collections import defaultdict
from segmentshandler import SegmentedLoader

from math import log2

import argparse

parser = argparse.ArgumentParser(description="Measure various statistics on morph-segmented corpora read from STDIN.")
parser.add_argument("-f", "--format", metavar="FORMAT", dest="input_format", help="the format of the input. Available: vbpe, hbpe, spl, hmorph.", choices=["vbpe", "hbpe", "spl", "hmorph"], required=True)
args = parser.parse_args()


def entropy(d, total, word_count):
	"""Calculate the entropy of dictionary of counts d, which has the total sum total (precomputed for speed)."""
	# Entropie je - Sum_morf p(morf) * log_2 p(morf)
	#  p(morf) = c(morf) / c(all)
	e = 0
	for count in d.values():
		p = count/total
		type_e = - p * log2(p)
		e += type_e * count
	return e / word_count


sents  = 0
words  = 0
morphs = 0
chars  = 0

morph_vocab = defaultdict(int)

for sentence in SegmentedLoader(args.input_format, filehandle=sys.stdin):
	sents += 1
	for word in sentence:
		words += 1
		for morph in word:
			morphs += 1
			chars += len(morph)
			morph_vocab[morph] += 1

print("Sents: %d, words: %d, morphs: %d, chars: %d." % (sents, words, morphs, chars))
print("Chars per morph: %f, morphs per word: %f, morphs per sentence: %f." % (chars / morphs, morphs / words, morphs / sents))
print("Morph vocabulary size: %d." % (len(morph_vocab.keys())))
e = entropy(morph_vocab, morphs, words)
print("Avg. per word unigram entropy of morphs: %f, perplexity: %f" % (e, 2**e))
