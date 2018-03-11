#!/usr/bin/env python3

# Load segmented words from STDIN and gold-standard segmentation from file, measure accuracy, precision, recall and F1-measure.

import sys
from segmentshandler import SegmentedLoader
from itertools import zip_longest
from collections import Counter

import argparse

parser = argparse.ArgumentParser(description="Measure precision of STDIN against gold-standard segmentation.")
parser.add_argument("-f", "--format", metavar="FORMAT", dest="input_format", help="the format of the input. Available: vbpe, hbpe, spl, hmorph.", choices=["vbpe", "hbpe", "spl", "hmorph"], required=True)
parser.add_argument("-g", "--gold-format", metavar="FORMAT", dest="gold_format", help="the format of the gold-standard data. Available: vbpe, hbpe, spl, hmorph. Default: hmorph", choices=["vbpe", "hbpe", "spl", "hmorph"], default="hmorph")
parser.add_argument("gold", metavar="FILE", help="the gold-standard annotation file name")
args = parser.parse_args()




def enumerate_bounds(word):
	"""Read word, which is a list of strings, and return a set of positions of breaks between the strings"""
	lens = [len(morph) for morph in word]
	
	bounds = {0}
	last_bound = 0
	for l in lens:
		last_bound += l
		bounds.add(last_bound)
	
	return bounds



morphs = 0
morphs_correct = 0
morphs_incorrect = 0
bounds_true_positive = 0
bounds_false_positive = 0
bounds_false_negative = 0
words_correct = 0
words_incorrect = 0

with open(args.gold, "rt") as gold_file:
	for predicted_sentence, gold_sentence in zip_longest(SegmentedLoader(args.input_format, filehandle=sys.stdin), SegmentedLoader(args.gold_format, filehandle=gold_file)):
		assert len(predicted_sentence) == len(gold_sentence), "These sentences are misaligned, because their lengths don't match: '%r' and '%r'" % (predicted_sentence, gold_sentence)
		for predicted_word, gold_word in zip(predicted_sentence, gold_sentence):
			# Assert that the words (when concatenated) are the same.
			pw = "".join(predicted_word)
			gw = "".join(gold_word)
			if pw != gw:
				print("The word '%s' doesn't match the gold-standard word '%s'!" % (pw, gw), file=sys.stderr)
				# TODO mark all morphs in these words as bad.
				#morphs_incorrect += max(predicted_word, gold_word)
				#bounds_false_negative += # TODO
				sys.exit(1)
			
			if predicted_word == gold_word:
				# The segmentations are fully identical.
				words_correct += 1
			else:
				# The segmentations differ.
				words_incorrect += 1
			
			# For each morph type, count how many of them were predicted properly and improperly.
			# N.b. this doesn't take into account their ordering, but who cares. It makes the task a little easier, but probably not by much.
			pmset = Counter(predicted_word)
			gmset = Counter(gold_word)
			for pm in pmset:
				correct = min(pmset[pm], gmset[pm])
				incorrect = pmset[pm] - correct
				
				morphs_correct += correct
				morphs_incorrect += incorrect
			
			# If the predicted word is shorter than the gold one, the „incorrectly missing“ morphs were not counted in the loop above.
			# Count them here.
			if len(predicted_word) > len(gold_word):
				morphs_incorrect += len(predicted_word) - len(gold_word)
			
			# Remember how many morphs are there in the gold-standard data
			morphs += len(gold_word)
			
			
			
			# Compute precision and recall over morph boundaries.
			p_bounds = enumerate_bounds(predicted_word)
			g_bounds = enumerate_bounds(gold_word)
			
			tp = p_bounds.intersection(g_bounds)
			fp = p_bounds.difference(g_bounds)
			fn = g_bounds.difference(p_bounds)
			
			bounds_true_positive += len(tp)
			bounds_false_positive += len(fp)
			bounds_false_negative += len(fn)


morph_precision = morphs_correct / (morphs_correct + morphs_incorrect)
morph_recall = morphs_correct / morphs
morph_f1 = 2 * morph_precision * morph_recall / (morph_precision + morph_recall)
print("Morphs correct: %d, incorrect: %d, gold: %d." % (morphs_correct, morphs_incorrect, morphs))
print("Morph precision: %.02f %%" % (100 * morph_precision))
print("Morph recall: %.02f %%" % (100 * morph_recall))
print("Morph F1-measure: %02f %%" % (100 * morph_f1))

bounds_precision = bounds_true_positive / (bounds_true_positive + bounds_false_positive)
bounds_recall = bounds_true_positive / (bounds_true_positive + bounds_false_negative)
bounds_f1 = 2 * bounds_precision * bounds_recall / (bounds_precision + bounds_recall)
print("Bounds correct: %d, incorrect: %d, gold: %d." % (bounds_true_positive, bounds_false_positive, bounds_true_positive + bounds_false_negative))
print("Bounds precision: %.02f %%" % (100 * bounds_precision))
print("Bounds recall: %.02f %%" % (100 * bounds_recall))
print("Bounds F1-measure: %02f %%" % (100 * bounds_f1))

print("Word accuracy: %.02f %%" % (100 * words_correct / (words_correct + words_incorrect)))
