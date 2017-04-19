#!/usr/bin/env python3

# The first argument contains a file with segmentations ("pod vod ník\npod čár ník\n").
# STDIN contains input sentences.
# Read each sentence and write its morphs delimited by a special space-marking character.

import sys

segments_file_name = sys.argv[1]

words_to_segments = {}

with open(segments_file_name, "rt") as segments_file:
	for line in segments_file:
		morphs = line.rstrip().split(" ")
		word = "".join(morphs)
		
		assert word not in words_to_segments, "Word %s already in dict!" % word
		
		words_to_segments[word] = morphs

def segment_word(word, database=words_to_segments):
	if word in words_to_segments:
		return words_to_segments[word]
	else:
		return word


for sentence in sys.stdin:
	words = sentence.rstrip().split(" ")
	segmented_words = [" ".join(segment_word(word)) for word in words]
	segmented_sentence = " ◽ ".join(segmented_words)
	print(segmented_sentence)
