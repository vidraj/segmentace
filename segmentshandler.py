#!/usr/bin/env python3
import sys

# Two classes for handling morph-segmented corpora, plus a main function that enables users to convert between various formats.

# Takes a filename, returns a list of sentences, where each sentence is a list of words, where each word is a list of morphs.
class SegmentedLoader:
	def __init__(self, fileformat, filename=None, filehandle=None):
		# TODO switch on fileformat, select the appropriate next() function
		if fileformat == "vbpe":
			self.next = self.next_sent_vbpe
		elif fileformat == "hbpe":
			self.next = self.next_sent_hbpe
		elif fileformat == "spl":
			self.next = self.next_sent_spl
		elif fileformat == "hmorph":
			self.next = self.next_sent_hmorph
		else:
			raise ValueError("Unexpected file format '%s'." % fileformat)
		
		if filename is not None and filehandle is None:
			self.filename = filename
			self.filehandle = None
			self.from_file = True
		elif filename is None and filehandle is not None:
			self.filehandle = filehandle
			self.from_file = False
		else:
			raise Exception("Exactly one of {filename, filehandle} must be defined.")
	
	def __enter__(self):
		if self.from_file:
			self.filehandle = sys.open(self.filename, "rt")
		return self
	
	def __exit__(self, exc_type, exc_value, traceback):
		if self.from_file:
			self.filehandle.close()
	
	def __iter__(self):
		return self
	
	def __next__(self):
		return self.next()
	
	def next_sent_vbpe(self):
		"""Reads next sentence in the vertical+bpe format and returns it. Each line contains one morph. The last morph in a word is cleartext, the preceeding morphs end in @@. Sentences are delimited by an empty line."""
		morphs = []
		words = []
		
		for next_morph in self.filehandle:
			next_morph = next_morph.rstrip('\n')
			if next_morph == "":
				# Empty line, the previous sentence has ended.
				return words
			
			if next_morph.endswith("@@"):
				# A continuation morph.
				morphs.append(next_morph[0:-2])
			else:
				# Final morph which ends the word.
				morphs.append(next_morph)
				words.append(morphs)
				morphs = []
		
		# The iteration has ended, end-of-file was reached.
		if words:
			return words
		else:
			raise StopIteration()
	
	def next_sent_hbpe(self):
		"""Reads next sentence in the horizontal+bpe format and returns it. Each line contains one sentence, consisting of space-delimited morphs. The last morph in a word is cleartext, the preceeding morphs end in @@."""
		line = self.filehandle.readline()
		if line:
			line = line.rstrip("\n")
			morphs = line.split(" ")
			words = []
			word = []
			for morph in morphs:
				if morph.endswith("@@"):
					# A continuation morph.
					word.append(morph[0:-2])
				else:
					# Final morph which ends the word.
					word.append(morph)
					words.append(word)
					word = []
				
			return words
		else:
			raise StopIteration()
	
	def next_sent_spl(self):
		"""Reads next sentence formatted in the horizontal format (sentence per line, space-delimited words) and returns it."""
		line = self.filehandle.readline()
		if line:
			line = line.rstrip("\n")
			words = line.split(" ")
			return [[w] for w in words]
		else:
			raise StopIteration()
	
	def next_sent_hmorph(self):
		"""Reads next sentence formatted in the horizontal-morphs format (sentence per line, square-delimited words, space-delimited morphs) and returns it."""
		line = self.filehandle.readline()
		if line:
			line = line.rstrip("\n")
			words = line.split(" ◽ ")
			return [w.split(" ") for w in words]
		else:
			raise StopIteration()


class SegmentedStorer:
	def __init__(self, fileformat, filehandle):
		if fileformat == "vbpe":
			self.morph_separator = "@@\n"
			self.word_separator = "\n"
			self.sentence_separator = "\n"
			self.sentence_end = "\n"
		elif fileformat == "hbpe":
			self.morph_separator = "@@ "
			self.word_separator = " "
			self.sentence_separator = ""
			self.sentence_end = "\n"
		elif fileformat == "spl":
			self.morph_separator = ""
			self.word_separator = " "
			self.sentence_separator = ""
			self.sentence_end = "\n"
		elif fileformat == "hmorph":
			self.morph_separator = " "
			self.word_separator = " ◽ "
			self.sentence_separator = ""
			self.sentence_end = "\n"
		else:
			raise ValueError("Unexpected file format '%s'." % fileformat)
		
		self.filehandle = filehandle
		self.first = True # True for the first sentence that is ever printed, False for all other.
		
	
	def print_sentence(self, sentence):
		if self.first:
			formatted_output = ""
		else:
			formatted_output = self.sentence_separator
		
		formatted_output += self.format_sentence(sentence)
		print(formatted_output, file=self.filehandle, end=self.sentence_end)
		
		self.first = False
	
	def format_sentence(self, sentence):
		return self.word_separator.join([self.morph_separator.join(morphs) for morphs in sentence])

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(description="Convert corpora between various formats.")
	parser.add_argument("-f", "--from", metavar="FORMAT", dest="from_format", help="the format to convert from. Available: vbpe, hbpe, spl, hmorph.", choices=["vbpe", "hbpe", "spl", "hmorph"], required=True)
	parser.add_argument("-t", "--to", metavar="FORMAT", dest="to_format", help="the format to convert to. Available: vbpe, hbpe, spl, hmorph. Default: hmorph.", default="hmorph", choices=["vbpe", "hbpe", "spl", "hmorph"])
	args = parser.parse_args()
	
	storer = SegmentedStorer(args.to_format, filehandle=sys.stdout)
	for sentence in SegmentedLoader(args.from_format, filehandle=sys.stdin):
		storer.print_sentence(sentence)
