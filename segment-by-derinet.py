#!/usr/bin/env python3

import sys
import re

import unittest

import gzip
import lzma

import itertools

import argparse

from collections import defaultdict, Counter

from time import strftime

from segmentshandler import SegmentedLoader, SegmentedStorer


parser = argparse.ArgumentParser(description="Extract possible segmentations from dictionaries of derivations and inflections and segment corpora from STDIN.", epilog="By default, only lemmas from DeriNet are loaded. Since segmentation of lemmas only is too limited for most applications, you can optionally enable support for segmenting inflected forms by using the --analyzer or --morfflex options. Loading MorfFlex produces the most detailed segmentation, but it is very memory intensive. Using the MorphoDiTa analyzer is cheaper, but requires you to install the 'ufal.morphodita' package prom PyPI and doesn't segment all forms reliably.\n\nThe input should be in a “word per line, sentences separated by an empty line” format.")
parser.add_argument("derinet", metavar="DERINET.tsv.gz", help="a path to the compressed DeriNet dictionary.")
#parser.add_argument("morfflex", metavar="MORFFLEX.tab.csv.xz", help="a path to the compressed MorfFlex dictionary.")
parser.add_argument("-a", "--analyzer", metavar="DICTIONARY.tagger", help="a path to the MorphoDiTa tagger data. When used, will lemmatize the input data before segmenting, thus supporting segmentation of inflected forms.")
parser.add_argument("-m", "--morfflex", metavar="MORFFLEX.tab.csv.xz", help="a path to the compressed MorfFlex dictionary. When used, will enrich the dictionary with forms in addition to lemmas, thus supporting segmentation of inflected forms. Beware, this makes the program very memory intensive.")
#parser.add_argument("morpho", metavar="MORPHO", help="a path to the MorphoDiTa morphological resource.")
args = parser.parse_args()

def perr(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

def longest_common_substring_position(string_a, string_b):
	"""Returns (start of lcs in string_a, start of lcs in string_b, length of lcs)"""
	# Initialize the longest-suffixes table.
	length_table = [[0] * (1 + len(string_b)) for i in range(1 + len(string_a))]
	
	# These will hold information about where the longest common substring ends in the strings and about its length.
	longest_len = 0
	longest_substr_end_x = 0
	longest_substr_end_y = 0
	
	# Walk through the strings, matching all possible suffixes.
	for x in range(len(string_a)):
		for y in range(len(string_b)):
			if string_a[x] == string_b[y]:
				# We can extend the current longest suffix here by one.
				length_table[x + 1][y + 1] = length_table[x][y] + 1
				
				if length_table[x + 1][y + 1] > longest_len:
					# If we've found a new longest substring, store information about it for later retrieval.
					longest_len = length_table[x + 1][y + 1]
					longest_substr_end_x = x + 1
					longest_substr_end_y = y + 1
			else:
				# We cannot extend the current longest suffix, so start a new match instead of extending the old one.
				length_table[x + 1][y + 1] = 0
	
	return (longest_substr_end_x - longest_len, longest_substr_end_y - longest_len, longest_len)

def longest_common_substring(string_a, string_b):
	pos_x, pos_y, length = longest_common_substring_position(string_a, string_b)
	return string_a[pos_x:pos_x+length]

def divide_string(s, p1, p2):
	"""Divide s into 3 parts: [0:p1], [p1:p2], [p2:]"""
	return (s[0:p1], s[p1:p2], s[p2:])

def pairs(l):
	"""For [a,b,c, … y,z] in list l, return (a,b), (b,c) … (y,z)"""
	for lower in range(len(l) - 1):
		yield (l[lower], l[lower + 1])

def techlemma_to_lemma(techlemma):
	"""Cut off the technical suffixes from the string techlemma and return the raw lemma"""
	shortlemma = re.sub("[_`].+", "", techlemma)
	lemma = re.sub("-\d+$", "", shortlemma)
	return lemma

class Lexeme:
	# Types of morpheme changes encountered in the data and their frequencies.
	morph_change_types = {"padd": 0, # Prefix addition
	                      "prem": 0, # Prefix removal
	                      "pcha": 0, # Prefix substitution
	                      "sadd": 0, # Suffix addition
	                      "srem": 0, # Suffix removal
	                      "scha": 0, # Suffix substitution
	                      "conv": 0, # Conversion (no change)
	                      "circ": 0} # Circumfixation (or weird change on both sides)
	
	allowed_morph_change_types = {"padd", "sadd", "scha", "conv"}
	
	def __init__(self, lemma=None, morphs=None, id=None, parent_id=None, parent_lemma=None):
		self.id = int(id) if id else None
		
		
		morph_bounds = {0, len(lemma)}
		if lemma is not None and morphs is None:
			self.lemma = lemma
		elif morphs is not None:
			joined_lemma = "".join(morphs)
			if lemma is None or lemma == joined_lemma:
				self.lemma = joined_lemma
				
				given_bounds = []
				last_given_bound = 0
				for morph in morphs:
					last_given_bound += len(morph)
					given_bounds.append(last_given_bound)
				
				morph_bounds = morph_bounds.union(given_bounds)
			else:
				raise ValueError("The provided lemma '%s' doesn't match the morphs '%s'" % (lemma, joined_lemma))
		else:
			raise ValueError("You must provide at least one of {lemma, morphs}")
		
		self.parent_id = int(parent_id) if parent_id else None
		self.parent_lemma = parent_lemma # TODO if both parent_id and parent_lemma are filled in, cross-check their information so that it fits.
		self.parent = None
		self.children = []
		self.morph_bounds = morph_bounds # A set of morph boundaries in lemma. Each boundary is an int: an offset where a new morph starts in lemma.
		self.morph_change_type = None # One of the keys on morph_change_types, documenting the type of morph change present when going from parent to self.
		self.stem_map = None # A hash with keys "self", "parent", each with a tuple with (start, end) of the common stem (longest common substring) in lemma.
	
	def to_string(self):
		if self.id:
			return "%d (%s)" % (self.id, self.lemma)
		else:
			return "from MorfFlex (%s)" % self.lemma
	
	def is_cycle(self, node):
		"""Detect whether the parents of node include self, return True if they do, false otherwise.
		Used in detecting cycles when adding node as the parent of self."""
		while node is not None:
			if (node.id is not None and node.id == self.id) \
			  or (node.id is None and node.lemma == self.lemma):
				# A cycle has been detected
				return True
			else:
				node = node.parent
		return False
	
	def set_parent(self, parent):
		"""Set parent as the parent of self. Do some error checks in the process."""
		if not parent:
			raise ValueError("Setting empty parent not supported")
		if self.parent:
			raise Exception("Parent already set to %s" % self.parent)
		if self.parent_id is not None and parent.id != self.parent_id:
			raise ValueError("Parent ID of %d doesn't match expected ID of %d" % (parent.id, self.parent_id))
		if self.parent_lemma is not None and parent.lemma != self.parent_lemma:
			raise ValueError("Parent lemma '%s' doesn't match expected lemma '%s'" % (parent.lemma, self.parent_lemma))
		if self.is_cycle(parent):
			raise Exception("Found a cycle involving lexeme %s" % self.to_string())
			
		
		self.parent = parent
		parent.children.append(self)
	
	def detect_stems(self, parent=None):
		"""Look at the lemma of self and parent and try to detect any morph changes between the two.
		If there are any, set the appropriate morph bounds.
		If parent is None, use self.parent instead."""
		
		if parent is None:
			parent = self.parent
		
		if parent is None:
			#perr("Trying to detect morph bounds of %d (%s), which has no parent!" % (self.id, self.lemma))
			return
		
		plemma = parent.lemma
		slemma = self.lemma
		stem_start_parent, stem_start_self, stem_length = longest_common_substring_position(plemma.lower(), slemma.lower())
		
		bounds_self = (stem_start_self, stem_start_self + stem_length)
		bounds_parent = (stem_start_parent, stem_start_parent + stem_length)
		self.stem_map = {"self": bounds_self,
		                 "parent": bounds_parent}
		
		
		# Fill in the morph change type.
		stem_end_off_parent = len(plemma) - (stem_start_parent + stem_length)
		stem_end_off_self = len(slemma) - (stem_start_self + stem_length)
		morph_change_type = None
		if stem_start_parent == 0:
			# PADD / S / CONV
			if stem_start_self == 0:
				# S / CONV
				if stem_end_off_parent == 0:
					# SADD / CONV
					if stem_end_off_self == 0:
						morph_change_type = "conv"
					else:
						morph_change_type = "sadd"
				else:
					# SREM / SCHA
					if stem_end_off_self == 0:
						morph_change_type = "srem"
					else:
						morph_change_type = "scha"
			else:
				if stem_end_off_parent != 0 or stem_end_off_self != 0:
					#perr("Circumfixation in %s → %s." % (plemma, slemma))
					morph_change_type = "circ"
				else:
					morph_change_type = "padd"
		else:
			# PREM / PCHA
			if stem_end_off_parent != 0 or stem_end_off_self != 0:
				#perr("Circumfixation in %s → %s." % (plemma, slemma))
				morph_change_type = "circ"
			else:
				if stem_start_self == 0:
					morph_change_type = "prem"
				else:
					morph_change_type = "pcha"
		
		self.morph_change_type = morph_change_type;
		Lexeme.morph_change_types[morph_change_type] += 1
		
		# What can happen:
		#  |xxx|   →  |xxx|  (conversion)
		#  |xxx|   →  |xxx|x (suffixation)
		#  |xxx|x  →  |xxx|y (suffix change)
		#  |xxx|   → x|xxx|  (prefixation)
		#  *|xxx|x →  |xxx|  (suffix removal)
		#  *anything else?*
		
		pprefix, pstem, psuffix = divide_string(plemma, stem_start_parent, stem_start_parent + stem_length)
		psegments = [x for x in (pprefix, pstem, psuffix) if x != ""]
		sprefix, sstem, ssuffix = divide_string(slemma, stem_start_self, stem_start_self + stem_length)
		ssegments = [x for x in (sprefix, sstem, ssuffix) if x != ""]
		
		# Write the newly found bounds, unless they are weird.
		if morph_change_type in Lexeme.allowed_morph_change_types:
			#perr("Divided by %s: '%s' → '%s'" % (morph_change_type, "/".join(psegments), "/".join(ssegments)))
			self.morph_bounds = self.morph_bounds.union(bounds_self)
			parent.morph_bounds = parent.morph_bounds.union(bounds_parent)
		else:
			#perr("Divided by %s: '%s' → '%s' (spurious, not saving)" % (morph_change_type, "/".join(psegments), "/".join(ssegments)))
			pass
	
	def copy_morph_bounds(self, parent=None):
		"""Propagate the morph bounds from parent to self. If parent is None, use self.parent instead."""
		if parent is None:
			parent = self.parent
		
		if parent is not None:
			stem_bounds_parent = self.stem_map['parent']
			stem_bounds_self = self.stem_map['self']
			
			offset = stem_bounds_self[0] - stem_bounds_parent[0]
			mapped_self_splits = [split + offset for split in parent.morph_bounds if split > stem_bounds_parent[0] and split < stem_bounds_parent[1]]
			
			self.morph_bounds = self.morph_bounds.union(mapped_self_splits)
		
	
	def propagate_morph_bounds(self):
		"""Find whether splits of this lemma appear in its children and/or parent as well, and propagate them if neccessary.
		Call after detecting the stem bounds first.
		Best called on the root of a tree."""
		
		# First, go down the tree, towards its leaves.
		# After that's finished, go from the leaves up.
		
		# FIXME if one child gives me a new split, do I propagate it to the other children?
		self.copy_morph_bounds()
		
		
		#for child in self.children:
			#child.propagate_morph_bounds()
		
		for child in self.children + self.children:
			child.propagate_morph_bounds()
			# I have the child's stem_map, which documents how the morphs in child map to morphs in self.
			# All splits in child's stem map to splits in self's stem.
			# Other splits don't map.
			stem_bounds_self = child.stem_map['parent']
			stem_bounds_child = child.stem_map['self']
			
			offset = stem_bounds_self[0] - stem_bounds_child[0]
			mapped_child_splits = [split + offset for split in child.morph_bounds if split > stem_bounds_child[0] and split < stem_bounds_child[1]]
			
			self.morph_bounds = self.morph_bounds.union(mapped_child_splits)
			
		
		# TODO do it again.
	
	def morphs(self):
		"""Return a list of morphs, in order. Joined together, they form lemma."""
		return [self.lemma[lower:upper] for lower, upper in pairs(sorted(self.morph_bounds))]

class MorfFlexParser:
	def __init__(self, morfflex_file_name, derinet_db):
		self.filename = morfflex_file_name
		self.filehandle = None
		self.db = derinet_db
		self.seen = defaultdict(bool)
	
	def __enter__(self):
		self.filehandle = lzma.open(self.filename, "rt")
		return self
	
	def __exit__(self, exc_type, exc_value, traceback):
		self.filehandle.close()
	
	def __iter__(self):
		return self
	
	def __next__(self):
		return self.next()
	
	def next(self):
		returning = False
		while not returning:
			line = self.filehandle.readline()
			if line:
				line = line.rstrip('\n')
				techlemma, tag, form = line.split('\t')
				lemma = techlemma_to_lemma(techlemma)
				# Lemmas are already in DeriNet, so there's no need to fill them in again from MorfFlex.
				# And we don't need duplicate lexemes that don't differ in their derivations,
				#  which in case of MorfFlex lexemes means in their lemma, so filter them out.
				if form == lemma:
					continue
				else:
					return Lexeme(form, parent_lemma=lemma)
			else:
				raise StopIteration()

class DeriNetParser:
	def __init__(self, derinet_file_name):
		self.filename = derinet_file_name
		self.filehandle = None
	
	def __enter__(self):
		self.filehandle = gzip.open(self.filename, "rt", newline='\n')
		return self
	
	def __exit__(self, exc_type, exc_value, traceback):
		self.filehandle.close()
	
	def __iter__(self):
		return self
	
	def __next__(self):
		return self.next()
	
	def next(self):
		line = self.filehandle.readline()
		if line:
			line = line.rstrip('\n')
			id, lemma, techlemma, pos, parent = line.split('\t')
			return Lexeme(lemma, id=id, parent_id=parent)
		else:
			raise StopIteration()

class DeriNetDatabase:
	def __init__(self, parser):
		id_to_lexeme = {}
		lemma_to_lexemes = defaultdict(list)
		
		for lexeme in parser:
			if lexeme is None:
				raise Exception("Null lexeme encountered!")
			
			if id_to_lexeme.get(lexeme.id) is not None:
				raise Exception("Lexeme %s defined twice!" % lexeme)
			else:
				id_to_lexeme[lexeme.id] = lexeme
				lemma_to_lexemes[lexeme.lemma].append(lexeme)
		
		for lexeme in id_to_lexeme.values():
			if lexeme.parent_id is not None:
				lexeme.set_parent(id_to_lexeme[lexeme.parent_id])
			if lexeme.parent_lemma is not None:
				parents = lemma_to_lexemes[lexeme.parent_lemma]
				if parents:
					lexeme.set_parent(parents[0])
				else:
					raise Exception("Parent with lemma '%s' not found in the database." % lemma)
		
		self.id_to_lexeme = id_to_lexeme
		self.lemma_to_lexemes = lemma_to_lexemes
	
	def get_by_lemma(self, lemma):
		"""Return a list of lexemes that share lemma."""
		return self.lemma_to_lexemes[lemma]
	
	def iter_trees(self):
		"""Iterate over root nodes."""
		for lexeme in self.id_to_lexeme.values():
			if not lexeme.parent:
				yield lexeme
	
	#def __iter__(self):
		#return self
	
	#def __next__(self):
		#return self.next()
	
	#def next(self):
		#line = self.filehandle.readline()
		#if line:
			#line = line.rstrip('\n')
			#id, lemma, techlemma, pos, parent = line.split('\t')
			#return Lexeme(lemma, id=id, parent_id=parent)
		#else:
			#raise StopIteration()
	def iter(self):
		for lexeme in self.id_to_lexeme.values():
			yield lexeme


class MorfFlexDatabase:
	def __init__(self, parser, derinet_db):
		form_to_lexemes = derinet_db.lemma_to_lexemes
		lexemes = list(derinet_db.id_to_lexeme.values())
		for lexeme in parser:
			# If the lexeme is in the database already, we have to ensure it has a different parent (or no parent at all).
			# Otherwise don't add duplicates.
			duplicate = False
			for already_present_node in derinet_db.get_by_lemma(lexeme.lemma):
				if already_present_node.parent and already_present_node.parent.lemma == lexeme.parent_lemma:
					duplicate = True
					break
			
			if not duplicate:
				parents = derinet_db.get_by_lemma(lexeme.parent_lemma)
				if parents:
					lexeme.set_parent(parents[0])
					lexemes.append(lexeme)
					form_to_lexemes[lexeme.lemma].append(lexeme)
				else:
					#raise Exception("Parent of '%s' with lemma '%s' not found in the database." % (lexeme.lemma, lexeme.parent_lemma))
					parent = Lexeme(lexeme.parent_lemma)
					lexemes.append(parent)
					form_to_lexemes[parent.lemma].append(parent)
					lexeme.set_parent(parent)
					lexemes.append(lexeme)
					form_to_lexemes[lexeme.lemma].append(lexeme)
		
		self.lexemes = lexemes
		self.form_to_lexemes = form_to_lexemes
	
	def get_by_lemma(self, form):
		return self.form_to_lexemes[form]
	
	def iter(self):
		for lexeme in self.lexemes:
			yield lexeme
	
	def iter_trees(self):
		"""Iterate over root nodes."""
		for lexeme in self.lexemes:
			if not lexeme.parent:
				yield lexeme
		

def process_stdin(derinet_file_name, morfflex_file_name, morpho_file_name):
	perr("Loading derivations.")
	with DeriNetParser(derinet_file_name) as derinet:
		#for lexeme in itertools.islice(derinet, 10):
			#print(lexeme.lemma)
		
		derinet_db = DeriNetDatabase(derinet)
	perr("Derivations loaded at %s" % strftime("%c"))
	
	if morfflex_file_name is not None:
		perr("Loading inflections.")
		with MorfFlexParser(morfflex_file_name, derinet) as morfflex:
			db = MorfFlexDatabase(morfflex, derinet_db)
		perr("Inflections loaded at %s" % strftime("%c"))
	else:
		db = derinet_db
	
	perr("Detecting stem bounds.")
	
	for node in db.iter():
		node.detect_stems()
	
	perr("Stem bounds detected at %s" % strftime("%c"))
	perr("Propagating morph bounds.")
	
	for root in db.iter_trees():
		root.propagate_morph_bounds()
	
	perr("Morph bounds propagated at %s" % strftime("%c"))
	
	if morpho_file_name is not None:
		perr("Loading morphology")
		tagger = None
		try:
			from ufal.morphodita import Tagger, TaggedLemmas
			tagger = Tagger.load(morpho_file_name)
			lemmas = TaggedLemmas()
		except ImportError:
			perr("You need to install the MorphoDiTa Python bindings!")
		finally:
			if not tagger:
				perr("Cannot load morphological dictionary from file '%s'." % morpho_file_name)
				sys.exit(1)
		perr("Morphology loaded at %s" % strftime("%c"))
	else:
		perr("No morphological dictionary specified. Inflectional morphology will not be available.")
		tagger = None
		lemmas = []
		
	
	perr("Splitting STDIN.")
	
	#morphCounter = Counter()
	#for node in db.iter():
		#morphs = node.morphs()
		#for morph in morphs:
			#morphCounter[morph] += 1
		#print("/".join(morphs))
	
	
	#for morph, count in morphCounter.most_common():
		#print("%s\t%d" % (morph, count))
	
	# TODO process STDIN and print divided morphs.
	storer = SegmentedStorer("vbpe", filehandle=sys.stdout)
	for input_sentence in SegmentedLoader("spl", filehandle=sys.stdin):
		words = ["".join(morphs) for morphs in input_sentence]
		output_sentence = []
		
		if tagger is not None:
			tagger.tag(words, lemmas)
		
		for word, analysis in itertools.zip_longest(words, lemmas):
			node = None
			parent_node = None
			
			# First, try to find the word in the database.
			nodes = db.get_by_lemma(word)
			if nodes:
				node = nodes[0]
			elif analysis is not None:
				# If the word is not in the database itself, try to find its lemma there.
				lemma = techlemma_to_lemma(analysis.lemma)
				parent_nodes = db.get_by_lemma(lemma)
				if parent_nodes:
					# The word is not in the database, but its lemma is.
					# Create a new node for the word and propagate the bounds to it.
					parent_node = parent_nodes[0]
					node = Lexeme(word, parent_lemma=lemma)
					node.detect_stems(parent_node)
					node.copy_morph_bounds(parent_node)
			
			if node:
				output_sentence.append(node.morphs())
			else:
				# If all else fails, consider the word to be a single morph.
				output_sentence.append([word])
		
		#perr(output_sentence)
		
		storer.print_sentence(output_sentence)
	
	#perr(Lexeme.morph_change_types)





class TestDerinetSegmentation(unittest.TestCase):
	def test_lcs(self):
		self.assertEqual(longest_common_substring("ahoj", "ahoj"), "ahoj")
		self.assertEqual(longest_common_substring("ahoj", "bhoj"), "hoj")
		self.assertEqual(longest_common_substring("ahoj", "bhojahoj"), "ahoj")
		self.assertEqual(longest_common_substring("hoja", "hojb"), "hoj")
		self.assertEqual(longest_common_substring("ahoja", "bhojb"), "hoj")
		self.assertEqual(longest_common_substring("amatér", "amater"), "amat")
		self.assertEqual(longest_common_substring("", ""), "")
		self.assertEqual(longest_common_substring("", "ahoj"), "")
		self.assertEqual(longest_common_substring("ahoj", ""), "")
		self.assertEqual(longest_common_substring("ahojxahoj", "ahojyahoj"), "ahoj")
		self.assertEqual(longest_common_substring("xyaz", "abc"), "a")
		self.assertEqual(longest_common_substring("xyz", "abc"), "")
		self.assertEqual(longest_common_substring("ahoj", "hojahoj"), "ahoj")
		self.assertEqual(longest_common_substring("hojahoj", "ahoj"), "ahoj")
		self.assertIn(longest_common_substring("abcxyz", "xyzabc"), ("abc", "xyz"))
		self.assertIn(longest_common_substring("xyzabc", "abcxyz"), ("abc", "xyz"))




if __name__ == '__main__':
	#unittest.main()
	perr("Started at %s." % strftime("%c"))
	derinet_file_name = args.derinet
	morfflex_file_name = args.morfflex
	morpho_file_name = args.analyzer
	process_stdin(derinet_file_name, morfflex_file_name, morpho_file_name)
	perr("Finished at %s." % strftime("%c"))
