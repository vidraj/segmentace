#!/usr/bin/env python3

import re
import itertools
from collections import defaultdict
import logging

import gzip
import lzma

from lexeme import Lexeme

# Load MorphoDiTa if available, otherwise fail silently.
# MorphoDiTa availability must be tested anywhere it is used in the program!
morphodita_available = False
try:
	from ufal.morphodita import Tagger, TaggedLemmas
	morphodita_available = True
except ImportError:
	pass

logger = logging.getLogger(__name__)


def techlemma_to_lemma(techlemma):
	"""Cut off the technical suffixes from the string techlemma and return the raw lemma"""
	shortlemma = re.sub("[_`].+", "", techlemma)
	lemma = re.sub("-\d+$", "", shortlemma)
	return lemma

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
	def __init__(self, derinet_file_name):
		id_to_lexeme = {}
		lemma_to_lexemes = defaultdict(list)
		
		with DeriNetParser(derinet_file_name) as derinet:
			for lexeme in derinet:
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
		with MorfFlexParser(morfflex_file_name, derinet_db) as morfflex:
			for lexeme in morfflex:
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
		

class Segmentace:
	def __init__(self, derinet_file_name, morfflex_file_name, morpho_file_name):
		logger.info("Loading derivations.")
		derinet_db = DeriNetDatabase(derinet_file_name)
		logger.info("Derivations loaded.")
		
		if morfflex_file_name is not None:
			logger.info("Loading inflections.")
			db = MorfFlexDatabase(morfflex_file_name, derinet_db)
			logger.info("Inflections loaded.")
		else:
			logger.info("Not loading inflections.")
			db = derinet_db
		
		logger.info("Detecting stem bounds.")
		
		for node in db.iter():
			node.detect_stems()
		
		logger.info("Stem bounds detected.")
		logger.info("Propagating morph bounds.")
		
		for root in db.iter_trees():
			root.propagate_morph_bounds()
		
		logger.info("Morph bounds propagated.")
		
		lemmas = []
		tagger = None
		if morpho_file_name is not None:
			logger.info("Loading morphology")
			if morphodita_available:
				tagger = Tagger.load(morpho_file_name)
			else:
				logger.error("You need to install the MorphoDiTa Python bindings!")
			
			if not tagger:
				logger.critical("Cannot load morphological dictionary from file '%s'.",  morpho_file_name)
				sys.exit(1)
			
			lemmas = TaggedLemmas()
			logger.info("Morphology loaded.")
		else:
			logger.info("No morphological dictionary specified. Inflectional morphology will not be available.")
			tagger = None
		
		self.db = db
		self.tagger = tagger
		self.lemmas = lemmas
	
	def segment_word(self, word, analysis=None):
		"""Takes a string representation of the word form to segment and (optionally) its analysis (Lemma) returned by MorphoDiTa. Returns a list of strings representing the individual morphs of the word."""
		node = None
		parent_node = None
		
		# First, try to find the word in the database.
		nodes = self.db.get_by_lemma(word)
		if nodes:
			node = nodes[0]
		elif analysis is not None:
			# If the word is not in the database itself, try to find its lemma there.
			lemma = techlemma_to_lemma(analysis.lemma)
			parent_nodes = self.db.get_by_lemma(lemma)
			if parent_nodes:
				# The word is not in the database, but its lemma is.
				# Create a new node for the word and propagate the bounds to it.
				parent_node = parent_nodes[0]
				node = Lexeme(word, parent_lemma=lemma)
				node.detect_stems(parent_node)
				node.copy_morph_bounds(parent_node)
		
		if node:
			return node.morphs()
		else:
			# If all else fails, consider the word to be a single morph.
			logger.debug("Word '%s' not recognized. No segmentation given.", word)
			return [word]
		
	
	def segment_sentence(self, input_sentence):
		words = ["".join(morphs) for morphs in input_sentence]
		output_sentence = []
		
		if self.tagger:
			self.tagger.tag(words, self.lemmas)
		
		for word, analysis in itertools.zip_longest(words, self.lemmas):
			output_sentence.append(self.segment_word(word, analysis))
		
		return output_sentence
