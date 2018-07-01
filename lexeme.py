#!/usr/bin/env python3

from collections import defaultdict, namedtuple
import itertools
import math

import unittest
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

INS = 1
DEL = 2
SUB = 3

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


def extend_hypothesis(tables, hypothesis, f, t):
	change = (f, t)
	change_prob = tables.get_change_prob(f, t)
	if hypothesis is not None:
		# Extend the existing hypothesis with the change.
		return (hypothesis[0] * change_prob, hypothesis[1] + [change])
	else:
		return (change_prob, [change])

def init_hypotheses():
	return ((1.0, True, tuple()), )

def extend_hypotheses(tables, hypotheses, f, t):
	assert hypotheses, "Hypotheses must not be empty; use init_hypotheses first."
	change = (f, t)
	change_prob = tables.get_change_prob(f, t)
	
	# Extend the existing hypothesis with the change.
	if t == "":
		# Deletion. Check that it is allowed.
		return tuple((prob * change_prob, del_allowed, changes + (change, )) for prob, del_allowed, changes in hypotheses if del_allowed)
	elif f == "":
		# Insertion. Disallow deletion in the next step.
		return tuple((prob * change_prob, False, changes + (change, )) for prob, _, changes in hypotheses)
	else:
		# Substitution. Allow deletion in the next step.
		return tuple((prob * change_prob, True, changes + (change, )) for prob, _, changes in hypotheses)

def generate_string_mapping_hypotheses(tables, parent_stem, child_stem, hypothesis, allowed_map_types):
	if not parent_stem and not child_stem:
		assert hypothesis is not None, "Hypothesis must not be None, probably a mapping to an empty string was attempted."
		yield hypothesis
	else:
		if SUB in allowed_map_types and parent_stem and child_stem:
			#print("sub", parent_stem[0], child_stem[0])
			sub_hypothesis = extend_hypothesis(tables, hypothesis, parent_stem[0], child_stem[0])
			yield from generate_string_mapping_hypotheses(tables, parent_stem[1:], child_stem[1:], sub_hypothesis, {INS, DEL, SUB})

		if INS in allowed_map_types and child_stem:
			#print("ins", child_stem[0])
			ins_hypothesis = extend_hypothesis(tables, hypothesis, "", child_stem[0])
			yield from generate_string_mapping_hypotheses(tables, parent_stem, child_stem[1:], ins_hypothesis, allowed_map_types - {DEL})

		if DEL in allowed_map_types and parent_stem:
			#print("del", parent_stem[0])
			del_hypothesis = extend_hypothesis(tables, hypothesis, parent_stem[0], "")
			yield from generate_string_mapping_hypotheses(tables, parent_stem[1:], child_stem, del_hypothesis, allowed_map_types - {INS})


#def map_strings(tables, parent_stem, child_stem, new_tables, prob_modifier):
	## I need an algorithm to walk all possibilities and add the final probabilities to new_tables.
	#total_prob = 0.0
	#for prob, hypothesis in generate_string_mapping_hypotheses(tables, parent_stem, child_stem, None, {INS, DEL, SUB}):
		##print("Obtained hypothesis with prob {}:".format(prob), hypothesis)
		#total_prob += prob
		#for (f, t) in hypothesis:
			#new_tables.add_change(f, t, prob * prob_modifier)
	#return total_prob
#def map_strings(tables, parent_stem, child_stem, new_tables, prob_modifier):
	#"""The complete version of the tabular algorithm, with the 'only keep two lines in memory at once' optimization."""
	## I need an algorithm to walk all possibilities and add the final probabilities to new_tables.
	## Table algorithm.
	
	#assert len(parent_stem) >= 1, "The parent stem must not be empty."
	#assert len(child_stem) >= 1, "The child stem must not be empty."
	
	## Initialize the first row.
	#prev_line = [None] * (len(child_stem) + 1)
	#prev_line[0] = init_hypotheses() # The initial empty hypothesis.
	#for j in range(1, len(child_stem) + 1):
		#prev_line[j] = extend_hypotheses(tables, prev_line[j - 1], "", child_stem[j - 1])
	
	## Initialize the second row, i.e. the one to be filled next.
	#cur_line = [None] * (len(child_stem) + 1)
	
	## Fill in the rest of the table.
	#for i in range(1, len(parent_stem) + 1):
		## Fill in the first item (it is special, because there is only one hypothesis instead of three.
		#cur_line[0] = extend_hypotheses(tables, prev_line[0], parent_stem[i - 1], "")
		
		## Fill in the rest of the line.
		#for j in range(1, len(child_stem) + 1):
			#cur_line[j] = (extend_hypotheses(tables, prev_line[j - 1], parent_stem[i - 1], child_stem[j - 1])
			             #+ extend_hypotheses(tables, cur_line[j - 1],  "",                 child_stem[j - 1])
			             #+ extend_hypotheses(tables, prev_line[j],     parent_stem[i - 1], ""))
		
		## Swap the lines (the filled-in cur_line will be overwritten in the next iteration).
		#tmp = prev_line
		#prev_line = cur_line
		#cur_line = tmp
	
	## The bottom right corner contains the hypothesis list. Calculate the final prob from it.
	## Beware that the lines have been swapped at the end of the last cycle, so the bottom right corner is in prev_line.
	#total_prob = 0.0
	#for prob, hypothesis in prev_line[-1]:
		##print("Obtained hypothesis with prob {}:".format(prob), hypothesis)
		#modified_prob = prob * prob_modifier
		#total_prob += prob
		#for (f, t) in hypothesis:
			#new_tables.add_change(f, t, modified_prob)
	#return total_prob
#def map_strings(tables, parent_stem, child_stem, new_tables, prob_modifier):
	#"""The windowed of the tabular algorithm, with the 'only keep two lines in memory at once' and 'only consider a small window' (borrowed from DTW) optimizations."""
	
	## Lengths of the window – to the left and to the right.
	#window_size = (-2, 2)
	
	#assert len(parent_stem) >= 1, "The parent stem must not be empty."
	#assert len(child_stem) >= 1, "The child stem must not be empty."
	
	## TODO enlarge the window dynamically if we map a very short stem to a very long one, so that we can reach the start from the first character and the end from the last one.
	## TODO and make sure that the windows overlap between steps.
	
	## Initialize the first row.
	#prev_line = [None] * (len(child_stem) + 1)
	#prev_line[0] = init_hypotheses() # The initial empty hypothesis.
	#for j in range(1, min(len(child_stem), window_size[1]) + 1):
		#prev_line[j] = extend_hypotheses(tables, prev_line[j - 1], "", child_stem[j - 1])
	
	## Fill in the rest of the table. Go through the whole parent stem.
	#for i in range(1, len(parent_stem) + 1):
		## Initialize the second row, i.e. the one to be filled next.
		#cur_line = [None] * (len(child_stem) + 1)
		
		## Calculate the center of the window.
		#cur_item = i * (len(child_stem) + 1) / (len(parent_stem) + 1)
		
		## Calculate the window bounds. The upper bound is exclusive.
		#window_start = max(0, math.floor(cur_item) + window_size[0])
		#window_end = min(len(child_stem), math.ceil(cur_item) + window_size[1]) + 1
		
		## Fill in the rest of the line. Map only the items from within the window.
		#for j in range(window_start, window_end):
			#cur_line[j] = tuple()
			#if j >= 1:
				#if prev_line[j - 1] is not None:
					#cur_line[j] += extend_hypotheses(tables, prev_line[j - 1], parent_stem[i - 1], child_stem[j - 1])
				#if cur_line[j - 1] is not None:
					#cur_line[j] += extend_hypotheses(tables, cur_line[j - 1], "", child_stem[j - 1])
			#if prev_line[j] is not None:
				#cur_line[j] += extend_hypotheses(tables, prev_line[j], parent_stem[i - 1], "")
		
		## Swap the lines.
		## It doesn't matter that cur_line stays the same, it is overwritten with a new initialization next.
		#prev_line = cur_line
	
	#assert prev_line[-1] is not None, "The last item never got into the window for stems '{}' and '{}'! Window sizes: cur: {}, start: {}, end: {}.".format(parent_stem, child_stem, cur_item, window_start, window_end)
	
	## The bottom right corner contains the hypothesis list. Calculate the final prob from it.
	## Beware that the lines have been swapped at the end of the last cycle, so the bottom right corner is in prev_line.
	## That one item should definitely be in the window, so it should never be None.
	#total_prob = 0.0
	#for prob, hypothesis in prev_line[-1]:
		##print("Obtained hypothesis with prob {}:".format(prob), hypothesis)
		#modified_prob = prob * prob_modifier
		#total_prob += prob
		#for (f, t) in hypothesis:
			#new_tables.add_change(f, t, modified_prob)
	#return total_prob
def map_strings(tables, parent_stem, child_stem, new_tables, prob_modifier):
	"""The complete version of the tabular algorithm, with the 'only keep two lines in memory at once' optimization and n-best-list option culling."""
	# I need an algorithm to walk all possibilities and add the final probabilities to new_tables.
	# Table algorithm.
	
	# TODO put this into an argument.
	n_best_limit = 100
	
	assert len(parent_stem) >= 1, "The parent stem must not be empty."
	assert len(child_stem) >= 1, "The child stem must not be empty."
	
	# Initialize the first row.
	prev_line = [None] * (len(child_stem) + 1)
	prev_line[0] = init_hypotheses() # The initial empty hypothesis.
	for j in range(1, len(child_stem) + 1):
		prev_line[j] = extend_hypotheses(tables, prev_line[j - 1], "", child_stem[j - 1])
	
	# Initialize the second row, i.e. the one to be filled next.
	cur_line = [None] * (len(child_stem) + 1)
	
	# Fill in the rest of the table.
	for i in range(1, len(parent_stem) + 1):
		# Fill in the first item (it is special, because there is only one hypothesis instead of three.
		cur_line[0] = extend_hypotheses(tables, prev_line[0], parent_stem[i - 1], "")
		
		# Fill in the rest of the line.
		for j in range(1, len(child_stem) + 1):
			hypotheses = (extend_hypotheses(tables, prev_line[j - 1], parent_stem[i - 1], child_stem[j - 1])
			            + extend_hypotheses(tables, cur_line[j - 1],  "",                 child_stem[j - 1])
			            + extend_hypotheses(tables, prev_line[j],     parent_stem[i - 1], ""))
			cur_line[j] = sorted(hypotheses, reverse=True)[:n_best_limit]
		
		# Swap the lines (the filled-in cur_line will be overwritten in the next iteration).
		tmp = prev_line
		prev_line = cur_line
		cur_line = tmp
	
	# The bottom right corner contains the hypothesis list. Calculate the final prob from it.
	# Beware that the lines have been swapped at the end of the last cycle, so the bottom right corner is in prev_line.
	total_prob = 0.0
	# TODO normalize probs before updating the counts.
	#  We assume that there has to be a mapping from parent_stem to child_stem with a probability of prob_modifier.
	#   FIXME is this correct? Shouldn't we normalize the prob_modifier somewhere first? Across all possible segmentations of a single word? Across all parent segmentation choices? Etc.
	normalizer = prob_modifier / sum([prob for prob, del_allowed, hypothesis in prev_line[-1]])
	for prob, del_allowed, hypothesis in prev_line[-1]:
		#print("Obtained hypothesis with prob {}:".format(prob), hypothesis)
		modified_prob = prob * normalizer
		total_prob += prob
		for (f, t) in hypothesis:
			new_tables.add_change(f, t, modified_prob)
	return total_prob



class Lexeme:
	__slots__ = "id", "lemma", "parent_id", "parent_lemma", "parent", "children", "morph_bounds", "morph_change_type", "stem_map"
	
	# Types of morpheme changes encountered in the data and their frequencies.
	# TODO detect and ignore suppletion.
	morph_change_types = {"padd": 0, # Prefix addition
	                      "prem": 0, # Prefix removal
	                      "pcha": 0, # Prefix substitution
	                      "sadd": 0, # Suffix addition
	                      "srem": 0, # Suffix removal
	                      "scha": 0, # Suffix substitution
	                      "conv": 0, # Conversion (no change)
	                      "circ": 0} # Circumfixation (or weird change on both sides)
	
	allowed_morph_change_types = {"padd", "sadd", "scha", "conv",
	                              "prem", "pcha", "srem", "circ" # Normally disallowed types, allowed for debugging.
	                              }
	
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
		if self.id is not None:
			return "%s (%d)" % (self.lemma, self.id)
		else:
			return "%s (from MorfFlex)" % self.lemma
	
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
	
	def count_stems_simple(self, tables):
		"""Find the longest common substring of self.lemma and parent.lemma. Consider that substring to be the stem, and the rest to be prefixes/suffixes. Store counts of the suffixes of self in tables."""
		
		# If my own stem map can be filled in, but has not been filled in yet, do it now.
		if self.parent is not None:
			if self.stem_map is None:
				pprefix, pstem, psuffix, sprefix, sstem, ssuffix = self.find_stem_map_simple()
				tables.add_affix(pprefix, psuffix, sprefix, ssuffix, 1.0)
				#assert pstem == sstem, "The stems found by string comparison don't match: '{}' and '{}'".format(pstem, sstem) # This assert is wrong, we ignore case, making them unequal.
				for char in pstem:
					tables.add_change(char, char, 1.0)
			else:
				logger.warn("Stem map of lexeme '%s' has already been filled in.", self.lemma)
	
	def estimate_probabilities(self, tables, new_tables):
		if self.parent is None:
			# We don't attempt to divide roots at all. They will only be divided in the next phase, transitively using the divisions found in their children.
			return 1.0
		
		best_prob = -1.0
		hypotheses = []
		for child_stem_start in range(len(self.lemma)):
			for child_stem_end in range(len(self.lemma), child_stem_start, -1):
				child_prefix, child_stem, child_suffix = divide_string(self.lemma, child_stem_start, child_stem_end)
				
				for parent_stem_start in range(len(self.parent.lemma)):
					for parent_stem_end in range(len(self.parent.lemma), parent_stem_start, -1):
						parent_prefix, parent_stem, parent_suffix = divide_string(self.parent.lemma, parent_stem_start, parent_stem_end)
				
						# TODO when the prefix is unknown, take the next smallest known suffix and multiply it by a discount based on the length difference.
						#  That may prevent probability collapse caused by the roots – attempting to find the best root will always find the smallest one,
						#  because there, the length actually plays a role.
						#  Or rather, perform some smoothing.
						child_affix_prob = tables.get_affix_prob(parent_prefix, parent_suffix, child_prefix, child_suffix)
						
						if child_affix_prob == 0.0:
							# Don't perform the computationally expensive stem mapping, if the word probability will be 0 anyway.
							continue
						
						child_stem_prob = map_strings(tables, parent_stem, child_stem, new_tables, child_affix_prob)
						
						# FIXME right now, the probability is p(seen prefix) * p(stem maps to parent) * p(seen suffix). This is imbalanced and leads to spurious splits (esp. spurious prefixes).
						#  Solution: probability should be p(seen prefix) * p(seen stem) * p(seen suffix) * p(prefix does not map to parent) * p(stem maps to parent) * p(suffix does not map to parent)
						#  = p(seen prefix) * p(seen stem) * p(seen suffix) * (1 - p(prefix maps to parent)) * p(stem maps to parent) * (1 - p(suffix maps to parent))
						#   NO! p(seen stem) should be almost constantly zero. Only words that share a parent may have the same stem.
						#   Therefore, it should be p(seen prefix) * p(seen suffix) * (1 - p(prefix maps to parent)) * p(stem maps to parent) * (1 - p(suffix maps to parent))
						word_prob = child_stem_prob * child_affix_prob
						
						if word_prob > 0.0:
							# Record the hypothesis if it is plausible.
							hypotheses.append((word_prob, child_affix_prob, child_stem_prob, parent_prefix, parent_suffix, child_prefix, child_stem, child_suffix, child_stem_start, child_stem_end, parent_stem_start, parent_stem_end))
		
		
		normalizer = 1 / sum([item[0] for item in hypotheses])
		
		for word_prob, child_affix_prob, child_stem_prob, parent_prefix, parent_suffix, child_prefix, child_stem, child_suffix, child_stem_start, child_stem_end, parent_stem_start, parent_stem_end in hypotheses:
			new_tables.add_affix(parent_prefix, parent_suffix, child_prefix, child_suffix, word_prob * normalizer)
		
		# Record information about the best segmentation.
		best_prob, child_affix_prob, child_stem_prob, parent_prefix, parent_suffix, child_prefix, child_stem, child_suffix, child_stem_start, child_stem_end, parent_stem_start, parent_stem_end = max(hypotheses)
		best_prob = best_prob * normalizer
		logger.debug("Dividing '%s-%s-%s'.\tp(stem) = %e, p(affix) = %e, p(word) = %e", child_prefix, child_stem, child_suffix, child_stem_prob, child_affix_prob, best_prob)
		self.stem_map = {"self": (child_stem_start, child_stem_end),
		                 "parent": (parent_stem_start, parent_stem_end)}
		self.morph_bounds = {0, self.stem_map["self"][0], self.stem_map["self"][1], len(self.lemma)}
		
		if best_prob == 0.0:
			logger.warn("No segmentation obtained for '%s'.", self.lemma)
		
		assert best_prob >= 0.0 and best_prob <= 1.0, "Best found p(%s) = %e is out of bounds." % (self.lemma, best_prob)
		
		return best_prob
		
	
	def find_stem_map_simple(self, parent=None):
		"""Look at the lemma of self and parent and try to detect any morph changes between the two.
		If there are any, set the appropriate morph bounds.
		If parent is None, use self.parent instead.
		Return the obtained prefix, stem and suffix of self.lemma."""
		
		if parent is None:
			parent = self.parent
		
		if parent is None:
			logger.warn("Trying to detect morph bounds of %s, which has no parent!", self.to_string())
			return None, None, None
		
		plemma = parent.lemma
		slemma = self.lemma
		stem_start_parent, stem_start_self, stem_length = longest_common_substring_position(plemma.lower(), slemma.lower())
		# FIXME what if there is no common substring? stem_length can be zero. Fix that with a special code case.
		
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
					logger.debug("Circumfixation in %s → %s.", plemma, slemma)
					morph_change_type = "circ"
				else:
					morph_change_type = "padd"
		else:
			# PREM / PCHA
			if stem_end_off_parent != 0 or stem_end_off_self != 0:
				logger.debug("Circumfixation in %s → %s.", plemma, slemma)
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
			logger.debug("Divided by %s: '%s' → '%s'", morph_change_type, "/".join(psegments), "/".join(ssegments))
			self.morph_bounds = self.morph_bounds.union(bounds_self)
			parent.morph_bounds = parent.morph_bounds.union(bounds_parent)
		else:
			logger.debug("Divided by %s: '%s' → '%s' (spurious, not saving)", morph_change_type, "/".join(psegments), "/".join(ssegments))
			pass
		
		return pprefix, pstem, psuffix, sprefix, sstem, ssuffix
	
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
	pass
