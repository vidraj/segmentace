#!/usr/bin/env python3

import unittest

from log import perr

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
