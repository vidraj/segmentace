#!/usr/bin/env python3

# Usage: xz -dkvvc morfflex.xz | list-morfflex-affixes pref.txt suf.txt
# List all affixes of forms, detected using a simple substring algorithm,
#  and their counts in the data.

import sys
from collections import Counter

from lexeme import longest_common_substring_position, divide_string
from segmentace import techlemma_to_lemma

#def longest_common_substring_position(string_a, string_b):
#	"""Returns (start of lcs in string_a, start of lcs in string_b, length of lcs)"""
#	# Initialize the longest-suffixes table.
#	length_table = [[0] * (1 + len(string_b)) for i in range(1 + len(string_a))]
#	
#	# These will hold information about where the longest common substring ends in the strings and about its length.
#	longest_len = 0
#	longest_substr_end_x = 0
#	longest_substr_end_y = 0
#	
#	# Walk through the strings, matching all possible suffixes.
#	for x in range(len(string_a)):
#		for y in range(len(string_b)):
#			if string_a[x] == string_b[y]:
#				# We can extend the current longest suffix here by one.
#				length_table[x + 1][y + 1] = length_table[x][y] + 1
#				
#				if length_table[x + 1][y + 1] > longest_len:
#					# If we've found a new longest substring, store information about it for later retrieval.
#					longest_len = length_table[x + 1][y + 1]
#					longest_substr_end_x = x + 1
#					longest_substr_end_y = y + 1
#			else:
#				# We cannot extend the current longest suffix, so start a new match instead of extending the old one.
#				length_table[x + 1][y + 1] = 0
#	
#	return (longest_substr_end_x - longest_len, longest_substr_end_y - longest_len, longest_len)
#
#def longest_common_substring(string_a, string_b):
#	pos_x, pos_y, length = longest_common_substring_position(string_a, string_b)
#	return string_a[pos_x:pos_x+length]
#
#def divide_string(s, p1, p2):
#	"""Divide s into 3 parts: [0:p1], [p1:p2], [p2:]"""
#	return (s[0:p1], s[p1:p2], s[p2:])
#
#
#def techlemma_to_lemma(techlemma):
#	"""Cut off the technical suffixes from the string techlemma and return the raw lemma"""
#	shortlemma = re.sub("[_`].+", "", techlemma)
#	lemma = re.sub("-\d+$", "", shortlemma)
#	return lemma

def stem_by_link(parent, child):
    """Divide child into three parts (prefix, stem, suffix) according to string similarity to parent.
    Return the three parts."""
    start_parent, start_child, length = longest_common_substring_position(parent, child)
    prefix, stem, suffix = divide_string(child, start_child, start_child + length)
    return prefix, stem, suffix


def get_better_parent_tags(tag):
    """Given a morphological tag, find the tags for parents which are morphologically closer than the lemma.

    The Czech inflectinoal morphology captures some borderline-derivational processes as well – these are grade (comparative / superlative) and negation. Comparing the morphological suffixes of an e.g. superlative form with its lemma will not yield a single-morph suffix on the form's side. To obtain such a minimal pair, we have to compare inflected superlative forms with the base superlative form, base superlatives with comparatives etc.

    This function finds the tag of such a minimal pair. And because sometimes the form with the closest tag doesn't exist (i.e. for some stylistically varied forms), it returns a list sorted by closeness."""

    if tag[9] not in {"1", "-"} or tag[10] not in {"A", "-"}:
        tag_list = list(tag)
        tag_candidates = []

        # How to find the tag of the immediate parent for a form?
        # 1) Partially lemmatize superlatives and comparatives → „This means nominative singular for nouns, the same plus masculine positive for adjectives, similarly for pronouns and numerals. Verbs are represented by their infinitive forms.“

        # Drop number, case and gender.
        changed_number_case_gender = False
        if tag_list[3] != "-" and tag_list[3] != "S":
            # Drop number.
            tag_list[3] = "S"
            changed_number_case_gender = True

        if tag_list[4] != "-" and tag_list[4] != "1":
            # Drop case.
            tag_list[4] = "1"
            changed_number_case_gender = True

        # Drop the gender only for non-nouns. (In Czech, the gender of nouns is a lexical property.)
        if tag_list[0] != "N" and tag_list[2] != "-" and tag_list[2] != "M":
            # Drop gender.
            tag_list[2] = "M"
            changed_number_case_gender = True

        if changed_number_case_gender:
            # Append the partially lemmatized tag…
            tag_candidates.append("".join(tag_list))
            # … and if the variant is not base, drop that as a backup.
            if tag_list[14] != '-':
                tag_candidates.append("".join(tag_list[:14]) + "-")



        # The tag is already partially lemmatized. Now, drop the grade, or the negation, whatever is possible.
        # Because of the Czech morpheme order (nejne…), we want to process position grade first, then negation.

        if tag_list[9] != "1" and tag_list[9] != "-":
            # Drop grade.
            if tag_list[9] == "3":
                tag_list[9] = "2"
            elif tag_list[9] == "2":
                tag_list[9] = "1"
            else:
                print("Error: Bad grade in tag {}".format(tag), file=sys.stderr, flush=True)

            # Append the partially lemmatized tag…
            tag_candidates.append("".join(tag_list))
            # … and if the variant is not base, drop that as a backup.
            if tag_list[14] != '-':
                tag_candidates.append("".join(tag_list[:14]) + "-")

        if tag_list[10] == "N":
            # Drop negation.
            tag_list[9] = "A"

            # Append the partially lemmatized tag…
            tag_candidates.append("".join(tag_list))
            # … and if the variant is not base, drop that as a backup.
            if tag_list[14] != '-':
                tag_candidates.append("".join(tag_list[:14]) + "-")

        return tag_candidates
    else:
        return []


prefixes = Counter()
suffixes = Counter()
#total = 0

tag_to_form = {}
deferred_pairs = []
last_lemma = ""

for line in sys.stdin:
    lemma, tag, form = line.rstrip("\n").split("\t")

    if lemma != last_lemma:
        # We have started processing another tree.
        # Process all the deferred pairs.
        for candidate_parent_tags, child in deferred_pairs:
            # Go over all the proposed candidate tags, find the first one that matches an actual form and consider that one to be the true parent.
            for parent_tag in candidate_parent_tags:
                if parent_tag in tag_to_form:
                    parent = tag_to_form[parent_tag]
                    prefix, stem, suffix = stem_by_link(parent, child)

                    prefixes[prefix] += 1
                    suffixes[suffix] += 1

                    break
            else:
                # If no candidates were found, compare the child with the lemma.
                    parent = last_lemma
                    prefix, stem, suffix = stem_by_link(parent, child)
                    prefixes[prefix] += 1
                    suffixes[suffix] += 1

        # Reset the deferred storage.
        tag_to_form = {}
        deferred_pairs = []
        last_lemma = lemma


    tag_to_form[tag] = form

    # We need to partially treeify the flat list – some forms are “derived”
    #  from lemmas, but some descend more naturally from other forms, e.g.
    #  negatives, comparatives and superlatives.
    # But there is no guarantee that the parent was read before the child.
    # So instead of processing outright, store these pairs in a list and
    #  process them after all forms have been read.

    # Check whether the form is derived from another form.
    better_parent_tags = get_better_parent_tags(tag)

    if not better_parent_tags:
        # The default case when there is no better parent tag.
        parent = techlemma_to_lemma(lemma)
    else:
        if better_parent_tags[0] in tag_to_form:
            # We have already loaded the best parent.
            parent = tag_to_form[better_parent_tags[0]]
        else:
            # The parent will (hopefully) come in the future.
            deferred_pairs.append((better_parent_tags, form))
            parent = None

    if parent is not None:
        # The parent is not deferred, process it outright.
        prefix, stem, suffix = stem_by_link(parent, form)
        prefixes[prefix] += 1
        suffixes[suffix] += 1

with open(sys.argv[1], "wt") as f:
    for prefix, count in prefixes.most_common():
        print("{}\t{}".format(prefix, count), file=f)

with open(sys.argv[2], "wt") as f:
    for suffix, count in suffixes.most_common():
        print("{}\t{}".format(suffix, count), file=f)
