from collections import defaultdict

import pickle

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ProbTables:
	__slots__ = "affix_counts", "affix_probs", "affix_default_count", "affix_default_prob", "change_counts", "change_probs", "change_default", "change_sub_default", "change_ins_default", "change_del_default", "change_insertions", "change_deletions", "change_substitutions"
	
	def __init__(self, affix_default=0.0, change_default=0.00001):
		self.affix_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
		self.affix_probs = None
		self.affix_default_count = affix_default

		self.change_counts = {"sub": defaultdict(lambda: defaultdict(float)),
		                      "ins": defaultdict(float),
		                      "del": defaultdict(float)}
		self.change_probs = None
		self.change_default = change_default
		self.change_insertions = 0
		self.change_deletions = 0
		self.change_substitutions = 0
	
	def save(self, f):
		pickle.dump(self, f, protocol=pickle.DEFAULT_PROTOCOL, fix_imports=False)
	
	def load(f):
		self = pickle.load(f, fix_imports=False)
		return self
	
	
	def __str__(self):
		return "Affix counts: {}, affix probs: {}, affix_default_count: {}, change_counts: {}, change_probs: {}, change_default: {}, change_sub_default: {}, change_ins_default: {}, change_del_default: {}, change_insertions: {}, change_deletions: {}, change_substitutions: {}".format(str(self.affix_counts), str(self.affix_probs), str(self.affix_default_count), str(self.change_counts), str(self.change_probs), str(self.change_default), str(self.change_sub_default), str(self.change_ins_default), str(self.change_del_default), str(self.change_insertions), str(self.change_deletions), str(self.change_substitutions))
	
	def finalize(self):
		"""Transform the accumulated counts into probs."""
		self.normalize_change_counts()
		self.change_counts = None
		self.normalize_affix_counts()
		self.affix_counts = None
	
	def normalize_change_counts(self):
		# TODO the deletions are further normalized by prevalence. We should normalize the other components as well by the residual amount.
		# TODO normalize properly for the Add-lambda smoothing. We should add the lambda to each type (and to the total once for each type) and to the total once more for the unseen type.
		
		# FIXME normalize the changes and insertions and deletions separetely.
		# FIXME normalize correctly. The tricks with hard integer counts of the various types don't work, because the counts are given by the structure and don't change with iterations. But in reality, I'd like deletions and insertions to become rarer and rarer as time progresses.
		#  I'll have to deduce the correct equations from the laws of probability.
		
		
		# FIXME how to normalize: If we normalize everything as a joint probability, the resulting probs are too low and mess up the stemming by overpowering the affix probabilities. Everything is then analyzed as the shortest possible stem, even if it requires many deletions.
		# Therefore, we have to normalize as conditional probability. But there we have a problem with the insertions – they can occur anywhere, so if we normalize them as a group conditioned by \lambda, they will be too common.
		# A better idea: Consider an insertion to compete with any other change.
		# So: We normalize insertions by c(ins) / c(everything); we normalize substitutions by c(sub y for x) / (c(sub any for x) + c(del x) + c(ins any)); we normalize deletions by c(del x) / (c(sub any for x) + c(del x) + c(ins any))
		# TODO should we exclude insertions from the deletion normalization?
		# TODO maybe a better idea: Consider insertions to be fertility. Therefore, they will be conditioned on a letter – the next one in the parent. Does that help?

		self.change_probs = {"sub": defaultdict(dict),
		                     "ins": {},
		                     "del": {}}
		
		change_insertions_smooth = self.change_insertions + (len(self.change_counts["ins"]) + 1) * self.change_default
		change_deletions_smooth = self.change_deletions + (len(self.change_counts["del"]) + 1) * self.change_default
		change_substitutions_smooth = self.change_substitutions + (sum(len(d) for d in self.change_counts["sub"].values()) + 1) * self.change_default
		changes_total = change_insertions_smooth + change_deletions_smooth + change_substitutions_smooth
		
		if self.change_insertions > 0.0:
			# Normalize insertions.
			insertion_normalizer = 1.0 / changes_total
			self.change_probs["ins"] = {t: (count + self.change_default) * insertion_normalizer for t, count in self.change_counts["ins"].items()}
		
		if self.change_deletions > 0.0:
			# Normalize deletions.
			for f, count in self.change_counts["del"].items():
				#deletion_normalizer = 1.0 / (count + sum(self.change_counts["sub"][f].values()) + self.change_insertions)
				deletion_normalizer = 1.0 / changes_total # TODO
				self.change_probs["del"][f] = (count + self.change_default) * deletion_normalizer
		
		if self.change_substitutions > 0.0:
			# Normalize substitutions.
			for f, to_dict in self.change_counts["sub"].items():
				#substitution_normalizer = 1.0 / (sum(self.change_counts["sub"][f].values()) + self.change_counts["del"][f] + self.change_substitutions)
				substitution_normalizer = 1.0 / changes_total # TODO
				self.change_probs["sub"][f] = {t: (count + self.change_default) * substitution_normalizer for t, count in to_dict.items()}
		
		# Normalize the default values in the proportion the different types were encountered.
		self.change_ins_default = self.change_default * change_insertions_smooth / (changes_total * changes_total)
		self.change_del_default = self.change_default * change_deletions_smooth / (changes_total * changes_total)
		self.change_sub_default = self.change_default * change_substitutions_smooth / (changes_total * changes_total)
	
	
	def normalize_affix_counts(self):
		# Normalize with the Add-lambda smoothing – add the default count to each affix type and once more for the unseen type.
		total = 0.0
		type_count = 0
		
		for pprefix, d1 in self.affix_counts.items():
			for psuffix, d2 in d1.items():
				for cprefix, d3 in d2.items():
					for csuffix, count in d3.items():
						total += count
						type_count += 1
		
		normalizer = 1.0 / (total + (type_count + 1) * self.affix_default_count)
		self.affix_default_prob = self.affix_default_count * normalizer
		
		# Alas, we cannot use defaultdict here, because it allocates all entries upon retrieval, including nonexistent ones, gradually exhausting all available memory.
		# We want nonexistent entries to stay that way.
		#self.affix_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: default))))
		self.affix_probs = {}
		
		#with open("table-normalization", "wt") as f:
		for pprefix, d1 in self.affix_counts.items():
			for psuffix, d2 in d1.items():
				for cprefix, d3 in d2.items():
					for csuffix, count in d3.items():
						normalized_count = (count + self.affix_default_count) * normalizer
						# Not a defaultdict, see above.
						#self.affix_probs[pprefix][psuffix][cprefix][csuffix] = normalized_count
						self.affix_probs.setdefault(pprefix, {}).setdefault(psuffix, {}).setdefault(cprefix, {})[csuffix] = normalized_count
						#print("{}\t{}\t{}\t{}\t{} -> {}".format(pprefix, psuffix, cprefix, csuffix, count, normalized_count), file=f)

	def get_change_prob(self, f, t):
		if f:
			if t:
				return self.change_probs["sub"][f].get(t, self.change_sub_default)
			else:
				return self.change_probs["del"].get(f, self.change_del_default)
		else:
			if t:
				return self.change_probs["ins"].get(t, self.change_ins_default)
			else:
				raise Exception("Attempted to change a zero string to a zero string.")
	
	def add_change(self, f, t, prob):
		if f:
			if t:
				self.change_counts["sub"][f][t] += prob
				self.change_substitutions += prob
			else:
				self.change_counts["del"][f] += prob
				self.change_deletions += prob
		else:
			if t:
				self.change_counts["ins"][t] += prob
				self.change_insertions += prob
			else:
				raise Exception("Attempted to record a change from a zero string to a zero string.")
	
	def get_affix_prob(self, pprefix, psuffix, cprefix, csuffix):
		if pprefix in self.affix_probs and psuffix in self.affix_probs[pprefix] and cprefix in self.affix_probs[pprefix][psuffix]:
			return self.affix_probs[pprefix][psuffix][cprefix].get(csuffix, self.affix_default_prob)
		else:
			return self.affix_default_prob
	
	def add_affix(self, pprefix, psuffix, cprefix, csuffix, prob):
		assert prob >= 0.0
		self.affix_counts[pprefix][psuffix][cprefix][csuffix] += prob
