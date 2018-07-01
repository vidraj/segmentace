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
		
		changes_total = self.change_insertions + self.change_deletions + self.change_substitutions
		if changes_total > 0:
			# There are some recorded changes. Normalize them. (Otherwise do nothing to prevent division-by-zero errors.
			
			# Normalize substitutions.
			substitution_prevalence = self.change_substitutions / changes_total
			for f, options in self.change_counts["sub"].items():
				normalizer_nominator = sum(options.values())
				if normalizer_nominator <= 0.0:
					# Prevent a divide-by-zero.
					# Skip the update here, effectively resetting all transforms from f
					#  to the default (smoothed) value.
					# TODO this may not be exactly what we want.
					logger.warn("Normalizer nominator of '%s' is %f.", f, normalizer_nominator)
					continue
				normalizer = substitution_prevalence / normalizer_nominator
				#logger.debug("Normalizer for {} is {}.".format(f, normalizer))
				self.change_probs["sub"][f] = {t: count * normalizer for t, count in options.items()}
			
			
			insertion_sum = sum(self.change_counts["ins"].values())
			if insertion_sum != 0.0:
				# Normalize insertions.
				insertion_normalizer = self.change_insertions / (changes_total * insertion_sum)
				self.change_probs["ins"] = {t: count * insertion_normalizer for t, count in self.change_counts["ins"].items()}
			else:
				logger.warn("No insertions.")
				self.change_probs["ins"] = {}
			
			
			deletion_sum = sum(self.change_counts["del"].values())
			if deletion_sum != 0.0:
				# Normalize deletions.
				deletion_normalizer = self.change_deletions / (changes_total * deletion_sum)
				self.change_probs["del"] = {f: count * deletion_normalizer for f, count in self.change_counts["del"].items()}
			else:
				logger.warn("No deletions.")
				self.change_probs["del"] = {}
		
			# TODO normalize the defaults as well.
			self.change_sub_default = self.change_default * self.change_substitutions / changes_total
			self.change_ins_default = self.change_default * self.change_insertions / changes_total
			self.change_del_default = self.change_default * self.change_deletions / changes_total
		else:
			# No changes were recorded, the normalizers are unknown. Take the default at its face value.
			self.change_sub_default = self.change_default
			self.change_ins_default = self.change_default
			self.change_del_default = self.change_default
	
	
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
