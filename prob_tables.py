from collections import defaultdict

class ProbTables:
	__slots__ = "prefix_counts", "prefix_probs", "prefix_default_count", "prefix_default_prob", "prefix_total", "suffix_counts", "suffix_probs", "suffix_default_count", "suffix_default_prob", "suffix_total", "change_counts", "change_probs", "change_default", "change_insertions", "change_deletions", "change_substitutions"
	
	def __init__(self, affix_defaults=0.0, change_default=0.00001):
		self.prefix_counts = defaultdict(float)
		self.prefix_probs = None
		self.prefix_default_count = affix_defaults
		self.prefix_default_prob = None
		self.prefix_total = 0.0

		self.suffix_counts = defaultdict(float)
		self.suffix_probs = None
		self.suffix_default_count = affix_defaults
		self.suffix_default_prob = None
		self.suffix_total = 0.0

		self.change_counts = defaultdict(lambda: defaultdict(float))
		self.change_probs = None
		self.change_default = change_default
		self.change_insertions = 0
		self.change_deletions = 0
		self.change_substitutions = 0
	
	def finalize(self):
		"""Transform the accumulated counts into probs."""
		self.normalize_change_counts()
		self.change_counts = None
		self.normalize_affix_counts()
		self.prefix_counts = None
		self.suffix_counts = None
	
	def normalize_change_counts(self):
		# TODO the deletions are further normalized by prevalence. We should normalize the other components as well by the residual amount.
		# TODO normalize properly for the Add-lambda smoothing. We should add the lambda to each type (and to the total once for each type) and to the total once more for the unseen type.

		normalized_counts = defaultdict(dict)
		for f, options in self.change_counts.items():
			normalizer_nominator = sum(options.values())
			if normalizer_nominator <= 0.0:
				# Prevent a divide-by-zero.
				# Skip the update here, effectively resetting all transforms from f
				#  to the default (smoothed) value.
				# TODO this may not be exactly what we want.
				logger.warn("Normalizer nominator of '%s' is %f.", f, normalizer_nominator)
				continue
			normalizer = 1.0 / sum(options.values())
			#print("Normalizer for {} is {}.".format(f, normalizer))
			normalized_counts[f] = {t: count * normalizer for t, count in options.items()}

		if (self.change_insertions + self.change_deletions + self.change_substitutions) > 0:
			# Some changes were recorded. Proceed – otherwise bail out of this step because of a division-by-zero.
			# Further discount deletions by their prevalence in the data.
			insertions_normalizer = self.change_insertions / (self.change_insertions + self.change_deletions + self.change_substitutions)
			normalized_counts[""] = {t: prob * insertions_normalizer for t, prob in normalized_counts[""].items()}

		self.change_probs = normalized_counts
	
	def normalize_affix_counts(self):
		# Normalize with the Add-lambda smoothing – add the default count to each affix type and once more for the unseen type.
		prefix_normalizer = self.prefix_total + (len(self.prefix_counts) + 1) * self.prefix_default_count
		self.prefix_probs = {prefix: (count + self.prefix_default_count) / prefix_normalizer for prefix, count in self.prefix_counts.items()}
		self.prefix_default_prob = self.prefix_default_count / prefix_normalizer
		
		suffix_normalizer = self.suffix_total + (len(self.suffix_counts) + 1) * self.suffix_default_count
		self.suffix_probs = {suffix: (count + self.suffix_default_count) / suffix_normalizer for suffix, count in self.suffix_counts.items()}
		self.suffix_default_prob = self.suffix_default_count / suffix_normalizer

	def get_change_prob(self, f, t):
		return self.change_probs[f].get(t, self.change_default)
	
	def add_change(self, f, t, prob):
		self.change_counts[f][t] += prob
		if f:
			if t:
				self.change_substitutions += 1
			else:
				self.change_deletions += 1
		else:
			if t:
				self.change_insertions += 1
			else:
				raise Exception("Attempted to record a change from a zero string to a zero string.")
	
	def get_affix_prob(self, prefix, suffix):
		return self.prefix_probs.get(prefix, self.prefix_default_prob) * self.suffix_probs.get(suffix, self.suffix_default_prob)
	
	def add_affix(self, prefix, suffix, prob):
		self.prefix_counts[prefix] += prob
		self.suffix_counts[suffix] += prob
		self.prefix_total += prob
		self.suffix_total += prob
