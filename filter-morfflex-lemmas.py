import sys

# Load the list of lemmas from DeriNet.
allowed_lemma_list = []
with open(sys.argv[1], "rt", encoding="utf-8") as f:
	for line in f:
		lemma = line.rstrip()
		allowed_lemma_list.append(lemma)
allowed_lemma_set = set(allowed_lemma_list)
del(allowed_lemma_list)

for line in sys.stdin:
	lemma, pos, form = line.rstrip().split("\t")
	if lemma in allowed_lemma_set:
		print(line, end="")
