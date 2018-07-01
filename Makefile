.PHONY: all compare download clean plot

# Never ever remove any intermediate files.
# .PRECIOUS:
.SECONDARY:

# If PYTHON is undefined, set it to pypy3 or python3, whichever comes first.
ifeq ($(origin PYTHON), undefined)
    PYTHON::=$(shell { command -v pypy3 || command -v python3 ; } 2>/dev/null)
endif


MORFESSOR_MODEL::=morfessor-model

DATA_SOURCE::=wmt17-nmt-training-task-package/README wmt17-nmt-training-task-package/train.cs.iso8859-2.txt
GOLD_STANDARD_DATA::=gold-standard-data.txt

TRAIN_CORPUS-cs::=wmt17-nmt-training-task-package/train.cs.gz
TRAIN_CORPUS-cs-iso::=wmt17-nmt-training-task-package/train.cs.iso8859-2.txt
TRAIN_CORPUS-en::=wmt17-nmt-training-task-package/train.en.gz

MORPHO_TAGGER::=czech-morfflex-pdt-161115/czech-morfflex-pdt-161115-pos_only.tagger
DERINET::=derinet-1-5-1.tsv.gz
MORFFLEX::=~/škola/svn/derinet/tools/data-api/perl-derivmorpho/data/morfflex-cz.2016-11-15.utf8.lemmaID_suff-tag-form.tab.csv.xz

DATE::=$(shell date '+%Y-%m-%d-%H-%M-%S')

# all: stats-derinet-morphodita-cs.txt
all: precision-recall-derinet-$(DATE).txt

plot: pref-A-count-histogram.png pref-D-count-histogram.png pref-N-count-histogram.png pref-V-count-histogram.png
plot: suf-A-count-histogram.png suf-D-count-histogram.png suf-N-count-histogram.png suf-V-count-histogram.png
plot: pref-A-coverage.png pref-D-coverage.png pref-N-coverage.png pref-V-coverage.png
plot: suf-A-coverage.png suf-D-coverage.png suf-N-coverage.png suf-V-coverage.png

quick-test.out: derinet-test.tsv
	cut -f2 "$<" | "${PYTHON}" ./segment-by-derinet.py --from spl --to hmorph --em-threshold 0.1 "$<" > "$@"



compare: stats-morfessor-cs.txt stats-morfessor-en.txt # stats-affisix-cs-iso.txt
compare: stats-derinet-morphodita-cs.txt
compare: stats-bpe-1000-cs.txt stats-bpe-30000-cs.txt stats-bpe-50000-cs.txt stats-bpe-85000-cs.txt
compare: stats-bpe-1000-en.txt stats-bpe-30000-en.txt stats-bpe-50000-en.txt stats-bpe-85000-en.txt
compare: stats-corpus-cs.txt stats-corpus-en.txt

download: czech-morfflex-pdt-161115/README $(DERINET)

# Run Morfessor on the input, save a mapping from words to segments.
morfessor-vocab-%.txt: $(MORFESSOR_MODEL)-%.bin $(DATA_SOURCE)
	# TODO strip all punctuation and uninteresting words from the input.
	ulimit -t unlimited && zcat $(TRAIN_CORPUS-$*) | sed -e 's/ /\n/g' | sort -u | nice -n 19 morfessor -l "$<" -T - --logfile "morfessor-predict-$*-log.txt" > "$@"

# TODO $(TRAIN_CORPUS) blows when DATA_SOURCE wasn't made yet.
segments-affisix-%.txt: $(DATA_SOURCE)
	ulimit -t unlimited && nice -n 19 ../../../svn/affisix-stable/affisix-2.1.99/src/affisix -i $(TRAIN_CORPUS-$*) -o "$@" -r suffix -c '&(>(fentr(i);1.0);>(bentr(i);2.0))' -s 'fentr(i);bentr(i)' -f '%s - f: %1 b: %2'


morfessor-model-%.bin: $(DATA_SOURCE)
	ulimit -t unlimited && nice -n 19 morfessor -t $(TRAIN_CORPUS-$*) -s "$@" -x "lexicon-$*.txt" --logfile "morfessor-train-$*-log.txt"

segments-morfessor-%.txt: morfessor-vocab-%.txt $(DATA_SOURCE)
	zcat $(TRAIN_CORPUS-$*) | "${PYTHON}" ./reconstruct-sentences.py "$<" > "$@"

wmt17-nmt-training-task-package.tgz:
	wget -O "$@" 'http://data.statmt.org/wmt17/nmt-training-task/wmt17-nmt-training-task-package.tgz'

wmt17-nmt-training-task-package/README: wmt17-nmt-training-task-package.tgz
	tar -xvf "$<"

wmt17-nmt-training-task-package/train.cs.iso8859-2.txt: wmt17-nmt-training-task-package/train.cs.gz
	zcat "$<" | sed -e 's/ /\n/g' | grep -v '[^[:alpha:]]' | iconv -f UTF-8 -t ISO-8859-2//TRANSLIT | LC_CTYPE=C grep -v "[?^']" > "$@"


bpe/learn_bpe.py:
	git clone 'https://github.com/rsennrich/subword-nmt.git' 'bpe'

bpe-vocab-%-cs.txt: bpe/learn_bpe.py $(DATA_SOURCE)
	zcat $(TRAIN_CORPUS-cs) | ./bpe/learn_bpe.py -s "$*" > "$@"
segments-bpe-%-cs.txt: bpe-vocab-%-cs.txt bpe/learn_bpe.py $(DATA_SOURCE)
	zcat $(TRAIN_CORPUS-cs) | ./bpe/apply_bpe.py -c "$<" > "$@"

bpe-vocab-%-en.txt: bpe/learn_bpe.py $(DATA_SOURCE)
	zcat $(TRAIN_CORPUS-en) | ./bpe/learn_bpe.py -s "$*" > "$@"
segments-bpe-%-en.txt: bpe-vocab-%-en.txt bpe/learn_bpe.py $(DATA_SOURCE)
	zcat $(TRAIN_CORPUS-en) | ./bpe/apply_bpe.py -c "$<" > "$@"



czech-morfflex-pdt-161115.zip:
	wget -O "$@" 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1836/czech-morfflex-pdt-161115.zip?sequence=1&isAllowed=y'

czech-morfflex-pdt-161115/README: czech-morfflex-pdt-161115.zip
	unzip -DD "$<"

$(DERINET):
	wget -O - 'http://ufal.mff.cuni.cz/~zabokrtsky/derinet/$(@:.gz=)' |gzip -cv9 > "$@"


segments-derinet-cs.txt: $(DATA_SOURCE) $(DERINET)
	zcat $(TRAIN_CORPUS-cs) | "${PYTHON}" ./segment-by-derinet.py --from spl --to hmorph $(DERINET) > "$@"

segments-derinet-morphodita-cs.txt: $(DATA_SOURCE) $(DERINET) czech-morfflex-pdt-161115/README
	zcat $(TRAIN_CORPUS-cs) | "${PYTHON}" ./segment-by-derinet.py --from spl --to hmorph $(DERINET) -a "$(MORPHO_TAGGER)" > "$@"

segments-derinet-morfflex-cs.txt: $(DATA_SOURCE) $(DERINET) morfflex-filtered.tsv.xz
	zcat $(TRAIN_CORPUS-cs) | "${PYTHON}" ./segment-by-derinet.py --from spl --to hmorph $(DERINET) -m morfflex-filtered.tsv.xz > "$@"


stats-bpe-%.txt: segments-bpe-%.txt segmentation-statistics.py
	"${PYTHON}" ./segmentation-statistics.py -f hbpe < "$<" > "$@"
stats-morfessor-%.txt: segments-morfessor-%.txt segmentation-statistics.py
	"${PYTHON}" ./segmentation-statistics.py -f hmorph < "$<" > "$@"
stats-derinet-%.txt: segments-derinet-%.txt segmentation-statistics.py
	"${PYTHON}" ./segmentation-statistics.py -f hmorph < "$<" > "$@"
stats-corpus-%.txt: $(DATA_SOURCE)
	zcat $(TRAIN_CORPUS-$*) | ./segmentation-statistics.py -f spl > "$@"


gold-predicted-derinet-$(DATE).txt: $(GOLD_STANDARD_DATA) $(DERINET) segment-by-derinet.py
# 	sed -e 's/ //g' < "$<" | python3 -X tracemalloc ./segment-by-derinet.py -f spl -t hmorph "$(DERINET)" > "$@"
	sed -e 's/ //g' < "$<" | "${PYTHON}" ./segment-by-derinet.py -f spl -t hmorph --save "prob-tables-$(DATE)" "$(DERINET)" > "$@"

precision-recall-derinet-$(DATE).txt: gold-predicted-derinet-$(DATE).txt measure-precision-recall.py $(GOLD_STANDARD_DATA)
	echo -n 'Stats measured on $(GOLD_STANDARD_DATA) on ' > "$@"
	date >> "$@"
	"${PYTHON}" ./measure-precision-recall.py -f hmorph "$(GOLD_STANDARD_DATA)" < "$<" >> "$@"

gold-standard-data.txt:
	@echo -e 'Sorry, our evaluation dataset cannot be published due to licensing restrictions.\nYou will have to provide your own. We apologize for the inconvenience.'
	false


trigram-entropy-%.txt: segments-% lm.pl
	# This tool requires a vertical data format – single word per line.
	./lm.pl <(head -n 50000 "$<") <(tail -n +50001 "$<" | head -n 20000) <(tail -n 70000 "$<") > "$@"

%-count-histogram.png: %.tsv plot-affix-count-histogram.gp
	# TODO convert the gnuplot script to accept STDIN and paremetrize the output name.
	gnuplot -c plot-affix-count-histogram.gp "$<" > "$@"
	#gnuplot plot-affix-count-histogram.gp

%-coverage.png: %-coverage.tsv plot-affix-coverage.gp
	gnuplot -c plot-affix-coverage.gp "$<" > "$@"

pref-%.tsv: morfflex-filtered.tsv.xz list-morfflex-affixes.py
	xzcat -vv "$<" | grep "	$*.*	" | "${PYTHON}" ./list-morfflex-affixes.py "pref-$*.tsv" "suf-$*.tsv"

%-coverage.tsv: %.tsv
	# I use "read line && cut" instead of "read suffix count", because suffix
	#  may be empty and then the Bourne shell ignores the first TAB on the line,
	# reads the count as the suffix and leaves count empty.
	total=`cut -f 2 "$<" | paste -sd+ | bc`; \
	partial='0'; \
	while IFS= read -r line; do \
		count=`echo "$${line}" | cut -f 2`; \
		partial=`echo "$${partial}" + "$${count}" | bc`; \
		echo 100 \
		     \* \
		     "$${partial}" \
		     / \
		     "$${total}" | bc -l; \
	done < "$<" | paste "$<" - > "$@"

techlemmas-from-derinet.txt: $(DERINET)
	zcat "$<" | cut -f3 > "$@"

morfflex-filtered.tsv.xz: $(MORFFLEX) techlemmas-from-derinet.txt
	# Filter only those lines from MorfFlex that are in DeriNet and have [ADNV] POS.
	# We want to do this because of lemma-form pairs such as A → ampér.
	# Also, abbreviations, numerals and other weirdness.
	xzcat -vv "$<" | grep -E '	[ADNV].{14}	' | "${PYTHON}" ./filter-morfflex-lemmas.py techlemmas-from-derinet.txt | xz -c - > "$@"

clean:
# 	rm -rf wmt17-nmt-training-task-package wmt17-nmt-training-task-package.tgz
# 	rm -f $(DERINET)
# 	rm -rf czech-morfflex-pdt-161115/
	rm -f morfessor-model-*.bin segments-*.txt lexicon-*.txt morfessor-*-log.txt
	rm -f bpe-vocab-*.txt morfessor-vocab-*.txt
	rm -f stats-*.txt
	rm -f gold-predicted-*.txt precision-recall-*.txt
