.PHONY: all download clean

# Never ever remove any intermediate files.
# .PRECIOUS:
.SECONDARY:

MORFESSOR_MODEL::=morfessor-model

DATA_SOURCE::=wmt17-nmt-training-task-package/README wmt17-nmt-training-task-package/train.cs.iso8859-2.txt

TRAIN_CORPUS-cs::=wmt17-nmt-training-task-package/train.cs.gz
TRAIN_CORPUS-cs-iso::=wmt17-nmt-training-task-package/train.cs.iso8859-2.txt
TRAIN_CORPUS-en::=wmt17-nmt-training-task-package/train.en.gz

MORPHO_TAGGER::=czech-morfflex-pdt-161115/czech-morfflex-161115-pos_only.dict

all: stats-morfessor-cs.txt stats-morfessor-en.txt # stats-affisix-cs-iso.txt
all: stats-derinet-morphodita-cs.txt
all: stats-bpe-1000-cs.txt stats-bpe-30000-cs.txt stats-bpe-50000-cs.txt stats-bpe-85000-cs.txt
all: stats-bpe-1000-en.txt stats-bpe-30000-en.txt stats-bpe-50000-en.txt stats-bpe-85000-en.txt
all: stats-corpus-cs.txt stats-corpus-en.txt

download: czech-morfflex-pdt-161115/README derinet-1-4.tsv.gz

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
	zcat $(TRAIN_CORPUS-$*) | ./reconstruct-sentences.py "$<" > "$@"

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

derinet-1-4.tsv.gz:
	wget -O "$@" 'https://www.jonys.cz/derinet/search/derinet-1-4.tsv.gz'


segments-derinet-cs.txt: $(DATA_SOURCE) derinet-1-4.tsv.gz
	zcat $(TRAIN_CORPUS-cs) | sed -e 's/$$/\n/; s/\s\+/\n/g' | ./segment-by-derinet.py derinet-1-4.tsv.gz > "$@"

segments-derinet-morphodita-cs.txt: $(DATA_SOURCE) derinet-1-4.tsv.gz czech-morfflex-pdt-161115/README
	zcat $(TRAIN_CORPUS-cs) | sed -e 's/$$/\n/; s/\s\+/\n/g' | ./segment-by-derinet.py derinet-1-4.tsv.gz -a "$(MORPHO_TAGGER)" > "$@"

segments-derinet-morfflex-cs.txt: $(DATA_SOURCE) derinet-1-4.tsv.gz morfflex-cz.2016-11-15.utf8.lemmaID_suff-tag-form.tab.csv.xz
	zcat $(TRAIN_CORPUS-cs) | sed -e 's/$$/\n/; s/\s\+/\n/g' | ./segment-by-derinet.py derinet-1-4.tsv.gz -m morfflex-cz.2016-11-15.utf8.lemmaID_suff-tag-form.tab.csv.xz > "$@"


stats-bpe-%.txt: segments-bpe-%.txt segmentation-statistics.py
	./segmentation-statistics.py -f hbpe < "$<" > "$@"
stats-morfessor-%.txt: segments-morfessor-%.txt segmentation-statistics.py
	./segmentation-statistics.py -f hmorph < "$<" > "$@"
stats-derinet-%.txt: segments-derinet-%.txt segmentation-statistics.py
	./segmentation-statistics.py -f vbpe < "$<" > "$@"
stats-corpus-%.txt: $(DATA_SOURCE)
	zcat $(TRAIN_CORPUS-$*) | ./segmentation-statistics.py -f spl > "$@"



clean:
# 	rm -rf wmt17-nmt-training-task-package wmt17-nmt-training-task-package.tgz
# 	rm -f derinet-1-4.tsv.gz
# 	rm -rf czech-morfflex-pdt-161115/
	rm -f morfessor-model-*.bin segments-*.txt lexicon-*.txt morfessor-*-log.txt
	rm -f bpe-vocab-*.txt morfessor-vocab-*.txt
	rm -f stats-*.txt
