.PHONY: all download clean

# Never ever remove any intermediate files.
# .PRECIOUS:
.SECONDARY:

MORFESSOR_MODEL::=morfessor-model

DATA_SOURCE::=wmt17-nmt-training-task-package/README

TRAIN_CORPUS-cs::=wmt17-nmt-training-task-package/train.cs.gz
TRAIN_CORPUS-en::=wmt17-nmt-training-task-package/train.en.gz

# all: output-segmented-morfessor-cs.txt output-segmented-morfessor-en.txt # output-segmented-affisix-cs-iso.txt
all: segments-derinet-cs.txt

download: czech-morfflex-pdt-161115/README derinet-1-4.tsv.gz

# Run Morfessor on the input, save a mapping from words to segments.
segments-morfessor-%.txt: $(MORFESSOR_MODEL)-%.bin $(DATA_SOURCE)
	# TODO strip all punctuation and uninteresting words from the input.
	ulimit -t unlimited && gunzip -ckv $(TRAIN_CORPUS-$*) | sed -e 's/ /\n/g' | sort -u | nice -n 19 morfessor -l "$<" -T - --logfile "morfessor-predict-$*-log.txt" > "$@"

# TODO $(TRAIN_CORPUS) blows when DATA_SOURCE wasn't made yet.
segments-affisix-%.txt: $(DATA_SOURCE)
	ulimit -t unlimited && nice -n 19 ../../../svn/affisix-stable/affisix-2.1.99/src/affisix -i $(TRAIN_CORPUS-$*) -o "$@" -r suffix -c '&(>(fentr(i);1.0);>(bentr(i);2.0))' -s 'fentr(i);bentr(i)' -f '%s - f: %1 b: %2'


morfessor-model-%.bin: $(DATA_SOURCE)
	ulimit -t unlimited && nice -n 19 morfessor -t $(TRAIN_CORPUS-$*) -s "$@" -x "lexicon-$*.txt" --logfile "morfessor-train-$*-log.txt"

output-segmented-%.txt: segments-%.txt $(DATA_SOURCE)
	gunzip -ckv $(TRAIN_CORPUS-$*) | ./reconstruct-sentences.py "$<" > "$@"

wmt17-nmt-training-task-package.tgz:
	wget -O "$@" 'http://data.statmt.org/wmt17/nmt-training-task/wmt17-nmt-training-task-package.tgz'

wmt17-nmt-training-task-package/README: wmt17-nmt-training-task-package.tgz
	tar -xvf "$<"


czech-morfflex-pdt-161115.zip:
	wget -O "$@" 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1836/czech-morfflex-pdt-161115.zip?sequence=1&isAllowed=y'

czech-morfflex-pdt-161115/README: czech-morfflex-pdt-161115.zip
	unzip -DD "$<"

derinet-1-4.tsv.gz:
	wget -O "$@" 'https://www.jonys.cz/derinet/search/derinet-1-4.tsv.gz'

segments-derinet-cs.txt: $(DATA_SOURCE) derinet-1-4.tsv.gz czech-morfflex-pdt-161115/README
	 zcat $(TRAIN_CORPUS-cs) | sed -e 's/$$/\n/; s/\s\+/\n/g' | ./segment-by-derinet.py derinet-1-4.tsv.gz -a czech-morfflex-pdt-161115/czech-morfflex-161115-pos_only.dict > "$@"

clean:
# 	rm -rf wmt17-nmt-training-task-package wmt17-nmt-training-task-package.tgz
	rm -f morfessor-model-*.bin segments-*.txt output-segmented-*.txt lexicon-*.txt morfessor-*-log.txt

