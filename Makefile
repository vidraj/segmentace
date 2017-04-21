.PHONY: all clean

# Never ever remove any intermediate files.
.PRECIOUS:
.SECONDARY:

MORFESSOR_MODEL::=morfessor-model

DATA_SOURCE::=wmt17-nmt-training-task-package/README

TRAIN_CORPUS-cs::=wmt17-nmt-training-task-package/train.cs.gz
TRAIN_CORPUS-en::=wmt17-nmt-training-task-package/train.en.gz

all: output-segmented-cs.txt output-segmented-en.txt

# Run Morfessor on the input, save a mapping from words to segments.
segments-%.txt: $(MORFESSOR_MODEL)-%.bin $(DATA_SOURCE)
	# TODO strip all punctuation and uninteresting words from the input.
	ulimit -t unlimited && gunzip -ckv $(TRAIN_CORPUS-$*) | sed -e 's/ /\n/g' | sort -u | nice -n 19 morfessor -l "$<" -T - --logfile "morfessor-predict-$*-log.txt" > "$@"

morfessor-model-%.bin: $(DATA_SOURCE)
	ulimit -t unlimited && nice -n 19 morfessor -t $(TRAIN_CORPUS-$*) -s "$@" -x "lexicon-$*.txt" --logfile "morfessor-train-$*-log.txt"

output-segmented-%.txt: segments-%.txt $(DATA_SOURCE)
	gunzip -ckv $(TRAIN_CORPUS-$*) | ./reconstruct-sentences.py "$<" > "$@"

wmt17-nmt-training-task-package.tgz:
	wget -O "$@" 'http://data.statmt.org/wmt17/nmt-training-task/wmt17-nmt-training-task-package.tgz'

wmt17-nmt-training-task-package/README: wmt17-nmt-training-task-package.tgz
	tar -xvf "$<"

clean:
# 	rm -rf wmt17-nmt-training-task-package wmt17-nmt-training-task-package.tgz
	rm -f morfessor-model-*.bin segments-*.txt output-segmented-*.txt lexicon-*.txt morfessor-*-log.txt

