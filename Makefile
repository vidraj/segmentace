.PHONY: all clean
# Never ever remove any intermediate files.
.PRECIOUS:

MORFESSOR_MODEL::=morfessor-model

all: output-segmented-cs.txt output-segmented-en.txt

# Run Morfessor on the input, save a mapping from words to segments.
segments-%.txt: $(MORFESSOR_MODEL)-%.bin wmt17-nmt-training-task-package/README
	# TODO strip all punctuation and uninteresting words from the input.
	gunzip -ckv "wmt17-nmt-training-task-package/train.$*.gz" | sed -e 's/ /\n/g' | sort -u | morfessor -l "$<" -T - --logfile "morfessor-predict-$*-log.txt" > "$@"

morfessor-model-%.bin: wmt17-nmt-training-task-package/README
	morfessor -t "wmt17-nmt-training-task-package/train.$*.gz" -s "$@" -x "lexicon-$*.txt" --logfile "morfessor-train-$*-log.txt"

output-segmented-%.txt: segments-%.txt wmt17-nmt-training-task-package/README
	gunzip -ckv "wmt17-nmt-training-task-package/train.$*.gz" | ./reconstruct-sentences.py "$<" > "$@"

wmt17-nmt-training-task-package.tgz:
	wget -O "$@" 'http://data.statmt.org/wmt17/nmt-training-task/wmt17-nmt-training-task-package.tgz'

wmt17-nmt-training-task-package/README: wmt17-nmt-training-task-package.tgz
	tar -xvf "$<"

clean:
# 	rm -rf wmt17-nmt-training-task-package wmt17-nmt-training-task-package.tgz
	rm -f morfessor-model-*.bin segments-*.txt output-segmented-*.txt lexicon-*.txt morfessor-*-log.txt

