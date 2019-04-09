.PHONY: quick-test
SHELL::=/bin/bash

# SRC_DB::=../derinet-test.tsv
# SRC_DB_ANNOT::=../quick-test.out
SRC_DB::=../derinet-1-5-1.tsv
SRC_DB_ANNOT::=inprogress-predict-prob-tables-2018-07-04-18-18-15-1-pretrain.p.txt
GOLD_STANDARD_DATA::=../gold-standard-data.txt
PYTHON_TFENV_ACTIVATE::=$(HOME)/škola/deep-learning/npfl114/bin/activate

quick-test: segmenter.py train.tsv dev.tsv test.tsv
	source "$(PYTHON_TFENV_ACTIVATE)" && ./segmenter.py --batch_size=500 --epochs=4 --threads=1 --char_dim=64 --rnn_dim=128 --dropout=0.5 --learning_rate=0.001 --encoder_layers=2 --decoder_layers=1 --segmenter_layers=2 train.tsv dev.tsv test.tsv

train.tsv: corpus.tsv test.tsv
	# Take all lines from corpus.tsv that don't occur in the test set, beginning with line 80001.
	cut -f 2 test.tsv | grep -Fvwf- "$<" | tail -n +80001 > "$@"

dev.tsv: corpus.tsv test.tsv
	# Take the first 80000 lines from corpus.tsv that don't match the test set.
	cut -f 2 test.tsv | grep -Fvwf- "$<" | head -n 80000 > "$@"

test.tsv: $(GOLD_STANDARD_DATA)
	sed -e 's/ //g' < "$<" | paste <(yes _ | head -n `wc -l < "$<"`) - <(yes _ | head -n `wc -l < "$<"`) > "$@"

corpus.tsv: $(SRC_DB) $(SRC_DB_ANNOT)
	join -t '	' -1 1 -2 2 <(cut -f1,2 "$<" | sort -t '	' -k 1,1) <(cut -f2,5 "$<" | paste - <(sed -e 's/[^ ]/C/g; s/ C/B/g; s/^C/B/' < $(SRC_DB_ANNOT)) | sort -t '	' -k 2b,2) | cut -f 2-4 | ./shuf-seeded.sh 42 > "$@"
