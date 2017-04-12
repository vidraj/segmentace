.PHONY: all clean

MORFESSOR_MODEL::=morfessor-model-derivonly.bin

all: output-segmented.txt

# Run Morfessor on the input, save a mapping from words to segments.
segments.txt: input-sentences.txt $(MORFESSOR_MODEL)
	# TODO strip all punctuation and uninteresting words from the input.
	sed -e 's/ /\n/g' "$<" | sort -u | morfessor -l $(MORFESSOR_MODEL) -T - --logfile morfessor-predict-log.txt > "$@"

morfessor-model.bin: morfessor-train.txt
	morfessor -t "$<" -s "$@" -x lexicon.txt --logfile morfessor-train-log.txt

output-segmented.txt: input-sentences.txt segments.txt
	./reconstruct-sentences.py segments.txt < "$<" > "$@"
