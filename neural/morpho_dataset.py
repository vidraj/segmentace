import numpy as np

class MorphoDataset:
    """Class capable of loading morphological datasets in vertical format.

    The dataset is assumed to be composed of factors (by default FORMS, LEMMAS and TAGS),
    each an object containing the following fields:
    - strings: Strings of the original words.
    - word_ids: Word ids of the original words (uses <unk> and <pad>).
    - words_map: String -> word_id map.
    - words: Word_id -> string list.
    - alphabet_map: Character -> char_id map.
    - alphabet: Char_id -> character list.
    - charseq_ids: Character_sequence ids of the original words.
    - charseqs_map: String -> character_sequence_id map.
    - charseqs: Character_sequence_id -> [characters], where character is an index
        to the dataset alphabet.
    """

    PLEMMAS = 0
    LEMMAS = 1
    SEGMENTS = 2
    FACTORS = 3

    class _Factor:
        def __init__(self, train=None):
            self.alphabet_map = train.alphabet_map if train else {'<pad>': 0, '<unk>': 1, '<bow>': 2, '<eow>': 3}
            self.alphabet = train.alphabet if train else ['<pad>', '<unk>', '<bow>', '<eow>']
            self.charseqs_map = {'<pad>': 0}
            self.charseqs = [[self.alphabet_map['<pad>']]]
            self.charseq_ids = []
            self.strings = []

    def __init__(self, filename, train=None, shuffle_batches=True, max_examples=None, add_bow_eow=False):
        """Load dataset from file in vertical format.

        Arguments:
        add_bow_eow: Whether to add BOW/EOW characters to the word characters.
        train: If given, the words and alphabets are reused from the training data.
        """

        # Create word maps
        self._factors = []
        for f in range(self.FACTORS):
            self._factors.append(self._Factor(train._factors[f] if train else None))

        # Load the sentences
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                if max_examples is not None and len(self._factors[self.FORMS].word_ids) >= max_examples:
                    break

                line = line.rstrip("\r\n")
                columns = line.split("\t")
                assert len(columns) == self.FACTORS, "Malformed file '{}' on line '{}': got {} columns, expected {}.".format(filename, line, len(columns), self.FACTORS)

                for f in range(self.FACTORS):
                    factor = self._factors[f]
                    word = columns[f] #if f < len(columns) else '<pad>'
                    factor.strings.append(word)

                    # Character-level information
                    if word not in factor.charseqs_map:
                        factor.charseqs_map[word] = len(factor.charseqs)
                        factor.charseqs.append([])
                        if add_bow_eow:
                            factor.charseqs[-1].append(factor.alphabet_map['<bow>'])
                        for c in word:
                            if c not in factor.alphabet_map:
                                if train:
                                    c = '<unk>'
                                else:
                                    factor.alphabet_map[c] = len(factor.alphabet)
                                    factor.alphabet.append(c)
                            factor.charseqs[-1].append(factor.alphabet_map[c])
                        if add_bow_eow:
                            factor.charseqs[-1].append(factor.alphabet_map['<eow>'])
                    factor.charseq_ids.append(factor.charseqs_map[word])

        # Compute charseq lengths
        charseqs = len(self._factors[self.LEMMAS].charseqs)
        self._charseq_lens = np.zeros([charseqs], np.int32)
        for i in range(charseqs):
            self._charseq_lens[i] = len(self._factors[self.LEMMAS].charseqs[i])

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._factors[self.LEMMAS].charseq_ids)) if self._shuffle_batches else np.arange(len(self._factors[self.LEMMAS].charseq_ids))

    @property
    def charseq_lens(self):
        return self._charseq_lens

    @property
    def factors(self):
        """Return the factors of the dataset.

        The result is an array of factors, each an object containing:
        strings: Strings of the original words.
        word_ids: Word ids of the original words (uses <unk> and <pad>).
        words_map: String -> word_id map.
        words: Word_id -> string list.
        alphabet_map: Character -> char_id map.
        alphabet: Char_id -> character list.
        charseq_ids: Character_sequence ids of the original words.
        charseqs_map: String -> character_sequence_id map.
        charseqs: Character_sequence_id -> [characters], where character is an index
          to the dataset alphabet.
        """

        return self._factors

    def next_batch(self, batch_size):
        """Return the next batch.

        Returns: (batch_charseq_ids, batch_charseqs, batch_charseq_lens)
        batch_charseq_ids: For each factor, batch of charseq_ids
          (with the ids pointing into batch_charseqs).
        batch_charseqs: For each factor, all unique charseqs in the batch,
          indexable by batch_charseq_ids. Contains indices of characters from self.alphabet.
        batch_charseq_lens: For each factor, length of charseqs in batch_charseqs.
        """

        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]
        return self._next_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._factors[self.LEMMAS].charseq_ids)) if self._shuffle_batches else np.arange(len(self._factors[self.LEMMAS].charseq_ids))
            return True
        else:
            return False

    def _next_batch(self, batch_perm):
        batch_size = len(batch_perm)

        # Character-level data
        batch_charseq_ids, batch_charseqs, batch_charseq_lens = [], [], []
        for factor in self._factors:
            batch_charseq_ids.append(np.zeros([batch_size], np.int32))
            charseqs_map = {}
            charseqs = []
            charseq_lens = []
            for i in range(batch_size):
                charseq_id = factor.charseq_ids[batch_perm[i]]
                if charseq_id not in charseqs_map:
                    charseqs_map[charseq_id] = len(charseqs)
                    charseqs.append(factor.charseqs[charseq_id])
                batch_charseq_ids[-1][i] = charseqs_map[charseq_id]

            batch_charseq_lens.append(np.array([len(charseq) for charseq in charseqs], np.int32))
            batch_charseqs.append(np.zeros([len(charseqs), np.max(batch_charseq_lens[-1])], np.int32))
            for i in range(len(charseqs)):
                batch_charseqs[-1][i, 0:len(charseqs[i])] = charseqs[i]

        return batch_charseq_ids, batch_charseqs, batch_charseq_lens
