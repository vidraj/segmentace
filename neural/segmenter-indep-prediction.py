#!/usr/bin/env python3
import sys
import re
import math
import random
import numpy as np
import tensorflow as tf

import morpho_dataset

def indicators_to_segments(lemma, indicators, quiet=False):
    """Convert a lemma-indicators pair as obtained from the neural network to a list of morphs. Return that list.
    If quiet, permit malformed output instead of raising an Exception on malformed input."""
    indicators = indicators.replace("<pad>", "X").replace("<unk>", "X").replace("<bow>", "X").replace("<eow>", "X")
    if not quiet:
        assert len(lemma) == len(indicators), "The segmentation indicators '{}' don't fit the lemma '{}'".format(indicators, lemma)

    if indicators[0] == "B":
        segments = [[lemma[0]]]
    else:
        if quiet:
            segments = [["<ERR>"], []]
        else:
            raise Exception("The segmentation indicator doesn't start with B, but with '{}' in '{}' '{}'".format(indicators[0], lemma, indicators))

    for char, ind in zip(lemma[1:], indicators[1:]):
        if ind == "C":
            # Continuation of a segment.
            segments[-1].append(char)
        elif ind == "B":
            # Start of a new segment.
            segments.append([char])
        else:
            if quiet:
                segments.append(["<ERR>"])
                segments.append([])
            else:
                raise Exception("Illegal segmentation indicator '{}' found in '{}' '{}'".format(ind, lemma, indicators))

    return ["".join(segment) for segment in segments]

def create_encoder_layers(layer_count, layer_input, input_lengths, rnn_dim, char_dim, dropout_keep_prob):
    # TODO transfer states from one layer to another.

    # Using a GRU with dimension args.rnn_dim, process the embedded self.source_seqs
    # using bidirectional RNN. Store the summed fwd and bwd outputs in `source_encoded`
    # and the summed fwd and bwd states into `source_states`.
    encoder_rnn_cell_fwd = tf.nn.rnn_cell.GRUCell(num_units=rnn_dim)
    encoder_rnn_cell_bwd = tf.nn.rnn_cell.GRUCell(num_units=rnn_dim)

    if args.dropout is not None:
        encoder_rnn_cell_fwd = tf.nn.rnn_cell.DropoutWrapper(encoder_rnn_cell_fwd, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob, state_keep_prob=1.0, variational_recurrent=False, input_size=char_dim, dtype=tf.float32)
        encoder_rnn_cell_bwd = tf.nn.rnn_cell.DropoutWrapper(encoder_rnn_cell_bwd, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob, state_keep_prob=1.0, variational_recurrent=False, input_size=char_dim, dtype=tf.float32)
        #encoder_rnn_cell_fwd = tf.nn.rnn_cell.DropoutWrapper(encoder_rnn_cell_fwd, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob, state_keep_prob=dropout_keep_prob, variational_recurrent=True, input_size=char_dim, dtype=tf.float32)
        #encoder_rnn_cell_bwd = tf.nn.rnn_cell.DropoutWrapper(encoder_rnn_cell_bwd, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob, state_keep_prob=dropout_keep_prob, variational_recurrent=True, input_size=char_dim, dtype=tf.float32)

    (source_encoded_fwd, source_encoded_bwd), (source_states_fwd, source_states_bwd) = tf.nn.bidirectional_dynamic_rnn(encoder_rnn_cell_fwd, encoder_rnn_cell_bwd, layer_input, sequence_length=input_lengths, dtype=tf.float32, swap_memory=True, scope="rnn_encoder_0")


    output = source_encoded_fwd + source_encoded_bwd
    states = source_states_fwd + source_states_bwd

    for layer_id in range(1, layer_count):
        encoder_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_dim)

        if args.dropout is not None:
            encoder_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_rnn_cell, input_keep_prob=1.0, output_keep_prob=dropout_keep_prob, state_keep_prob=1.0, variational_recurrent=False, input_size=rnn_dim, dtype=tf.float32)

        encoder_rnn_cell = tf.nn.rnn_cell.ResidualWrapper(encoder_rnn_cell)

        output, states = tf.nn.dynamic_rnn(encoder_rnn_cell, output, sequence_length=input_lengths, dtype=tf.float32, swap_memory=True, scope="rnn_encoder_{}".format(layer_id))

    return output, states



class Decoder:
    def __init__(self, layer_count, rnn_dim, char_dim, dropout_keep_prob):
        decoder_rnn_cells = []

        # Generate a decoder GRU with dimension args.rnn_dim.
        decoder_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_dim)

        if args.dropout is not None:
            decoder_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_rnn_cell, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob, state_keep_prob=1.0, variational_recurrent=False, input_size=(rnn_dim + char_dim), dtype=tf.float32)
            # The input_size is |source_encoded| + |target_embeddings|.
            #decoder_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_rnn_cell, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob, state_keep_prob=dropout_keep_prob, variational_recurrent=True, input_size=(rnn_dim + char_dim), dtype=tf.float32)

        decoder_rnn_cells.append(decoder_rnn_cell)


        for layer_id in range(1, layer_count):
            decoder_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_dim)

            if args.dropout is not None:
                decoder_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_rnn_cell, input_keep_prob=1.0, output_keep_prob=dropout_keep_prob, state_keep_prob=1.0, variational_recurrent=False, input_size=rnn_dim, dtype=tf.float32)
                # The input_size is |source_encoded| + |target_embeddings|.
                #decoder_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_rnn_cell, input_keep_prob=1.0, output_keep_prob=dropout_keep_prob, state_keep_prob=dropout_keep_prob, variational_recurrent=True, input_size=rnn_dim, dtype=tf.float32)

            decoder_rnn_cell = tf.nn.rnn_cell.ResidualWrapper(decoder_rnn_cell)
            decoder_rnn_cells.append(decoder_rnn_cell)

        self.decoder_rnn_cells = decoder_rnn_cells

    def __call__(self, layer_input):
        new_states = []
        for layer_id, (decoder_rnn_cell, prev_state) in enumerate(zip(self.decoder_rnn_cells, self.states)):
            layer_input, new_state = decoder_rnn_cell(layer_input, prev_state, scope="rnn_decoder_{}".format(layer_id))
            new_states.append(new_state)

        self.states = new_states
        return layer_input, new_state

    def set_initial_state(self, state, batch_size):
        self.states = [state]
        for cell in self.decoder_rnn_cells[1:]:
            # TODO initialize the states for all other cells in self.decoder_rnn_cells.
            self.states.append(cell.zero_state(batch_size, dtype=tf.float32))
        return state




class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, train_size, source_chars, target_chars, target_seg_chars, bow, eow):
        with self.session.graph.as_default():
            # Inputs
            self.source_ids = tf.placeholder(tf.int32, [None], name="source_ids")
            self.source_seqs = tf.placeholder(tf.int32, [None, None], name="source_seqs")
            self.source_seq_lens = tf.placeholder(tf.int32, [None], name="source_seq_lens")
            self.target_ids = tf.placeholder(tf.int32, [None], name="target_ids")
            self.target_seqs = tf.placeholder(tf.int32, [None, None], name="target_seqs")
            self.target_seq_lens = tf.placeholder(tf.int32, [None], name="target_seq_lens")
            self.target_seg_ids = tf.placeholder(tf.int32, [None], name="target_seg_ids")
            self.target_seg_seqs = tf.placeholder(tf.int32, [None, None], name="target_seg_seqs")

            # TODO: Training. The rest of the code assumes that
            # - when training the decoder, the output layer with logis for each generated
            #   character is in `output_layer` and the corresponding predictions are in
            #   `self.predicted_lemmas`.
            # - the `target_ids` contains the gold generated characters
            # - the `target_lens` contains number of valid characters for each lemma
            # - when running decoder inference, the predictions are in `self.predictions`
            #   and their lengths in `self.prediction_lens`.

            self.dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name="dropout_keep_prob")
            self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
            #self.gold_predict_ratio = tf.placeholder(tf.float32, shape=(), name="gold_predict_mix")


            global_step = tf.train.create_global_step() # Moved from below.
            #gold_predict_ratio_decay_rate = pow(0.01/1.0, 1/(args.epochs - 1))
            gold_predict_ratio_batches_per_epoch = train_size // args.batch_size
            gold_predict_ratio_total_batches = gold_predict_ratio_batches_per_epoch * args.epochs
            gold_predict_ratio_zero = gold_predict_ratio_total_batches / 2
            gold_predict_ratio_normalized = tf.cast(global_step, tf.float32) / tf.constant(gold_predict_ratio_total_batches, dtype=tf.float32, shape=())
            gold_predict_ratio = 1 - tf.nn.sigmoid((gold_predict_ratio_normalized - 0.5) * 10.0)
            #gold_predict_ratio = 1 - tf.nn.sigmoid(tf.constant(int(gold_predict_ratio_zero), dtype=tf.float32, shape=()) - tf.cast(global_step, tf.float32))
            #gold_predict_ratio = 1 - 1 / (1 + math.exp(gold_predict_ratio_zero - global_step))

            # Append EOW after target_seqs
            target_seqs = tf.reverse_sequence(self.target_seqs, self.target_seq_lens, 1)
            target_seqs = tf.pad(target_seqs, [[0, 0], [1, 0]], constant_values=eow)
            target_seq_lens = self.target_seq_lens + 1
            target_seqs = tf.reverse_sequence(target_seqs, target_seq_lens, 1)

            # Encoder
            # Generate source embeddings for source chars, of shape [source_chars, args.char_dim].
            source_embeddings = tf.get_variable("src_char_emb", [source_chars, args.char_dim])

            # Embed the self.source_seqs using the source embeddings.
            source_embedded = tf.nn.embedding_lookup(source_embeddings, self.source_seqs)

            source_encoded, source_states = create_encoder_layers(args.encoder_layers, source_embedded, self.source_seq_lens, args.rnn_dim, args.char_dim, self.dropout_keep_prob)
            source_states = tf.layers.dropout(source_states, rate=self.dropout_keep_prob, training=self.is_training)




            # Index the unique words using self.source_ids and self.target_ids.
            source_encoded = tf.nn.embedding_lookup(source_encoded, self.source_ids)
            # Use the word embeddings. TODO maybe find a better way?
            source_states = tf.nn.embedding_lookup(source_states, self.source_ids)
            source_lens = tf.nn.embedding_lookup(self.source_seq_lens, self.source_ids)

            target_seqs = tf.nn.embedding_lookup(target_seqs, self.target_ids)
            target_lens = tf.nn.embedding_lookup(target_seq_lens, self.target_ids)

            seg_seqs  = tf.nn.embedding_lookup(self.target_seg_seqs, self.target_seg_ids)
            seg_lens = source_lens



            # Decoder
            # Generate target embeddings for target chars, of shape [target_chars, args.char_dim].
            target_embeddings = tf.get_variable("tgt_char_emb", [target_chars, args.char_dim])

            # Embed the target_seqs using the target embeddings.
            target_embedded = tf.nn.embedding_lookup(target_embeddings, target_seqs)

            # Generate a decoder GRU with dimension args.rnn_dim.
            decoder = Decoder(args.decoder_layers, args.rnn_dim, args.char_dim, self.dropout_keep_prob)
            #decoder_rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=args.rnn_dim)
            #if args.dropout is not None:
                #decoder_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_rnn_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob, state_keep_prob=1.0, variational_recurrent=False)
                ## The input_size is |source_encoded| + |target_embeddings|.
                ##decoder_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_rnn_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob, state_keep_prob=self.dropout_keep_prob, variational_recurrent=True, input_size=(args.rnn_dim + args.char_dim), dtype=tf.float32)

            ##decoder_rnn_cell = tf.nn.rnn_cell.ResidualWrapper(decoder_rnn_cell)


            # Create a `decoder_layer` -- a fully connected layer with
            # target_chars neurons used in the decoder to classify into target characters.
            decoder_layer = tf.layers.Dense(target_chars, activation=None, name="decoder_layer")

            # Attention
            # Generate three fully connected layers without activations:
            # - `source_layer` with args.rnn_dim units
            # - `state_layer` with args.rnn_dim units
            # - `weight_layer` with 1 unit
            source_layer = tf.layers.Dense(args.rnn_dim, activation=None, name="source_layer")
            state_layer = tf.layers.Dense(args.rnn_dim, activation=None, name="state_layer")
            weight_layer = tf.layers.Dense(1, activation=None, name="weight_layer")

            # Project source_encoded using source_layer.
            att_source_encoded = source_layer(source_encoded)

            def with_attention(inputs, states):
                # Generate the attention

                # Change shape of states from [a, b] to [a, 1, b] and project it using state_layer.
                states = state_layer(tf.expand_dims(states, axis=1))

                # Sum the two above projections, apply tf.tanh and project the result using weight_layer.
                # The result has shape [x, y, 1].
                # The sum will automatically broadcast states (copy them several times and sum the resulting matrices), because of TensorFlow semantics.
                att_input = tf.tanh(att_source_encoded + states)
                att_weight_raw = weight_layer(att_input)

                # Apply tf.nn.softmax to the latest result, using axis corresponding to source characters.
                att_weight_normalized = tf.nn.softmax(att_weight_raw, axis=1)

                # Multiply the source_encoded by the latest result, and sum the results with respect
                # to the axis corresponding to source characters. This is the final attention.
                weighted_source_encoded = source_encoded * att_weight_normalized
                attention = tf.reduce_sum(weighted_source_encoded, axis=1)

                # Return concatenation of inputs and the computed attention.
                return tf.concat([inputs, attention], axis=1)
                #return inputs + attention

            # The DecoderTraining will be used during training. It will output logits for each
            # target character.
            class DecoderTraining(tf.contrib.seq2seq.Decoder):
                @property
                def batch_size(self): return tf.shape(source_states)[0] # Return size of the batch, using for example source_states size
                @property
                def output_dtype(self): return tf.float32 # Type for logits of target characters
                @property
                def output_size(self): return target_chars # Length of logits for every output

                def initialize(self, name=None):
                    finished = target_lens <= 0 # False if target_lens > 0, True otherwise
                    states = decoder.set_initial_state(source_states, self.batch_size) # Initial decoder state to use
                    inputs = with_attention(tf.nn.embedding_lookup(target_embeddings, tf.fill([self.batch_size], bow)), states) # Call with_attention on the embedded BOW characters of shape [self.batch_size].
                             # You can use tf.fill to generate BOWs of appropriate size.
                    return finished, inputs, states

                def step(self, time, inputs, states, name=None):
                    #outputs, states = decoder_rnn_cell(inputs, states, scope="rnn_decoder") # Run the decoder GRU cell using inputs and states.
                    outputs, states = decoder(inputs) # Run the decoder GRU cell using inputs and states.
                    outputs = decoder_layer(outputs) # Apply the decoder_layer on outputs.
                    predicted_output = tf.argmax(outputs, axis=-1, output_type=tf.int32)
                    ni = (1 - gold_predict_ratio) * tf.nn.embedding_lookup(target_embeddings, predicted_output) + gold_predict_ratio * tf.gather(target_embedded, time, axis=1)
                    next_input = with_attention(ni, states) # Next input is with_attention called on words with index `time` in target_embedded.
                    finished = target_lens <= time + 1 # False if target_lens > time + 1, True otherwise.

                    return outputs, states, next_input, finished
            output_layer, _, _ = tf.contrib.seq2seq.dynamic_decode(DecoderTraining(), swap_memory=True)
            self.predicted_lemmas_training = tf.argmax(output_layer, axis=2, output_type=tf.int32)

            # The DecoderPrediction will be used during prediction. It will
            # directly output the predicted target characters.
            class DecoderPrediction(tf.contrib.seq2seq.Decoder):
                @property
                def batch_size(self): return tf.shape(source_states)[0] # Return size of the batch, using for example source_states size
                @property
                def output_dtype(self): return tf.int32 # Type for predicted target characters
                @property
                def output_size(self): return 1 # Will return just one output

                def initialize(self, name=None):
                    finished = tf.fill([self.batch_size], False) # False of shape [self.batch_size].
                    states = decoder.set_initial_state(source_states, self.batch_size) # Initial decoder state to use.
                    inputs = with_attention(tf.nn.embedding_lookup(target_embeddings, tf.fill([self.batch_size], bow)), states) # Call with_attention on the embedded BOW characters of shape [self.batch_size].
                             # You can use tf.fill to generate BOWs of appropriate size.
                    return finished, inputs, states

                def step(self, time, inputs, states, name=None):
                    #outputs, states = decoder_rnn_cell(inputs, states, scope="rnn_decoder") # Run the decoder GRU cell using inputs and states.
                    outputs, states = decoder(inputs) # Run the decoder GRU cell using inputs and states.
                    outputs = decoder_layer(outputs) # Apply the decoder_layer on outputs.
                    outputs = tf.argmax(outputs, axis=-1, output_type=tf.int32) # Use tf.argmax to choose most probable class (supply parameter `output_type=tf.int32`).
                    next_input = with_attention(tf.nn.embedding_lookup(target_embeddings, outputs), states) # Embed `outputs` using target_embeddings and pass it to with_attention.
                    finished = tf.equal(outputs, eow) # True where outputs==eow, False otherwise
                    return outputs, states, next_input, finished
            self.predicted_lemmas, _, self.predicted_lemma_lengths = tf.contrib.seq2seq.dynamic_decode(
                DecoderPrediction(), maximum_iterations=tf.reduce_max(source_lens) + 10, swap_memory=True)


            segment_predict_layer = tf.layers.Dense(target_seg_chars, activation=None, name="segment_predict_layer")
            segment_output = segment_predict_layer(source_encoded)
            self.predicted_segments = tf.argmax(segment_output, axis=2, output_type=tf.int32)
            segment_weights = tf.sequence_mask(seg_lens, dtype=tf.float32)
            seg_loss = tf.losses.sparse_softmax_cross_entropy(seg_seqs, segment_output, weights=segment_weights)

            # Calculate word-level accuracy – accuracy is 1 iff the whole word is segmented correctly.
            #self.current_seg_accuracy, self.update_seg_accuracy = tf.metrics.accuracy(seg_seqs, self.predicted_segments, weights=segment_weights)
            accuracy = tf.reduce_all(tf.logical_or(
                tf.equal(seg_seqs, self.predicted_segments),
                tf.logical_not(tf.sequence_mask(seg_lens))), axis=1)
            self.current_seg_accuracy, self.update_seg_accuracy = tf.metrics.mean(accuracy)



            # Training
            weights = tf.sequence_mask(target_lens, dtype=tf.float32)
            loss = tf.losses.sparse_softmax_cross_entropy(target_seqs, output_layer, weights=weights) + seg_loss
            #global_step = tf.train.create_global_step() # Moved up in the source.
            if args.learning_rate_final is not None:
                decay_rate = pow(args.learning_rate_final/args.learning_rate, 1/(args.epochs - 1))
                batches_per_epoch = train_size // args.batch_size
                learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, batches_per_epoch, decay_rate, staircase=False)
            else:
                learning_rate = args.learning_rate
            self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name="training")
            #self.training = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name="training")

            # Summaries
            accuracy_training = tf.reduce_all(tf.logical_or(
                tf.equal(self.predicted_lemmas_training, target_seqs),
                tf.logical_not(tf.sequence_mask(target_lens))), axis=1)
            self.current_accuracy_training, self.update_accuracy_training = tf.metrics.mean(accuracy_training)

            minimum_length = tf.minimum(tf.shape(self.predicted_lemmas)[1], tf.shape(target_seqs)[1])
            accuracy = tf.logical_and(
                tf.equal(self.predicted_lemma_lengths, target_lens),
                tf.reduce_all(tf.logical_or(
                    tf.equal(self.predicted_lemmas[:, :minimum_length], target_seqs[:, :minimum_length]),
                    tf.logical_not(tf.sequence_mask(target_lens, maxlen=minimum_length))), axis=1))
            self.current_accuracy, self.update_accuracy = tf.metrics.mean(accuracy)

            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights) + tf.reduce_sum(segment_weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            with summary_writer.as_default():
                self.summary_flusher = tf.contrib.summary.flush()
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy_training),
                                           tf.contrib.summary.scalar("train/seg_accuracy", self.update_seg_accuracy),
                                           tf.contrib.summary.scalar("train/learning_rate", learning_rate),
                                           tf.contrib.summary.scalar("train/gold_predict_ratio", gold_predict_ratio)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy),
                                               tf.contrib.summary.scalar(dataset + "/seg_accuracy", self.current_seg_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size, dropout_strength):
        while not train.epoch_finished():
            charseq_ids, charseqs, charseq_lens = train.next_batch(batch_size)

            assert charseq_ids[train.LEMMAS].shape == charseq_ids[train.SEGMENTS].shape, "The lengths of source ids ({}) and target seq ids ({}) don't match.".format(charseq_ids[train.LEMMAS].shape, charseq_ids[train.SEGMENTS].shape)
            #assert charseq_ids[train.LEMMAS].shape == (batch_size,)
            #print("Seq shapes", charseqs[train.LEMMAS].shape, charseqs[train.SEGMENTS].shape)

            for i in range(len(charseq_ids[train.LEMMAS])):
                assert len(charseqs[train.LEMMAS][charseq_ids[train.LEMMAS][i]]) == len(charseqs[train.SEGMENTS][charseq_ids[train.SEGMENTS][i]]), "Seq lens don't match: '{}' and '{}'".format(charseqs[train.LEMMAS][charseq_ids[train.LEMMAS][i]], charseqs[train.SEGMENTS][charseq_ids[train.SEGMENTS][i]])

            self.session.run(self.reset_metrics)
            predictions, predicted_segments, _, _ = self.session.run(
                [self.predicted_lemmas_training, self.predicted_segments, self.training, self.summaries["train"]],
                {self.source_ids: charseq_ids[train.LEMMAS], self.target_ids: charseq_ids[train.PLEMMAS], self.target_seg_ids: charseq_ids[train.SEGMENTS],
                 self.source_seqs: charseqs[train.LEMMAS], self.target_seqs: charseqs[train.PLEMMAS], self.target_seg_seqs: charseqs[train.SEGMENTS],
                 self.source_seq_lens: charseq_lens[train.LEMMAS], self.target_seq_lens: charseq_lens[train.PLEMMAS],
                 self.is_training: True,
                 self.dropout_keep_prob: 1.0 - dropout_strength})

            ## Display a random word from the batch.

            # Uniformly choose a word index from the batch.
            nth_word = random.randrange(len(charseq_ids[train.LEMMAS]))
            # Get the IDs of the form at that position and its lemma.
            lemma_id = charseq_ids[train.LEMMAS][nth_word]
            plemma_id = charseq_ids[train.PLEMMAS][nth_word]
            seg_id = charseq_ids[train.SEGMENTS][nth_word]

            # Build the form and lemmas character-by-character.
            lemma, gold_plemma, system_plemma, gold_segments, system_segments = "", "", "", "", ""
            for i in range(charseq_lens[train.LEMMAS][lemma_id]):
                lemma += train.factors[train.LEMMAS].alphabet[charseqs[train.LEMMAS][lemma_id][i]]
            for i in range(charseq_lens[train.PLEMMAS][plemma_id]):
                gold_plemma += train.factors[train.PLEMMAS].alphabet[charseqs[train.PLEMMAS][plemma_id][i]]
                system_plemma += train.factors[train.PLEMMAS].alphabet[predictions[plemma_id][i]]
            for i in range(charseq_lens[train.SEGMENTS][seg_id]):
                gold_segments += train.factors[train.SEGMENTS].alphabet[charseqs[train.SEGMENTS][seg_id][i]]
                system_segments += train.factors[train.SEGMENTS].alphabet[predicted_segments[seg_id][i]]

            # Display the lemma and plemmas.
            print("Gold lemma: {} / {}, gold plemma: {}, predicted plemma: {} / {}".format(lemma, " ".join(indicators_to_segments(lemma, gold_segments, quiet=True)), gold_plemma, system_plemma, " ".join(indicators_to_segments(lemma, system_segments, quiet=True))), file=sys.stderr, flush=True)

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size)

            assert charseq_ids[dataset.LEMMAS].shape == charseq_ids[dataset.SEGMENTS].shape, "The lengths of source ids ({}) and target seq ids ({}) don't match.".format(charseq_ids[dataset.LEMMAS].shape, charseq_ids[dataset.SEGMENTS].shape)
            #assert charseq_ids[dataset.LEMMAS].shape == (batch_size,)
            #print("Seq shapes", charseqs[dataset.LEMMAS].shape, charseqs[dataset.SEGMENTS].shape)

            for i in range(len(charseq_ids[dataset.LEMMAS])):
                assert len(charseqs[dataset.LEMMAS][charseq_ids[dataset.LEMMAS][i]]) == len(charseqs[dataset.SEGMENTS][charseq_ids[dataset.SEGMENTS][i]]), "Seq lens don't match: '{}' and '{}'".format(charseqs[dataset.LEMMAS][charseq_ids[dataset.LEMMAS][i]], charseqs[dataset.SEGMENTS][charseq_ids[dataset.SEGMENTS][i]])

            self.session.run([self.update_accuracy, self.update_loss, self.update_seg_accuracy],
                             {self.source_ids: charseq_ids[dataset.LEMMAS], self.target_ids: charseq_ids[dataset.PLEMMAS], self.target_seg_ids: charseq_ids[dataset.SEGMENTS],
                              self.source_seqs: charseqs[dataset.LEMMAS], self.target_seqs: charseqs[dataset.PLEMMAS], self.target_seg_seqs: charseqs[dataset.SEGMENTS],
                              self.source_seq_lens: charseq_lens[dataset.LEMMAS], self.target_seq_lens: charseq_lens[dataset.PLEMMAS],
                              self.is_training: False,
                              self.dropout_keep_prob: 1.0})
        return self.session.run([self.current_accuracy, self.current_seg_accuracy, self.summaries[dataset_name]])[0:2]

    def predict(self, dataset, batch_size):
        lemmas = []
        segmentations = []
        while not dataset.epoch_finished():
            charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size)
            predicted_lemmas, predicted_lemma_lengths, predicted_segments = self.session.run(
                [self.predicted_lemmas, self.predicted_lemma_lengths, self.predicted_segments],
                {self.source_ids: charseq_ids[dataset.LEMMAS],
                 self.source_seqs: charseqs[dataset.LEMMAS], self.source_seq_lens: charseq_lens[dataset.LEMMAS],
                 self.is_training: False,
                 self.dropout_keep_prob: 1.0})

            for i, length in enumerate(predicted_lemma_lengths):
                lemmas.append("")
                for j in range(length - 1):
                    lemmas[-1] += dataset.factors[dataset.PLEMMAS].alphabet[predicted_lemmas[i][j]]

            for i in range(len(charseq_ids[dataset.LEMMAS])):
                lemma_id = charseq_ids[dataset.LEMMAS][i]
                segmentations.append("")
                for j in range(charseq_lens[dataset.LEMMAS][lemma_id]):
                    segmentations[-1] += dataset.factors[dataset.SEGMENTS].alphabet[predicted_segments[lemma_id][j]]

        return lemmas, segmentations


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)
    random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train", metavar="TRAIN.tsv", help="a path to the training data file.")
    parser.add_argument("dev", metavar="DEV.tsv", help="a path to the development evaluation data file.")
    parser.add_argument("test", metavar="TEST.tsv", help="a path to the final evaluation data file.")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--char_dim", default=None, type=int, help="Character embedding dimension.")
    parser.add_argument("--rnn_dim", default=None, type=int, help="Dimension of the encoder and the decoder.")
    parser.add_argument("--dropout", default=None, type=float, help="Dropout rate.")
    parser.add_argument("--encoder_layers", default=None, type=int, help="Layer count of the encoder.")
    parser.add_argument("--decoder_layers", default=None, type=int, help="Layer count of the decoder.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = morpho_dataset.MorphoDataset(args.train)
    dev = morpho_dataset.MorphoDataset(args.dev, train=train, shuffle_batches=False)
    test = morpho_dataset.MorphoDataset(args.test, train=train, shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args,
                      len(train.factors[train.LEMMAS].charseqs),
                      len(train.factors[train.LEMMAS].alphabet),
                      len(train.factors[train.PLEMMAS].alphabet),
                      len(train.factors[train.SEGMENTS].alphabet),
                      train.factors[train.LEMMAS].alphabet_map["<bow>"],
                      train.factors[train.LEMMAS].alphabet_map["<eow>"])

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size, args.dropout)

        accuracy, seg_accuracy = network.evaluate("dev", dev, args.batch_size)
        print("{:.3f} / {:.3f}".format(100 * accuracy, 100 * seg_accuracy), flush=True)

    # Predict test data
    with open("{}/segmenter_test.txt".format(args.logdir), "w", encoding="utf-8") as test_file:
        lemmas = test.factors[test.LEMMAS].strings
        plemmas, segmentations = network.predict(test, args.batch_size)
        for plemma, lemma, segmentation in zip(plemmas, lemmas, segmentations):
            try:
                print("{}\t{}\t{}".format(plemma, lemma, " ".join(indicators_to_segments(lemma, segmentation))), file=test_file)
            except Exception as e:
                # An error occured while segmenting. Print the lemma unchanged without segments.
                print(e, file=sys.stderr, flush=True)
                print("{}\t{}\t{}".format(plemma, lemma, lemma), file=test_file)

    network.session.run([network.summary_flusher])
