import numpy as np
import os
import tensorflow as tf
import time

from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        if self.config.char_use_mlstm:
            self.config.hidden_size_char=self.config.hidden_size_char*2
            self.config.dim_char=self.config.dim_char*2
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings
        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                if self.config.char_use_mlstm:
                    initial_hidden_states, initial_cell_states=self.mlstm_cell("char_mlstm", self.config.hidden_size_char,
                     word_lengths, char_embeddings, tf.identity(char_embeddings), 3)
                    initial_hidden_states=tf.reduce_sum(initial_hidden_states, axis=1)
                    output = tf.reshape(initial_hidden_states,
                            shape=[s[0], s[1], self.config.hidden_size_char])
                else:
                    # bi lstm on chars
                    cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                            state_is_tuple=True)
                    cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                            state_is_tuple=True)
                    _output = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw, cell_bw, char_embeddings,
                            sequence_length=word_lengths, dtype=tf.float32)
                    
                    # read and concat output
                    _, ((_, output_fw), (_, output_bw)) = _output
                    output = tf.concat([output_fw, output_bw], axis=-1)

                    # shape = (batch size, max sentence length, char hidden size)
                    output = tf.reshape(output,
                            shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)

    def lstm_layer(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def get_hidden_states_before(self, hidden_states, step, shape, hidden_size):
        #padding zeros
        padding=tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
        #remove last steps
        displaced_hidden_states=hidden_states[:,:-step,:]
        #concat padding
        return tf.concat([padding, displaced_hidden_states], axis=1)
        #return tf.cond(step<=shape[1], lambda: tf.concat([padding, displaced_hidden_states], axis=1), lambda: tf.zeros((shape[0], shape[1], self.config.hidden_size_sum), dtype=tf.float32))

    def get_hidden_states_after(self, hidden_states, step, shape, hidden_size):
        #padding zeros
        padding=tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
        #remove last steps
        displaced_hidden_states=hidden_states[:,step:,:]
        #concat padding
        return tf.concat([displaced_hidden_states, padding], axis=1)
        #return tf.cond(step<=shape[1], lambda: tf.concat([displaced_hidden_states, padding], axis=1), lambda: tf.zeros((shape[0], shape[1], self.config.hidden_size_sum), dtype=tf.float32))

    def sum_together(self, l):
        combined_state=None
        for tensor in l:
            if combined_state==None:
                combined_state=tensor
            else:
                combined_state=combined_state+tensor
        return combined_state

    def mlstm_cell(self, name_scope_name, hidden_size, lengths, initial_hidden_states, initial_cell_states, num_layers):
        with tf.name_scope(name_scope_name):
            #Word parameters 
            #forget gate for left 
            with tf.name_scope("f1_gate"):
                #current
                Wxf1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                #left right
                Whf1 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
                #initial state
                Wif1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
                #dummy node
                Wdf1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
            #forget gate for right 
            with tf.name_scope("f2_gate"):
                Wxf2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                Whf2 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
                Wif2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
                Wdf2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
            #forget gate for inital states     
            with tf.name_scope("f3_gate"):
                Wxf3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                Whf3 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
                Wif3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
                Wdf3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
            #forget gate for dummy states     
            with tf.name_scope("f4_gate"):
                Wxf4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                Whf4 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
                Wif4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
                Wdf4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
            #input gate for current state     
            with tf.name_scope("i_gate"):
                Wxi = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxi")
                Whi = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whi")
                Wii = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wii")
                Wdi = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdi")
            #input gate for output gate
            with tf.name_scope("o_gate"):
                Wxo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxo")
                Who = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Who")
                Wio = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wio")
                Wdo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdo")
            #bias for the gates    
            with tf.name_scope("biases"):
                bi = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bi")
                bo = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")
                bf1 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf1")
                bf2 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf2")
                bf3 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf3")
                bf4 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf4")

            #dummy node gated attention parameters
            #input gate for dummy state
            with tf.name_scope("gated_d_gate"):
                gated_Wxd = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                gated_Whd = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
            #output gate
            with tf.name_scope("gated_o_gate"):
                gated_Wxo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxo")
                gated_Who = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Who")
            #forget gate for states of word
            with tf.name_scope("gated_f_gate"):
                gated_Wxf = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxo")
                gated_Whf = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Who")
            #biases
            with tf.name_scope("gated_biases"):
                gated_bd = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bi")
                gated_bo = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")
                gated_bf = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")

        #filters for attention        
        mask_softmax_score=tf.cast(tf.sequence_mask(lengths), tf.float32)*1e25-1e25
        mask_softmax_score_expanded=tf.expand_dims(mask_softmax_score, dim=2)               
        #filter invalid steps
        sequence_mask=tf.expand_dims(tf.cast(tf.sequence_mask(lengths), tf.float32),axis=2)
        #filter embedding states
        initial_hidden_states=initial_hidden_states*sequence_mask
        initial_cell_states=initial_cell_states*sequence_mask
        #record shape of the batch
        shape=tf.shape(initial_hidden_states)
        
        #initial embedding states
        embedding_hidden_state=tf.reshape(initial_hidden_states, [-1, hidden_size])      
        embedding_cell_state=tf.reshape(initial_cell_states, [-1, hidden_size])

        #randomly initialize the states
        if self.config.random_initialize:
            initial_hidden_states=tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None, name=None)
            initial_cell_states=tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None, name=None)
            #filter it
            initial_hidden_states=initial_hidden_states*sequence_mask
            initial_cell_states=initial_cell_states*sequence_mask

        #inital dummy node states
        dummynode_hidden_states=tf.reduce_mean(initial_hidden_states, axis=1)
        dummynode_cell_states=tf.reduce_mean(initial_cell_states, axis=1)

        for i in range(num_layers):
            #update dummy node states
            #average states
            combined_word_hidden_state=tf.reduce_mean(initial_hidden_states, axis=1)
            reshaped_hidden_output=tf.reshape(initial_hidden_states, [-1, hidden_size])
            #copy dummy states for computing forget gate
            transformed_dummynode_hidden_states=tf.reshape(tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1],1]), [-1, hidden_size])
            #input gate
            gated_d_t = tf.nn.sigmoid(
                tf.matmul(dummynode_hidden_states, gated_Wxd) + tf.matmul(combined_word_hidden_state, gated_Whd) + gated_bd
            )
            #output gate
            gated_o_t = tf.nn.sigmoid(
                tf.matmul(dummynode_hidden_states, gated_Wxo) + tf.matmul(combined_word_hidden_state, gated_Who) + gated_bo
            )
            #forget gate for hidden states
            gated_f_t = tf.nn.sigmoid(
                tf.matmul(transformed_dummynode_hidden_states, gated_Wxf) + tf.matmul(reshaped_hidden_output, gated_Whf) + gated_bf
            )

            #softmax on each hidden dimension 
            reshaped_gated_f_t=tf.reshape(gated_f_t, [shape[0], shape[1], hidden_size])+ mask_softmax_score_expanded
            gated_softmax_scores=tf.nn.softmax(tf.concat([reshaped_gated_f_t, tf.expand_dims(gated_d_t, dim=1)], axis=1), dim=1)
            #split the softmax scores
            new_reshaped_gated_f_t=gated_softmax_scores[:,:shape[1],:]
            new_gated_d_t=gated_softmax_scores[:,shape[1]:,:]
            #new dummy states
            dummy_c_t=tf.reduce_sum(new_reshaped_gated_f_t * initial_cell_states, axis=1) + tf.squeeze(new_gated_d_t, axis=1)*dummynode_cell_states
            dummy_h_t=gated_o_t * tf.nn.tanh(dummy_c_t)

            #update word node states
            #get states before
            initial_hidden_states_before=[tf.reshape(self.get_hidden_states_before(initial_hidden_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
            initial_hidden_states_before=self.sum_together(initial_hidden_states_before)
            initial_hidden_states_after= [tf.reshape(self.get_hidden_states_after(initial_hidden_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
            initial_hidden_states_after=self.sum_together(initial_hidden_states_after)
            #get states after
            initial_cell_states_before=[tf.reshape(self.get_hidden_states_before(initial_cell_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
            initial_cell_states_before=self.sum_together(initial_cell_states_before)
            initial_cell_states_after=[tf.reshape(self.get_hidden_states_after(initial_cell_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
            initial_cell_states_after=self.sum_together(initial_cell_states_after)
            
            #reshape for matmul
            initial_hidden_states=tf.reshape(initial_hidden_states, [-1, hidden_size])
            initial_cell_states=tf.reshape(initial_cell_states, [-1, hidden_size])

            #concat before and after hidden states
            concat_before_after=tf.concat([initial_hidden_states_before, initial_hidden_states_after], axis=1)

            #copy dummy node states 
            transformed_dummynode_hidden_states=tf.reshape(tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1],1]), [-1, hidden_size])
            transformed_dummynode_cell_states=tf.reshape(tf.tile(tf.expand_dims(dummynode_cell_states, axis=1), [1, shape[1],1]), [-1, hidden_size])

            f1_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf1) + tf.matmul(concat_before_after, Whf1) + 
                tf.matmul(embedding_hidden_state, Wif1) + tf.matmul(transformed_dummynode_hidden_states, Wdf1)+ bf1
            )

            f2_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf2) + tf.matmul(concat_before_after, Whf2) + 
                tf.matmul(embedding_hidden_state, Wif2) + tf.matmul(transformed_dummynode_hidden_states, Wdf2)+ bf2
            )

            f3_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf3) + tf.matmul(concat_before_after, Whf3) + 
                tf.matmul(embedding_hidden_state, Wif3) + tf.matmul(transformed_dummynode_hidden_states, Wdf3) + bf3
            )

            f4_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf4) + tf.matmul(concat_before_after, Whf4) + 
                tf.matmul(embedding_hidden_state, Wif4) + tf.matmul(transformed_dummynode_hidden_states, Wdf4) + bf4
            )
            
            i_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxi) + tf.matmul(concat_before_after, Whi) + 
                tf.matmul(embedding_hidden_state, Wii) + tf.matmul(transformed_dummynode_hidden_states, Wdi)+ bi
            )
            
            o_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxo) + tf.matmul(concat_before_after, Who) + 
                tf.matmul(embedding_hidden_state, Wio) + tf.matmul(transformed_dummynode_hidden_states, Wdo) + bo
            )
            
            f1_t, f2_t, f3_t, f4_t, i_t=tf.expand_dims(f1_t, axis=1), tf.expand_dims(f2_t, axis=1),tf.expand_dims(f3_t, axis=1), tf.expand_dims(f4_t, axis=1), tf.expand_dims(i_t, axis=1)


            five_gates=tf.concat([f1_t, f2_t, f3_t, f4_t,i_t], axis=1)
            five_gates=tf.nn.softmax(five_gates, dim=1)
            f1_t,f2_t,f3_t, f4_t,i_t= tf.split(five_gates, num_or_size_splits=5, axis=1)
            
            f1_t, f2_t, f3_t, f4_t, i_t=tf.squeeze(f1_t, axis=1), tf.squeeze(f2_t, axis=1),tf.squeeze(f3_t, axis=1), tf.squeeze(f4_t, axis=1),tf.squeeze(i_t, axis=1)

            c_t = (f1_t * initial_cell_states_before) + (f2_t * initial_cell_states_after)+(f3_t * embedding_cell_state)+ (f4_t * transformed_dummynode_cell_states)+ (i_t * initial_cell_states)
            
            h_t = o_t * tf.nn.tanh(c_t)

            #update states
            initial_hidden_states=tf.reshape(h_t, [shape[0], shape[1], hidden_size])
            initial_cell_states=tf.reshape(c_t, [shape[0], shape[1], hidden_size])
            initial_hidden_states=initial_hidden_states*sequence_mask
            initial_cell_states=initial_cell_states*sequence_mask

            dummynode_hidden_states=dummy_h_t
            dummynode_cell_states=dummy_c_t

        initial_hidden_states = tf.nn.dropout(initial_hidden_states,self.dropout)
        initial_cell_states = tf.nn.dropout(initial_cell_states, self.dropout)

        return initial_hidden_states, initial_cell_states

    def create_layers_two_gates(self):
        
        initial_hidden_states, initial_cell_states=self.mlstm_cell("word_mlstm", self.config.hidden_size_sum,
         self.sequence_lengths, self.word_embeddings, tf.identity(self.word_embeddings), self.config.layer)

        #NER label predictions
        output=initial_hidden_states
        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[self.config.hidden_size_sum, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, self.config.hidden_size_sum])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])


    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        print("Using model: "+self.config.model_type)
        if self.config.model_type=='slstm':
            self.create_layers_two_gates()
        elif self.config.model_type=='lstm':
            self.lstm_layer()



    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch, test):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " dev - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        start_time = time.time()            
        metrics2 = self.run_evaluate(test)
        msg2 = " test - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics2.items()])
        self.logger.info(msg2)
        print("Len test "+str(len(test)))
        print("Decoding Time = %.3f seconds\n" % (time.time()-start_time))
            
        return metrics["f1"]


    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags, self.config))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags, self.config))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

                """p_one= float(len(lab_chunks & lab_pred_chunks))/len(lab_pred_chunks) if len(lab_chunks & lab_pred_chunks) > 0 else 0
                r_one= float(len(lab_chunks & lab_pred_chunks))/len(lab_chunks) if len(lab_chunks & lab_pred_chunks) > 0 else 0
                f1_one=2 * p_one * r_one / (p_one + r_one) if len(lab_chunks & lab_pred_chunks) > 0 else 0
                print("length: "+str(length))
                print("f1: "+str(f1_one))"""

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds

