from __future__ import print_function
from six.moves import xrange
import six.moves.cPickle as pickle
import gzip
import os
import numpy

def generate_matrix(seqs, maxlen, lengths):
    n_samples = len(seqs)
    x= numpy.zeros((n_samples, maxlen)).astype('int64')

    for idx, s in enumerate(seqs):
        if lengths[idx]>= maxlen:
            s=s[:maxlen]
        x[idx, :lengths[idx]] = s
    return x

def prepare_data(seqs, labels):
    lengths = [len(s) for s in seqs]
    labels = numpy.array(labels).astype('int32')
    return [numpy.array(seqs), labels, numpy.array(lengths).astype('int32')]

def remove_unk(x, n_words):
    return [[1 if w >= n_words else w for w in sen] for sen in x] 

def load_data(path, n_words):
    with open(path, 'rb') as f:
        dataset_x, dataset_label= pickle.load(f)
        train_set_x, train_set_y = dataset_x[0], dataset_label[0]
        valid_set_x, valid_set_y =dataset_x[1], dataset_label[1]
        test_set_x, test_set_y = dataset_x[2], dataset_label[2]
    #remove unknown words
    train_set_x = remove_unk(train_set_x, n_words)
    valid_set_x = remove_unk(valid_set_x, n_words)
    test_set_x = remove_unk(test_set_x, n_words)

    return [train_set_x, train_set_y], [valid_set_x, valid_set_y], [test_set_x, test_set_y]
