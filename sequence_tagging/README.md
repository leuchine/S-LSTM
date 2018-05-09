# Named Entity Recognition and Part of Speech Tagging with S-LSTM

Code adapted from https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html

This repo implements a NER/POS model using Tensorflow 

# Config
 Changing the entry `task` to `pos`/`ner` for named entity recognition/ part of speech tagging in `model/config.py`.
 Changing the entry `model_type` to `slstm`/`lstm` for using S-LSTM or vanilla LSTM  in `model/config.py`.

## Getting started
1. Download the GloVe vectors with

```
make glove
```

Alternatively, you can download them manually [here](https://nlp.stanford.edu/projects/glove/) and update the `glove_filename` entry in `config.py`. You can also choose not to load pretrained word vectors by changing the entry `use_pretrained` to `False` in `model/config.py`.


## Run Details

1. Build vocab from the data and extract trimmed glove vectors according to the config in `model/config.py`.

```
python build_data.py
```

2. Train the model with

```
python train.py iteration_num window_size
```

```iteration_num``` is the number of S-LSTM iterations and ```window_size``` decides how many left/right context words are used during S-LSTM iterations.

3. The accuracy and F1 scores on test sets are shown during training. (NER is evaluated using F1, while POS is evaluated using accuracies.)


## Training Data

The training data must be in the following format (identical to the CoNLL2003 dataset).

```
John B-PER
lives O
in O
New B-LOC
York I-LOC
. O

This O
is O
another O
sentence
```

```
# dataset
NER data
dev_filename = "data/eng.testa.iob"
test_filename = "data/eng.testb.iob"
train_filename = "data/eng.train.iob"
```
```
POS data
train_filename = "data/train.pos"
dev_filename= "data/dev.pos"
test_filename= "data/test.pos"
```
## License

This project is licensed under the terms of the apache 2.0 license (as Tensorflow and derivatives). If used for research, citation would be appreciated.

