# Classification using S-LSTM

## Getting started
Download the GloVe vectors
```
make glove
```

## Run Details
Place the dataset files at folder ```sst_data```.

1. Preprocess data 

```
python sst_preprocessing.py dataset_name
```

For example, 

```
python sst_preprocessing.py apparel
```

2. Train and evaluate the model with

```
python slstm.py iteration_num window_size dataset_name model_name
```

```iteration_num``` is the number of S-LSTM iterations and ```window_size``` decides how many left/right context words are used during S-LSTM iterations. ```model_name``` can be "slstm", "cnn" and "lstm"

For example, 
```
python slstm.py 7 2 apparel slstm
```