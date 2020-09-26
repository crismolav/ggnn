# Gated Graph Neural Networks

> ## Code based on https://github.com/microsoft/gated-graph-neural-network-samples.

This repository contains and implementation of a Gated Graph Neural Networks
of [Li et al. 2015](https://arxiv.org/abs/1511.05493) for tranduscing one dependency treebank into another. 

For this project we utilize two dependecy treebanks: Stanford and Matsumoto dependency tree banks and we convert one into another and vice-versa.

## Preparing data for training

This step is done in order to preprocess the input data to fit the format required. For this we use the code to_graph.py in the folder parser. We use it in the following way
```
python to_graph.py btb test std
```

The first argument is the type of problem, which should be btb, bank-to-bank (we previously tried with intermediate problems such as predicting a tree roots). 
The second argument indicates which is the set we are tranforming (train, valid or test). The last argument refers to the input treebank (either Stanford as std or Matsumoto as nivre).


## Running Graph Neural Network training

To run the code we run the following line.
```
python chem_tensorflow_dense.py --log_dir borrar  --restrict_data 100 --pr btb --alpha 0.003
```

where --restrict_data limits the number of training examples and -alpha sets the learning rate. To look at all the available options for running the code refer to chem_tensorflow_dense.py and look at the top of the code where it says "options".

the previous code will output something like the following

```
Average train batch size: 3.85

Average val batch size: 4.35

== Epoch 1
 Train: loss: 154.58027 | acc: 0:154.58028 | error_ratio: 0:2324.03578 | instances/sec: 13.91
Train Attachment scores - LAS : 10.6% - UAS : 14.7% - UAS_e : 57.6%
 Valid: loss: 103.24148 | acc: 0:103.24148 | error_ratio: 0:1552.18307 | instances/sec: 31.51
Valid Attachment scores - LAS : 19.1% - UAS : 23.2% - UAS_e : 75.0%
  (Best epoch so far, cum. val. acc decreased to 0.80929 from inf. Saving to 'borrar/2020-09-23-19-00-58_22474_model_best.pickle')
Test Attachment scores - LAS : 17.76% - UAS : 21.10% - UAS_e : 74.79%
```
## Restoring model

To restore a model and use it on the test set you can run the following code

``` 
python chem_tensorflow_dense.py --pr btb --log_dir best/st --evaluate --restore st_normal.pickle
```

For the above case we previsouly saved a model as st_normal.pickle in the folder best/st.



