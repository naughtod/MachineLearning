## About the Project
Multilayer Perceptron (MLP) implementation in C++.
Uses an object oriented approach with nodes and layers as opposed to
matrices which makes it less efficient but hopefully easy to understand.
 
All layers use sigmoid activation function, standard gradient descent 
optimisation, loss function is mean squared error, all layers except output have a bias, learning rate is 0.2,
epochs is 4000, weights are randomly initialised between 0 and 1. Note the predictions can be classified based on a decision 
boundary of 0.5 (i.e. 0.4 -> 0, 0.6 -> 1) however I haven't implemented this 
since it is more informative to know the value of the final output unit.  

The main function contains training data for learning the XOR function so it can be run straight away to see what it does.

## Libraries Used
    #include <iostream>
    #include <iomanip>
    #include <vector>
    #include <math.h>

## Getting Started
compile with `g++ -o mlp_gd mlp_gd.cpp`

run with `./mlp_gd`

## XOR test case output
    TRAINING DATA
    [00] : 0
    [01] : 1
    [10] : 1
    [11] : 0
    LEARNING CURVE
    Epoch:    0 | loss: 0.309562
    Epoch:  400 | loss: 0.242668
    Epoch:  800 | loss: 0.215312
    Epoch: 1200 | loss: 0.179951
    Epoch: 1600 | loss: 0.148613
    Epoch: 2000 | loss: 0.0597702
    Epoch: 2400 | loss: 0.0214261
    Epoch: 2800 | loss: 0.0114505
    Epoch: 3200 | loss: 0.00751715
    Epoch: 3600 | loss: 0.00550376
    Epoch: 4000 | loss: 0.00430355
    FINAL WEIGHTS
    layerId: 0
    layerId: 1
    node: 0 weight 0: 3.52227
    node: 0 weight 1: 3.49842
    node: 0 weight 2: -5.36145
    node: 1 weight 0: 5.87393
    node: 1 weight 1: 5.73709
    node: 1 weight 2: -2.37409
    layerId: 2
    node: 0 weight 0: -7.84222
    node: 0 weight 1: 7.20044
    node: 0 weight 2: -3.21116
    TEST PREDICTIONS
    [0, 0] -> 0.0669482
    [0, 1] -> 0.936711
    [1, 0] -> 0.937182
    [1, 1] -> 0.0691418
