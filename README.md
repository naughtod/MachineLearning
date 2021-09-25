## About the Project
Multilayer Perceptron (MLP) implementation in C++.
Uses an object oriented approach with nodes and layers as opposed to
matrices which makes it less efficient but hopefully easy to understand.
 
All layers use sigmoid activation function, standard gradient descent 
optimisation, all layers except output have a bias, learning rate is 0.2,
epochs is 4000. Note the predictions can be classified based on a decision 
boundary of 0.5 (i.e. 0.4 -> 0, 0.6 -> 1) however I haven't implemented this 
since it is more informative to know the value of the final output unit.  

The main function contains training data for learning the XOR function so it can be run straight away to see what it does.

## Libraries Used
Only standard C++ libraries

`#include <iostream>`

`#include <iomanip>`

`#include <vector>`

`#include <math.h>`

## Getting Started
compile with `g++ -o mlp_gd mlp_gd.cpp`

run with `./mlp_gd`

## XOR test case output
    Training Data: 
    [00] : 0  
    [01] : 1  
    [10] : 1  
    [11] : 0
    Epoch:    0 | loss: 0.309562
    Epoch:  500 | loss: 0.238596
    Epoch: 1000 | loss: 0.195912
    Epoch: 1500 | loss: 0.158219
    Epoch: 2000 | loss: 0.0597702
    Epoch: 2500 | loss: 0.0178055
    Epoch: 3000 | loss: 0.00911406
    Epoch: 3500 | loss: 0.00590626
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
    [0, 0] -> 0.0669482
    [0, 1] -> 0.936711
    [1, 0] -> 0.937182
    [1, 1] -> 0.0691418
