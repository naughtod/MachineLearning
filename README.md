## About the Project
Multilayer Perceptron (MLP) for Multi Class Classification implementation in C++.
Uses an object oriented approach with nodes and layers as opposed to
matrices which hopefully makes it easy to understand.
 
Hidden layers use sigmoid activation function, output layer uses softmax activation function, standard gradient descent 
optimisation, loss function is cross entropy, all layers except output have a bias, learning rate is 0.001,
epochs is 1500, weights are randomly initialised between 0 and 1.  

The main function uses data from `iris.txt` (see repo) as training and label data so it can be run straight away to see what it does. I've only added a single hidden layer of 4 nodes but you can have as many hidden layers as you like. The full data set is used for training and the accuracy reported is accuracy on the training set. This is just to show that the mlp model is learning but does not give any indication how well it generalises to unseen data. The loss continues to reduce as the Epochs goes up (after 10k Epochs, loss is approx. 0.30), however the accuracy also declines to about 91%. The highest accuracy achieved is approx 95% after around 1400 Epochs.

## Libraries Used
    #include <iostream>
    #include <iomanip>
    #include <vector>
    #include <math.h>
    #include <fstream>

## Getting Started
make sure file `iris.txt` is saved in the same folder as `mlp_multi_class`  
compile with `g++ -o mlp_multi_class mlp_multi_class.cpp`  
run with `./mlp_multi_class`
This will reproduce the test case output below.    

To use the MLP class more generally. First create a vector where each element represents a layer and the value represents the number of nodes in the layer e.g. `std::vector<int> model = {4, 4, 3};` represents a model with three layers, first an input layer with 4 nodes, then a hidden layer with 4 nodes and finally an output layer with 3 nodes. Then pass this vector to the MLP constructor e.g. `MLP mlp(model);`

Make sure your training data is loaded into a 2D vector of doubles where the width is equal to the number of nodes in your model input layer. Also make sure your labels are loaded into a 2D vector of ints where the width is equal to the number of nodes in your model output layer (n.b. if your labels are categorical you will need to one hot encode them). The height in both cases will be the number of instances in your data set.  

Then you can fit the model with your training and label data by calling `mlp.fit(data, labels);`. This will train the weights of the network and output useful information such as the loss, accuracy and final weights.

Now your model is trained you can pass it your test data in the same format as you training data and predict the labels by caling `std::vector<std::vector<double>> result = mlp.predict(test_data);` which will pass the softmax outputs from the final layer for each test instance into the 2D vector `result`.  

## IRIS test case output
    LEARNING CURVE
    Epoch:    0 | loss: 1.88559 - accuracy: 0.333333
    Epoch:  150 | loss: 1.58174 - accuracy: 0.333333
    Epoch:  300 | loss: 1.0912 - accuracy: 0.686667
    Epoch:  450 | loss: 0.811051 - accuracy: 0.773333
    Epoch:  600 | loss: 0.729946 - accuracy: 0.853333
    Epoch:  750 | loss: 0.668401 - accuracy: 0.9
    Epoch:  900 | loss: 0.609014 - accuracy: 0.92
    Epoch: 1050 | loss: 0.557554 - accuracy: 0.933333
    Epoch: 1200 | loss: 0.514846 - accuracy: 0.933333
    Epoch: 1350 | loss: 0.479148 - accuracy: 0.946667
    Epoch: 1500 | loss: 0.448747 - accuracy: 0.946667
    FINAL WEIGHTS
    layerId: 0
    layerId: 1
    node: 0 weight 0: 0.842853
    node: 0 weight 1: 0.396028
    node: 0 weight 2: 0.784158
    node: 0 weight 3: 0.798668
    node: 0 weight 4: 0.912182
    node: 1 weight 0: -0.976271
    node: 1 weight 1: -1.27988
    node: 1 weight 2: 1.71367
    node: 1 weight 3: 1.74994
    node: 1 weight 4: -0.029613
    node: 2 weight 0: 0.518924
    node: 2 weight 1: 0.660087
    node: 2 weight 2: 0.370831
    node: 2 weight 3: 0.512674
    node: 2 weight 4: 0.961164
    node: 3 weight 0: 0.92924
    node: 3 weight 1: 0.643977
    node: 3 weight 2: 0.721949
    node: 3 weight 3: 0.142506
    node: 3 weight 4: 0.609626
    layerId: 2
    node: 0 weight 0: 0.837746
    node: 0 weight 1: -6.50458
    node: 0 weight 2: 0.93341
    node: 0 weight 3: 1.6299
    node: 0 weight 4: 0.986531
    node: 1 weight 0: 0.587456
    node: 1 weight 1: 0.877893
    node: 1 weight 2: 0.30253
    node: 1 weight 3: 1.18244
    node: 1 weight 4: 0.399517
    node: 2 weight 0: -0.495025
    node: 2 weight 1: 6.83847
    node: 2 weight 2: -0.37726
    node: 2 weight 3: -0.713208
    node: 2 weight 4: -0.373559
    PREDICTIONS ON TRAINING DATA
    [1, 0, 0] -> [0.869199, 0.129249, 0.00155198]
    [1, 0, 0] -> [0.867821, 0.130595, 0.00158408]
    [1, 0, 0] -> [0.868294, 0.130133, 0.0015735]
    [1, 0, 0] -> [0.866877, 0.131517, 0.001606]
    [1, 0, 0] -> [0.86923, 0.129219, 0.00155132]
    [1, 0, 0] -> [0.869199, 0.12925, 0.0015513]
    [1, 0, 0] -> [0.867944, 0.130475, 0.00158118]
    [1, 0, 0] -> [0.868701, 0.129736, 0.00156338]
    [1, 0, 0] -> [0.865751, 0.132616, 0.00163301]
    [1, 0, 0] -> [0.868117, 0.130306, 0.00157716]
    [1, 0, 0] -> [0.86955, 0.128906, 0.00154361]
    [1, 0, 0] -> [0.868029, 0.130392, 0.00157883]
    [1, 0, 0] -> [0.867971, 0.130448, 0.00158092]
    [1, 0, 0] -> [0.867909, 0.130506, 0.00158421]
    [1, 0, 0] -> [0.870068, 0.1284, 0.00153172]
    [1, 0, 0] -> [0.869984, 0.128482, 0.00153338]
    [1, 0, 0] -> [0.869713, 0.128748, 0.00153987]
    [1, 0, 0] -> [0.869014, 0.12943, 0.00155611]
    [1, 0, 0] -> [0.869508, 0.128948, 0.00154424]
    [1, 0, 0] -> [0.869267, 0.129183, 0.00155012]
    [1, 0, 0] -> [0.868807, 0.129632, 0.00156047]
    [1, 0, 0] -> [0.868925, 0.129517, 0.00155788]
    [1, 0, 0] -> [0.869413, 0.129039, 0.00154821]
    [1, 0, 0] -> [0.86655, 0.131838, 0.00161231]
    [1, 0, 0] -> [0.866592, 0.131796, 0.00161156]
    [1, 0, 0] -> [0.867201, 0.131201, 0.00159791]
    [1, 0, 0] -> [0.867696, 0.130718, 0.00158609]
    [1, 0, 0] -> [0.869135, 0.129312, 0.00155327]
    [1, 0, 0] -> [0.869167, 0.12928, 0.00155266]
    [1, 0, 0] -> [0.867082, 0.131317, 0.00160088]
    [1, 0, 0] -> [0.866986, 0.131411, 0.00160305]
    [1, 0, 0] -> [0.868798, 0.129641, 0.00156069]
    [1, 0, 0] -> [0.869823, 0.128639, 0.00153743]
    [1, 0, 0] -> [0.869984, 0.128483, 0.00153361]
    [1, 0, 0] -> [0.868117, 0.130306, 0.00157716]
    [1, 0, 0] -> [0.869, 0.129443, 0.00155709]
    [1, 0, 0] -> [0.86965, 0.128808, 0.00154149]
    [1, 0, 0] -> [0.868117, 0.130306, 0.00157716]
    [1, 0, 0] -> [0.866875, 0.131518, 0.00160701]
    [1, 0, 0] -> [0.868847, 0.129593, 0.00155995]
    [1, 0, 0] -> [0.869086, 0.129359, 0.00155467]
    [1, 0, 0] -> [0.861439, 0.136825, 0.001736]
    [1, 0, 0] -> [0.867629, 0.130782, 0.00158933]
    [1, 0, 0] -> [0.867087, 0.131313, 0.00159991]
    [1, 0, 0] -> [0.867968, 0.130452, 0.0015794]
    [1, 0, 0] -> [0.867091, 0.131308, 0.0016009]
    [1, 0, 0] -> [0.86927, 0.12918, 0.00155004]
    [1, 0, 0] -> [0.867724, 0.13069, 0.00158658]
    [1, 0, 0] -> [0.869474, 0.12898, 0.00154538]
    [1, 0, 0] -> [0.868749, 0.129689, 0.00156247]
    [0, 1, 0] -> [0.256587, 0.66354, 0.0798725]
    [0, 1, 0] -> [0.129222, 0.71266, 0.158119]
    [0, 1, 0] -> [0.0684279, 0.685384, 0.246188]
    [0, 1, 0] -> [0.0241937, 0.568824, 0.406983]
    [0, 1, 0] -> [0.0398152, 0.631456, 0.328729]
    [0, 1, 0] -> [0.0231416, 0.562918, 0.41394]
    [0, 1, 0] -> [0.0489293, 0.654289, 0.296781]
    [0, 1, 0] -> [0.265601, 0.657885, 0.0765137]
    [0, 1, 0] -> [0.126835, 0.712651, 0.160514]
    [0, 1, 0] -> [0.0379591, 0.625833, 0.336208]
    [0, 1, 0] -> [0.0699584, 0.687048, 0.242993]
    [0, 1, 0] -> [0.0829639, 0.698789, 0.218247]
    [0, 1, 0] -> [0.144429, 0.71146, 0.144111]
    [0, 1, 0] -> [0.0231807, 0.56315, 0.41367]
    [0, 1, 0] -> [0.376052, 0.578202, 0.0457454]
    [0, 1, 0] -> [0.291333, 0.640952, 0.0677151]
    [0, 1, 0] -> [0.0156919, 0.510162, 0.474146]
    [0, 1, 0] -> [0.227089, 0.680611, 0.0922997]
    [0, 1, 0] -> [0.00699336, 0.401748, 0.591258]
    [0, 1, 0] -> [0.141817, 0.711804, 0.146379]
    [0, 1, 0] -> [0.00641484, 0.390732, 0.602854]
    [0, 1, 0] -> [0.25076, 0.667081, 0.0821591]
    [0, 1, 0] -> [0.00504953, 0.361015, 0.633935]
    [0, 1, 0] -> [0.0368444, 0.6223, 0.340856]
    [0, 1, 0] -> [0.218292, 0.685246, 0.0964615]
    [0, 1, 0] -> [0.210601, 0.689094, 0.100304]
    [0, 1, 0] -> [0.0517453, 0.660052, 0.288203]
    [0, 1, 0] -> [0.0121154, 0.474793, 0.513092]
    [0, 1, 0] -> [0.0268078, 0.582351, 0.390841]
    [0, 1, 0] -> [0.531182, 0.447073, 0.0217457]
    [0, 1, 0] -> [0.128462, 0.712646, 0.158892]
    [0, 1, 0] -> [0.230019, 0.679011, 0.09097]
    [0, 1, 0] -> [0.224608, 0.681941, 0.0934514]
    [0, 1, 0] -> [0.00248909, 0.281536, 0.715975]
    [0, 1, 0] -> [0.0107688, 0.458796, 0.530435]
    [0, 1, 0] -> [0.070666, 0.687877, 0.241457]
    [0, 1, 0] -> [0.0920227, 0.704356, 0.203621]
    [0, 1, 0] -> [0.029525, 0.594821, 0.375654]
    [0, 1, 0] -> [0.128839, 0.712656, 0.158505]
    [0, 1, 0] -> [0.0413321, 0.63575, 0.322918]
    [0, 1, 0] -> [0.0186713, 0.533921, 0.447408]
    [0, 1, 0] -> [0.0433431, 0.641142, 0.315515]
    [0, 1, 0] -> [0.136022, 0.712372, 0.151607]
    [0, 1, 0] -> [0.254504, 0.664791, 0.0807047]
    [0, 1, 0] -> [0.0422904, 0.638365, 0.319345]
    [0, 1, 0] -> [0.154596, 0.709635, 0.135769]
    [0, 1, 0] -> [0.087498, 0.701798, 0.210704]
    [0, 1, 0] -> [0.158375, 0.708773, 0.132851]
    [0, 1, 0] -> [0.520017, 0.456964, 0.0230189]
    [0, 1, 0] -> [0.0953081, 0.705938, 0.198754]
    [0, 0, 1] -> [0.000892244, 0.190439, 0.808668]
    [0, 0, 1] -> [0.00140028, 0.226948, 0.771652]
    [0, 0, 1] -> [0.00134779, 0.223648, 0.775005]
    [0, 0, 1] -> [0.00145364, 0.230223, 0.768324]
    [0, 0, 1] -> [0.00104825, 0.202882, 0.79607]
    [0, 0, 1] -> [0.00101013, 0.199964, 0.799026]
    [0, 0, 1] -> [0.00163924, 0.240982, 0.757379]
    [0, 0, 1] -> [0.00128315, 0.219455, 0.779262]
    [0, 0, 1] -> [0.00114991, 0.210334, 0.788516]
    [0, 0, 1] -> [0.00123176, 0.216018, 0.78275]
    [0, 0, 1] -> [0.00419998, 0.339066, 0.656734]
    [0, 0, 1] -> [0.00163936, 0.241001, 0.757359]
    [0, 0, 1] -> [0.00174205, 0.246601, 0.751657]
    [0, 0, 1] -> [0.00117128, 0.21184, 0.786989]
    [0, 0, 1] -> [0.00101953, 0.200687, 0.798293]
    [0, 0, 1] -> [0.00157202, 0.237198, 0.76123]
    [0, 0, 1] -> [0.00211928, 0.265355, 0.732526]
    [0, 0, 1] -> [0.00139727, 0.226765, 0.771837]
    [0, 0, 1] -> [0.000817576, 0.183967, 0.815215]
    [0, 0, 1] -> [0.00188959, 0.254244, 0.743867]
    [0, 0, 1] -> [0.00138894, 0.226245, 0.772366]
    [0, 0, 1] -> [0.00148861, 0.23232, 0.766192]
    [0, 0, 1] -> [0.000969266, 0.196751, 0.80228]
    [0, 0, 1] -> [0.00346911, 0.317191, 0.67934]
    [0, 0, 1] -> [0.00166992, 0.242694, 0.755636]
    [0, 0, 1] -> [0.00229322, 0.273204, 0.724503]
    [0, 0, 1] -> [0.00457042, 0.349032, 0.646397]
    [0, 0, 1] -> [0.00448517, 0.346797, 0.648718]
    [0, 0, 1] -> [0.00110971, 0.207442, 0.791448]
    [0, 0, 1] -> [0.00380427, 0.327627, 0.668569]
    [0, 0, 1] -> [0.00138154, 0.225782, 0.772836]
    [0, 0, 1] -> [0.00370016, 0.324466, 0.671834]
    [0, 0, 1] -> [0.00104458, 0.202604, 0.796351]
    [0, 0, 1] -> [0.00537846, 0.368727, 0.625894]
    [0, 0, 1] -> [0.00157186, 0.237186, 0.761242]
    [0, 0, 1] -> [0.00127434, 0.218873, 0.779853]
    [0, 0, 1] -> [0.00117626, 0.212193, 0.786631]
    [0, 0, 1] -> [0.00218037, 0.268162, 0.729658]
    [0, 0, 1] -> [0.00500012, 0.359824, 0.635176]
    [0, 0, 1] -> [0.00249346, 0.28172, 0.715786]
    [0, 0, 1] -> [0.00117956, 0.212424, 0.786396]
    [0, 0, 1] -> [0.00299346, 0.300986, 0.696021]
    [0, 0, 1] -> [0.00140028, 0.226948, 0.771652]
    [0, 0, 1] -> [0.00112612, 0.208632, 0.790241]
    [0, 0, 1] -> [0.00113637, 0.209369, 0.789495]
    [0, 0, 1] -> [0.00183588, 0.251516, 0.746648]
    [0, 0, 1] -> [0.0018541, 0.252447, 0.745699]
    [0, 0, 1] -> [0.00249136, 0.281632, 0.715876]
    [0, 0, 1] -> [0.00146278, 0.230776, 0.767761]
    [0, 0, 1] -> [0.00233233, 0.274905, 0.722763]
