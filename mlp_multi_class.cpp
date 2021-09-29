#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h> 
#include <fstream>

/**
 * Multilayer Perceptron (MLP) implementation in C++ by David Naughton.
 * Uses an object oriented approach with nodes and layers as opposed to
 * matrices which makes it less efficient but hopefully easy to understand.
 * 
 * Hidden layers use sigmoid activation function, output layer uses a softmax 
 * activation function, standard gradient descent optimisation, loss function is
 * cross entropy loss, all layers except output have a bias, learning rate is 
 * 0.001, epochs is 1500. 
 */

using namespace std;
#define LR 0.001
#define EPOCHS 1500
#define PRINT_PADDING 4
#define EPOCH_PRINTS 10

// forward declaration due to codependent classes node and layer
class Layer; 

/**
 * unit with value and weights connected to layer to the left
 */ 
class Node {
    public:
        int idInLayer;
        double value;
        double delta;
        std::vector<double> leftWeights;
        std::vector<double> gradients;

        /**
         * constructor sets size of weights to number of units in previous layer
         * initialises weights to random values between 0 and 1
         * note bias nodes and input nodes have no weights
         */
        Node(int leftUnits, int idInLayer) {
            this->idInLayer = idInLayer;

            for (int i=0;i<leftUnits;i++)
                leftWeights.push_back(rand()/(double)RAND_MAX);
            
            gradients.resize(leftUnits, 0);
        }

        /**
         * Update weights and then clear gradients for next epoch
         */
        void updateWeights() {            
            // new weight = old weight - LR * gradient
            for (int i=0;i<leftWeights.size();i++)
                leftWeights[i] -= LR * gradients[i];

            // clear gradients
            std::fill(gradients.begin(), gradients.end(), 0);
        }

        /**
         * prints the weights of node
         */
        void printWeights() {
            for (int i=0;i<leftWeights.size();i++)
                std::cout << "node: " << idInLayer << " " <<
                    "weight "<< i << ": " << leftWeights[i] << std::endl;
        }

        // methods implemented beneath layer class declaration
        void predict(Layer leftLayer);
        void predictOutput(Layer leftLayer);
        void calculateGradients(Layer leftLayer, double loss);
        void calculateGradients(Layer leftLayer, Layer rightLayer);
};

/**
 * softmax function applied to output vector
 */
std::vector<double> softmax(std::vector<double> values) {
    std::vector<double> result;
    double denominator = 0;

    for (int i=0;i<values.size();i++) 
        denominator += exp(values[i]);

    for (int i=0;i<values.size();i++) 
        result.push_back(exp(values[i]) / denominator);

    return result;
}

/**
 * sigmoid activation function
 */ 
double sigmoid(double innerProduct) {
    return 1 / (1 + exp(-innerProduct));
}

/**
 * sigmoid derivative
 */
double sigmoidDerivative(double sigmoid) {
    return sigmoid * (1 - sigmoid);
}


/**
 * cross entropy loss derivative wrt output, used for calculating deltas for
 * output layer nodes
 */
double lossDerivativeWRTOutput(std::vector<double> outputValues, 
    std::vector<int> labels, int outputIndex) {
    double sum = 0;
    
    for (int i=0;i<outputValues.size();i++) { 

        // cross entropy loss = - Sum_over_i pi * log qi
        // derivative wrt qj for term where i = j, 
        // (- pj / qj) * qj * (1 - qj) = pj * (qj - 1)
        // derivative wrt qj for terms where i != j,
        // (- pi / qi) * -qi * qj = pi * qj
        if (i == outputIndex) {
            sum += labels[i] * (outputValues[outputIndex] - 1);
        } else {
            sum += labels[i] * outputValues[outputIndex];
        }
    }
    
    return sum;
}


/**
 * takes the inner product of node values and weights 
 */
double innerProduct(std::vector<Node> v1, std::vector<double> v2) {
    double sum = 0;

    for (int i=0;i<v1.size();i++)
        sum += v1[i].value * v2[i];

    return sum;
}


/**
 * layer class contains nodes
 */ 
class Layer {
    public:
        int layerId;
        bool outputLayer;
        std::vector<Node> nodes;
        std::vector<double> outputValues;

        /**
         *  constructor creates nodes for layer
         */
        Layer(int units, int unitsLeft, bool bias, int layerId) {
            this->layerId = layerId;
            this->outputLayer = !bias;

            for (int i=0;i<units;i++) {
                // plus 1 for weight to bias in previous layer
                Node *node = new Node(unitsLeft + 1, i);
                nodes.push_back(*node);           
            }

            // add bias node with value 1 if not output layer
            if (bias) {
                Node *node = new Node(0, units);
                node->value = 1;
                nodes.push_back(*node);
            }  
        }

        /**
         * runs prediction function for every node in layer 
         */
        void predict(Layer layer) {

            // prediction is different for hidden layers vs output layer due to
            // different activation functions
            if (!outputLayer) {
                for (int i=0;i<nodes.size();i++) 
                    nodes[i].predict(layer); 

            } else {
                // convert values in output layer nodes to softmax values
                std::vector<double> outputValues; 

                for (int i=0;i<nodes.size();i++) {
                    nodes[i].predictOutput(layer);
                    outputValues.push_back(nodes[i].value);
                }

                outputValues = softmax(outputValues);
                this->outputValues = outputValues;

                for (int i=0;i<nodes.size();i++) 
                    nodes[i].value = outputValues[i];
            }

        }

        /**
         * runs update functions for every node in output layer
         * 
         */
        void calculateGradients(Layer leftLayer, std::vector<int> labels) {
            for (int i=0;i<nodes.size();i++)
                nodes[i].calculateGradients(leftLayer, 
                    lossDerivativeWRTOutput(this->outputValues, labels, i));
        }

        /**
         * runs update functions for every node in layer
         */
        void calculateGradients(Layer leftLayer, Layer rightLayer) {
            for (int i=0;i<nodes.size();i++)
                nodes[i].calculateGradients(leftLayer, rightLayer);
        }

        /**
         * runs update functions for every node in layer
         */
        void updateWeights() {
            for (int i=0;i<nodes.size();i++)
                nodes[i].updateWeights();
        }

        /**
         * prints the weights of nodes in layer
         */
        void printWeights() {
            std::cout << "layerId: " << layerId << std::endl;

            for (int i=0;i<nodes.size();i++)
                nodes[i].printWeights();
        }

};



/**
 * Calculate gradients cumulatively for hidden layers, this is the 
 * backpropagation algorithm
 */
void Node::calculateGradients(Layer leftLayer, Layer rightLayer) {
    // don't run for bias or input nodes
    if (leftWeights.size() > 0) {
        // update for non output layers
        delta = sigmoidDerivative(value);

        // multiply by the weighted sum of deltas from layer on the right
        double nodeError = 0;

        for (int i=0;i<rightLayer.nodes.size();i++) {
            // don't ask for weight / delta from bias nodes
            if (rightLayer.nodes[rightLayer.nodes.size()-1].leftWeights.size() > 0)
                nodeError += rightLayer.nodes[i].delta * 
                    rightLayer.nodes[i].leftWeights[idInLayer];
        }

        delta *= nodeError;

        // gradient for weight i is incremented by the value of node i * delta
        for (int i=0;i<gradients.size();i++)
            gradients[i] += leftLayer.nodes[i].value * delta;
    }
}


/**
 * Calculate gradients cumulatively for output layer, loss is calculated in the
 * lossDerivativeWRTOutput function
 */
void Node::calculateGradients(Layer leftLayer, double loss) {
    delta = loss;
    
    for (int i=0;i<leftWeights.size();i++)
        gradients[i] += leftLayer.nodes[i].value * delta;
}


/**
 * calculate the value of the node by taking the sigmoid of the inner product of
 * connected weights and values from previous layer
 */ 
void Node::predict(Layer leftLayer) {
    // don't run for bias or input nodes
    if (leftWeights.size() > 0)
        value = sigmoid(innerProduct(leftLayer.nodes, leftWeights));
}

/**
 * calculate the value of the nodes in output layer, no sigmoid as the softmax 
 * is taken on the layer level
 */ 
void Node::predictOutput(Layer leftLayer) {
    value = innerProduct(leftLayer.nodes, leftWeights);
}


/**
 * Multi-layer perceptron class 
 */ 
class MLP {
    public:
        std::vector<Layer> layers;

        /**
         * constructor initialises the layers and nodes in MLP architecture
         */         
        MLP(std::vector<int> layerSizes) {

            // add input unit layer, -1 because there is no previous layer
            Layer *l0 = new Layer(layerSizes[0], -1, true, 0);
            layers.push_back(*l0);

            // add other layers
            for (int i=0;i<layerSizes.size()-1;i++) {
                Layer *l;

                // if output layer then there is no bias node added
                if (i+1==layerSizes.size()-1) {
                    l = new Layer(layerSizes[i+1],layerSizes[i], false, i+1);
                } else {
                    l = new Layer(layerSizes[i+1],layerSizes[i], true, i+1);
                }

                layers.push_back(*l);
            }
        }


        /**
         *  predicts output for every row of data in input Matrix
         *  returns the output predictions in equivalent row of Matrix
         */
        std::vector<std::vector<double>> predict(std::vector<std::vector<double>> m) {
            int outputUnits = layers[layers.size()-1].nodes.size();
            
            std::vector<std::vector<double>> result;
            result.resize(m.size(), std::vector<double>(outputUnits));
            
            // calculate prediction for each instance
            for (int i=0;i<m.size();i++) {
            
                predictInstance(m, i);
                
                // transfer the values from the final layer to the result
                for (int j=0;j<outputUnits;j++) 
                    result[i][j] = layers[layers.size()-1].nodes[j].value;
            }

            return result;
        }


        /**
         * for a single instance will calculate output units via
         * a forward pass through network
         */
        void predictInstance(std::vector<std::vector<double>> m, int row) {
            // initialise input layer from data Matrix
            for (int i=0;i<m[0].size();i++) {
                layers[0].nodes[i].value = m[row][i];
            }      
            
            // calculates the values for all units in forward pass
            for (int i=1;i<layers.size();i++) {
                layers[i].predict(layers[i-1]);
            }            
        }


        /**
         * trains the weights of each node using gradient descent
         * gradients are calculated for every instance of data before
         * weights are updated
         */
        void fit(std::vector<std::vector<double>> m, 
            std::vector<std::vector<int>> labels) {

            // gradient descent
            for (int epoch=0;epoch<EPOCHS;epoch++) {

                // loss is printed intermittently
                if (epoch % ((EPOCHS + EPOCH_PRINTS -  1) / EPOCH_PRINTS) == 0) {
                    std::cout << "Epoch: " << std::setfill(' ') << 
                        std::setw(PRINT_PADDING) << epoch << " | ";
                    loss(m, labels);
                }

                // loop thorugh all data instances
                for (int i=0;i<m.size();i++) {
                    // forward pass on instance
                    predictInstance(m, i);

                    // calc gradients for output layer
                    layers[layers.size()-1].calculateGradients(
                        layers[layers.size()-2], labels[i]);

                    // calc gradients for other layers
                    for (int j=layers.size()-2;j>0;j--) {
                        layers[j].calculateGradients(layers[j-1], layers[j+1]);
                    } 
                }

                // update weights
                for (int i=0;i<layers.size();i++) {
                        layers[i].updateWeights();
                } 
            }

            // print final loss
            std::cout << "Epoch: " << std::setfill(' ') << 
                std::setw(PRINT_PADDING) << EPOCHS << " | ";
            loss(m, labels);

            // print final weights
            std::cout << "FINAL WEIGHTS" << std::endl;
            for (int i=0;i<layers.size();i++) {
                    layers[i].printWeights();
            } 
        }


        /**
         * calculate loss function value and accuracy over all data 
         */
        void loss(std::vector<std::vector<double>> m, 
            std::vector<std::vector<int>> labels) {
            
            std::vector<std::vector<double>> result = predict(m);

            double loss = 0, rowMax;
            float accuracy = 0;
            int labelIndex, rowMaxIndex;

            // iterate through instances 
            for (int i=0;i<labels.size();i++) {
                rowMax = 0;
                rowMaxIndex = -1;
                labelIndex = -1;

                for (int j=0;j<labels[0].size();j++) {
                    // calculate cross entropy loss
                    loss -= labels[i][j] * log2(result[i][j]);

                    // get label index and max result index 
                    if (labels[i][j] == 1) labelIndex = j;
                    if (result[i][j] > rowMax) {
                        rowMax = result[i][j];
                        rowMaxIndex = j;  
                    }
                }
                
                // predicted label is correct, increase accuracy 
                if (labelIndex == rowMaxIndex) accuracy++;
            }

            loss /= labels.size();
            accuracy /= labels.size();

            std::cout << "loss: "<< loss << " - accuracy: " 
                << accuracy << std::endl;
        }
};


/**
 * testing the MLP by learning on the IRIS dataset
 */
int main(int argc, char **argv){
    // build model architecture
    std::vector<int> layerSizes = {4,4,3};
    MLP mlp(layerSizes);

    // read in iris data
    std::string line;
    std::ifstream infile("iris.txt");
    std::vector<std::vector<double>> m;
    std::vector<std::vector<int>> labels;

    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string in_line;
        std::vector<double> row;
        // lavels are one hot encoded
        std::vector<int> label(3, 0);

        while (getline(ss, in_line, ',')) {
            row.push_back(std::stod(in_line, 0));
        }

        // last element of row is the label
        
        label[row.back()] = 1;
        labels.push_back(label);
        row.pop_back();

        m.push_back(row);
    }

    // train the model
    std::cout << "LEARNING CURVE" << std::endl;
    mlp.fit(m, labels);

    // forward pass
    std::vector<std::vector<double>> out = mlp.predict(m);

    // print predictions
    std::cout << "PREDICTIONS ON THE TRAINING SET" << std::endl;
    for (int i=0;i<out.size();i++) {
        std::cout << "[" << labels[i][0] << ", " << labels[i][1] 
            << ", " << labels[i][2] << "] -> ";
        std::cout << "[" << out[i][0] << ", " << out[i][1] 
            << ", " << out[i][2] << "]";
        std::cout << std::endl;
    }
 
    return 0;
}
