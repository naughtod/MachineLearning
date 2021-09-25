#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h> 

/**
 * Multilayer Perceptron (MLP) implementation in C++ by David Naughton.
 * Uses an object oriented approach with nodes and layers as opposed to
 * matrices which makes it less efficient but hopefully easy to understand.
 * 
 * All layers use sigmoid activation function, standard gradient descent 
 * optimisation, all layers except output have a bias, learning rate is 0.2,
 * epochs is 4000. 
 */

using namespace std;
#define LR 0.2

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
            
            leftWeights.resize(leftUnits, 0);

            for (int i=0;i<leftUnits;i++) {
                leftWeights[i] = rand()/(double)RAND_MAX;
            }
            
            gradients.resize(leftUnits, 0);
        }

        /**
         * Update weights and then clear gradients for next epoch
         */
        void updateWeights() {            
            // new weight = old weight - LR * gradient
            for (int i=0;i<leftWeights.size();i++) {
                leftWeights[i] -= LR * gradients[i];
            }

            // clear gradients
            std::fill(gradients.begin(), gradients.end(), 0);
        }

        /**
         * prints the weights of node
         */
        void printWeights() {
            for (int i=0;i<leftWeights.size();i++) {
                std::cout << "node: " << idInLayer << " " <<
                    "weight "<< i << ": " << leftWeights[i] << std::endl;
            }
        }

        // methods implemented beneath layer class declaration
        void predict(Layer leftLayer);
        void calculateGradients(Layer leftLayer, double label);
        void calculateGradients(Layer leftLayer, Layer rightLayer);
};

/**
 * layer class contains nodes
 */ 
class Layer {
    public:
        int layerId;
        std::vector<Node> nodes;

        /**
         *  constructor creates nodes for layer
         */
        Layer(int units, int unitsLeft, bool bias, int layerId) {
            this->layerId = layerId;

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
            for (int i=0;i<nodes.size();i++) {
                nodes[i].predict(layer);
            }
        }

        /**
         * runs update functions for every node in layer
         */
        void calculateGradients(Layer leftLayer, double label) {
            for (int i=0;i<nodes.size();i++) {
                nodes[i].calculateGradients(leftLayer, label);
            }
        }

        /**
         * runs update functions for every node in layer
         */
        void calculateGradients(Layer leftLayer, Layer rightLayer) {
            for (int i=0;i<nodes.size();i++) {
                nodes[i].calculateGradients(leftLayer, rightLayer);
            }
        }

        /**
         * runs update functions for every node in layer
         */
        void updateWeights() {
            for (int i=0;i<nodes.size();i++) {
                nodes[i].updateWeights();
            }
        }

        /**
         * prints the weights of nodes in layer
         */
        void printWeights() {
            std::cout << "layerId: " << layerId << std::endl;
            for (int i=0;i<nodes.size();i++) {
                nodes[i].printWeights();
            }
        }

};

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
 * loss function derivative for Mean Squared Error MSE 0.5 * (label - value) ^ 2 
 */
double lossDerivative(double value, double label) {
    return value - label;
}

/**
 * Calculate gradients cumulatively for hidden layers
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
 * Calculate gradients cumulatively for output layer
 */
void Node::calculateGradients(Layer leftLayer, double label) {
    delta = sigmoidDerivative(value) * lossDerivative(value, label);
    
    for (int i=0;i<leftWeights.size();i++)
        gradients[i] += leftLayer.nodes[i].value * delta;
}


/**
 * calculate the value of the node by taking the sigmoid of the inner product of
 * connected weights and values from previous layer
 */ 
void Node::predict(Layer leftLayer) {
    // don't run for bias or input nodes
    if (leftWeights.size() > 0) {
        double innerProduct = 0;

        // calculate inner product of input values and weights
        for (int i=0;i<leftLayer.nodes.size();i++) {
            innerProduct += leftWeights[i] * leftLayer.nodes[i].value;
        }

        value = sigmoid(innerProduct);
    }
}


/**
 * class for 2D double array to hold data
 */ 
class Matrix {
    public:
        int height, width;
        double *d;

        Matrix(int height, int width) {
            d = new double[height * width];
            this->height = height;
            this->width = width;
        }

        double get(int row, int col) {
            return d[row + col * height];
        }
        
        void set(int row, int col, double val) {
            d[row + col * height] = val;
        }

        void del() {
            delete[] d;
        }
};


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
        Matrix* predict(Matrix m) {
            int outputUnits = layers[layers.size()-1].nodes.size();
            Matrix *result = new Matrix(m.height, outputUnits);
            
            // calculate prediction for each instance
            for (int i=0;i<m.height;i++) {
            
                predictInstance(m, i);
                
                // transfer the values from the final layer to the result
                for (int j=0;j<outputUnits;j++) 
                    result->set(i, j, layers[layers.size()-1].nodes[j].value);
            }

            return result;
        }


        /**
         * for a single instance will calculate output units via
         * a forward pass through network
         */
        void predictInstance(Matrix m, int row) {
            // initialise input layer from data Matrix
            for (int i=0;i<m.width;i++) {
                layers[0].nodes[i].value = m.get(row,i);
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
        void fit(Matrix m, std::vector<int> labels) {
            // gradient descent
            for (int epoch=0;epoch<4000;epoch++) {

                // loss is printed intermittently
                if (epoch % 500 == 0) {
                    std::cout << "Epoch: " << std::setfill(' ') << 
                        std::setw(4) << epoch << " | ";
                    loss(m, labels);
                }

                // loop thorugh all data instances
                for (int i=0;i<m.height;i++) {
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

            // print final weights
            for (int i=0;i<layers.size();i++) {
                    layers[i].printWeights();
            } 
        }


        /**
         * calculate loss function MSE value over all data 
         */
        void loss(Matrix m, std::vector<int> labels) {
            Matrix *result = predict(m);
            double loss=0;

            // calculate loss 
            for (int i=0;i<m.height;i++) {
                for (int j=0;j<result->width;j++)
                    loss += (labels[i] - result->get(i,j)) * 
                        (labels[i] - result->get(i,j));
            }

            loss /= m.height;
            
            std::cout << "loss: "<< loss << std::endl;
        }
};


/**
 * testing the MLP by learning the XOR function
 */
int main(int argc, char **argv){
    // build model architecture
    std::vector<int> layerSizes = {2,2,1};
    MLP *mlp = new MLP(layerSizes);

    // set up XOR training data
    // [[0,0],[0,1],[1,0],[1,1]]
    Matrix *m = new Matrix(4,2);
    m->set(0,0,0);
    m->set(0,1,0);
   
    m->set(1,0,0);
    m->set(1,1,1);
    
    m->set(2,0,1);
    m->set(2,1,0);

    m->set(3,0,1);
    m->set(3,1,1);

    // label data for XOR training 
    // [0,0]->[0],[0,1]->[1],[1,0]->[1],[1,1]->[0]
    std::vector<int> labels = {0,1,1,0};

    // print training data
    std::cout << "Training Data:" << std::endl;
    for (int i=0;i<m->height;i++) {
        std::cout << "[";
        for (int j=0;j<m->width;j++) {
            std::cout << m->get(i,j) ;
        }
        std::cout << "] : " << labels[i] << std::endl;
    }

    // train the model
    mlp->fit(*m, labels);

    // forward pass
    Matrix *out = mlp->predict(*m);

    // print prediction
    for (int i=0;i<out->height;i++) {
        std::cout << "[" << m->get(i,0) << ", " << m->get(i,1) << "] -> ";
        for (int j=0;j<out->width;j++)
            std::cout << out->get(i,j) << std::endl;
    }

    // delete data matrix
    m->del();
 
    return 0;
}
