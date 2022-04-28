# MLP (Multi- Layer Perceptron)

Modified from https://github.com/AgrawalAmey/nnfl-resources

## Study and optimize an implementation of a Multi-Layer Perceptron from scratch

    We start from an implementation of a MPL with vanilla Back-Propagation Learning using sigmoidal units.
    The cost function is the root mean square
    Input data are normalized before training
    It is trained on 5 synthetic bidimensional data sets
    No cross-validation is implemented.
    You may prepare a different notebook for each item.

    Part A :

    For each data set find the minimal NN architecture. You should set the number of layers, the number of neurons for each layer, the initialization of weights and bias, the dimension of the mini-batchs, the number of iterations, the learning rate. 
    Plot the learning curve
    Augment the input layer in order to improve learning (e.g., introduce a new synthetic input as the sum of the squares of the two actual inputs:  𝑧=𝑥2+𝑦2
    
    Part B: (at least one of the following)

    Increase the size of your NN and implement weigth decay. 
    Implement drop-out. 
    Implement layers normalization. 
    Implement train with noise. 
    Change the activation function and its derivative (e.g., try with ReLu). 
