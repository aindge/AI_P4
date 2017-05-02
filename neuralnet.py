import math
import numpy as np
import sys
import random as r
import matplotlib.pyplot as plt

INPUT_SIZE = 2     # Size of input vector
OUTPUT_SIZE = 1    # Size of output vector
BUF_MAX = 0.2        # Maximum size of padding away from 0 and 1 for initial normalized weights

def get_input_layer():
    result = []
    for i in range(INPUT_SIZE):
        result.append(r.random())
    return result

def sigmoid(x):
    return -1 + 2 / (1 + math.exp(-x))

def ddx_sigmoid(x):
    return 0.5 * (1 + sigmoid(x)) * (1 - sigmoid(x))

def generate_random_weights(layer1_size, layer2_size):
    '''Generate normalized weights for a given size'''
    weights = np.random.normal(0, 1, [layer1_size, layer2_size])
    _min = np.amin(weights)
    buf = r.random()*BUF_MAX
    weights = weights - _min
    weights = weights / np.amax(weights) * (1 - buf * 2)
    weights = weights + buf
    return weights
    

if __name__ == "__main__":

    #Generate random initial weights based on a normal distribution
    hidden_layer_size = int(sys.argv[1])
    weights = []
    weights.append(generate_random_weights(INPUT_SIZE, hidden_layer_size))  # Input-to-hidden
    weights.append(generate_random_weights(hidden_layer_size, OUTPUT_SIZE)) # Hidden-to-output
    print(weights[0])
    print(weights[1])