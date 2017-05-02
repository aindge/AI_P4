import math
import numpy as np
import random as r
import matplotlib.pyplot as plt
from sys import argv

SIGMA = 0.5                   # Sigma constant for sigmoid functions (0.5 for basic sigmoid)
TRAINING_LENGTH = 5        # Number of times to train in an epoch
NUM_EPOCHS = 300             # Number of epochs to train
BUF_MAX = 0.2                 # When normalizing weights, specifies maximum bound on padding from 0 and 1
SAMPLE_INPUT = [[1,0.5,-1]]   # Sample input
TYPE1_INTERPRETATIONS = ['A', 'B', 'C', 'D', 'E', 'J', 'K'] # Interpretations for type1 classes.

if(int(argv[1]) == 1):
    INPUT_SIZE = 63               # Size of input vector
    OUTPUT_SIZE = 7               # Size of output vector
elif(int(argv[1]) == 2):
    INPUT_SIZE = 9
    OUTPUT_SIZE = 1
elif(int(argv[1]) == 3):
    INPUT_SIZE = 34
    OUTPUT_SIZE = 1
else:
    print("Usage: " + argv[0] + " <test> <neurons>")
    print("    test: One of the following:")
    print("        1: Letter Recognition")
    print("        2: Breast Cancer Wisconsin")
    print("        3: Ionosphere")
    print("    neurons: The number of neurons in the hidden layer")
    exit(1)

def get_input_layer():
    result = []
    for i in range(INPUT_SIZE):
        result.append(r.random())
    return result

def get_type1_input(filename):
    '''Read train1.txt into array for use.'''
    input_array = []
    text = open(filename).read().replace('\n', '')
    i = 0 # Letter
    j = 0 # Position in letter
    result = []
    for c in text:
        if j == 0:
            result.append([])
        if c == ".":
            result[i].append(-1)
        elif c == "#":
            result[i].append(1)
        else:
            result[i].append(0)
        if j == 62:
            j = 0
            i += 1
        else:
            j += 1
    
    # Check just in case...
    for subary in result:
        if len(subary) != 63:
            raise RuntimeError("One of the subarrays is of inappropriate size. (Expected 63, got " + str(len(subary)) + ".)")
    return (np.array(result), [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6])

def validate_input_test1(out, i, key):
    q = np.argmax(out) == i
    # print("Matched " + TYPE1_INTERPRETATIONS[np.argmax(out)] + " with " + TYPE1_INTERPRETATIONS[i] + " (" + str(out) + ")")
    return q

def get_loss_test1(out, i, key):
    expected = np.ones((1, OUTPUT_SIZE))
    expected[0][i] = -1
    expected = expected * -1
    return expected - out

def get_type2_input(filename):
    lines = open(filename).readlines()
    result = []
    key = []

    # Shuffle the list
    for i in range(len(lines)):
        t = lines.pop(int(r.random() * (len(lines)-i)))
        t = t.replace('?', '0')
        lines.append(t)

    # Make the list
    for line in lines:
        split_line = line.split(',')
        result.append(split_line[1:10])
        if(int(split_line[10]) == 4):
            key.append(1)
        elif(int(split_line[10]) == 2):
            key.append(-1)
        else:
            raise ValueError("There was a weird value in " + line + ", got " + split_line[10])
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = int(result[i][j])
    return (np.array(result[:int(len(result)/2)]), np.array(result[int(len(result)/2):]), key[:int(len(result)/2)], key[int(len(result)/2):])

def validate_input_test2(out, i, key):
    return math.copysign(1, out[0]) == key

def get_loss_test2(out, i, key):
    return key - out

def get_type3_input(filename):
    lines = open(filename).readlines()
    result = []
    key = []

    # Shuffle the list
    for i in range(len(lines)):
        t = lines.pop(int(r.random() * (len(lines)-i)))
        t = t.replace('?', '0')
        lines.append(t)

    # Make the list
    for line in lines:
        split_line = line.split(',')
        result.append(split_line[0:34])
        if(split_line[34].strip() == 'g'):
            key.append(1)
        elif(split_line[34].strip() == 'b'):
            key.append(-1)
        else:
            raise ValueError("There was a weird value in " + line + ", got " + split_line[34])
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = float(result[i][j])
    return (np.array(result[:int(len(result)/2)]), np.array(result[int(len(result)/2):]), key[:int(len(result)/2)], key[int(len(result)/2):])

def validate_input_test3(out, i, key):
    return math.copysign(1, out[0]) == key

def get_loss_test3(out, i, key):
    return key - out

def sigmoid(x):
    '''Take sigmoid function of x'''
    return math.tanh(SIGMA * x)

def ddx_sigmoid(x):
    '''Take derivative of sigmoid function of x'''
    return (2*SIGMA) / (math.cosh(2*SIGMA*x) + 1)

def generate_random_weights(layer1_size, layer2_size):
    '''Generate normalized weights for a given size'''
    weights = np.random.normal(0, 1, [layer1_size, layer2_size])
    buf = r.random()*BUF_MAX
    weights = weights - np.amin(weights)
    weights = weights / np.amax(weights) * (1 - buf * 2)
    weights = weights + buf
    return weights

def feed_forward(weights, in_vector):
    '''Run a network described by weights for the given input vector'''
    in_v = np.reshape(in_vector, (1,INPUT_SIZE))      # Force input to be correct size
    intermediate1 = np.dot(in_v, weights[0])          # Get sums for hidden layer
    v_sig = np.vectorize(sigmoid)                     # Create a vectorized sigmoid function that operates elementwise
    hidden_layer = v_sig(intermediate1)               # Apply sigmoid function to hidden-layer sums
    intermediate2 = np.dot(hidden_layer, weights[1])  # Get sums for output layer
    output_layer = v_sig(intermediate2)               # Apply sigmoid function to output sums
    return (intermediate1, hidden_layer, intermediate2, output_layer)  # Done!

def backprop(weights, in_vector, sums1, hidden_layer, sums2, out, loss):
    new_weights = [None, None]
    v_ddx = np.vectorize(ddx_sigmoid)
    d_outs = np.multiply(v_ddx(sums2), loss)
    d_out_weights = np.transpose(np.dot(np.transpose(d_outs), hidden_layer))
    new_weights[1] = weights[1] + d_out_weights
    d_hidden = np.multiply(weights[1], v_ddx(sums2))
    d_hidden_sum = np.dot(d_outs, np.transpose(d_hidden))
    d_hidden_weights = np.dot(np.transpose([in_vector]), d_hidden_sum)
    new_weights[0] = weights[0] + d_hidden_weights
    return new_weights

def train(training, _weights, training_key):
    weights = _weights
    for i in range(TRAINING_LENGTH):
        test_num = int(r.random()*len(training))
        in_vector = training[test_num]
        (sums1, hidden_layer, sums2, out) = feed_forward(weights, in_vector)
        loss = None
        if int(argv[1]) == 1:
            loss = get_loss_test1(out, test_num % 7, training_key[test_num])
        elif int(argv[1]) == 2:
            loss = get_loss_test2(out, test_num, training_key[test_num])
        elif int(argv[1]) == 3:
            loss = get_loss_test3(out, test_num, training_key[test_num])
        weights = backprop(weights, in_vector, sums1, hidden_layer, sums2, out, loss)
    return weights
    
def test(tests, weights, test_key):
    num_correct = 0
    for i in range(len(tests)):
        test_num = i
        in_vector = tests[test_num]
        (sums1, hidden_layer, sums2, out) = feed_forward(weights, in_vector)
        if int(argv[1]) == 1:
            if validate_input_test1(out, test_num % 7, test_key[test_num]):
                num_correct += 1
        elif int(argv[1]) == 2:
            if validate_input_test2(out, test_num % 7, test_key[test_num]):
                num_correct += 1
        elif int(argv[1]) == 3:
            if validate_input_test3(out, test_num % 7, test_key[test_num]):
                num_correct += 1
    return num_correct
    

if __name__ == "__main__":
    (training, tests) = (None, None) #Initialize
    if(int(argv[1]) == 1):
        (training, training_key) = get_type1_input("train1.txt")
        (tests, tests_key) = get_type1_input("test1.txt")
    elif(int(argv[1]) == 2):
        (training, tests, training_key, tests_key) = get_type2_input("breast_cancer_wisconsin.txt")
    else:
     (training, tests, training_key, tests_key) = get_type3_input("ionosphere.txt")

    # Generate random initial weights based on a normal distribution
    hidden_layer_size = int(argv[2])
    weights = []
    weights.append(generate_random_weights(INPUT_SIZE, hidden_layer_size))  # Input-to-hidden
    weights.append(generate_random_weights(hidden_layer_size, OUTPUT_SIZE)) # Hidden-to-output

    scores = []

    for i in range(NUM_EPOCHS):
        weights = train(training, weights, training_key)
        num_correct = test(tests, weights, tests_key)
        scores.append(num_correct)
        print("Epoch " + str(i) + ": The machine got " + str(num_correct) + " out of " + str(len(tests)) + " correct. \n\tAccuracy: " + str(100*num_correct/len(training)) + "%")

    plt.plot(np.arange(NUM_EPOCHS), scores, linewidth=3)
    plt.ylim([0, len(tests)])
    plt.show()
    exit(0)