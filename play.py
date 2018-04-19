# This is XOR


import numpy as np

momentum_rate = 0.9
learning_rate = 0.1
epochs = 3000

error_criterion = 0.02

def sigmoid_function(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivate(x):
    return x * (1 - x)


'''
Params.txt includes:
input n
hidden n
output n
learning constant
momentum constant
learning/error criterion
'''

# Input needs to be loaded in from in.text
input = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
    ]
)
# output needs to be loaded in from output.text
output = np.array([
    [0],
    [1],
    [1],
    [0]
])


input_layer_n = input.shape[1]
hidden_layer_n = 2
output_layer_n = output.shape[1]

io_pairs = input.shape[0]


number_solved = 0
total_epochs = 0
for a in range(50):
    # Initialise the hidden layer
    hidden_layer_weights = np.random.uniform(size=(input_layer_n, hidden_layer_n))
    hidden_layer_bias = np.random.uniform(size=(1, hidden_layer_n))

    # Initialize the output layer
    output_layer_weights = np.random.uniform(size=(hidden_layer_n, output_layer_n))
    output_layer_bias = np.random.uniform(size=(1, output_layer_n))
    # Start of learning
    prev_output_weight_change = 0
    prev_output_bias_change = 0
    prev_hidden_weight_change = 0
    prev_hidden_bias_change = 0;
    for i in range(epochs):

        #Feed Forward
        hidden_layer_input = input.dot(hidden_layer_weights) + hidden_layer_bias
        hidden_layer_activation = sigmoid_function(hidden_layer_input)
        output_layer_input = hidden_layer_activation.dot(output_layer_weights) + output_layer_bias
        model_output =  sigmoid_function(output_layer_input)

        #Determine Error
        error = output - model_output
        population_error = 0.5 * np.sum( error**2 ) / (output_layer_n * io_pairs)
        # Print Error
        '''if(i % 100 == 0):
            print("Epoch %d" % i)
            print(population_error)
        '''
        # Backpropogate
        if(population_error > error_criterion):
            solved = False
            output_layer_gradient = sigmoid_derivate(model_output)
            output_layer_delta = error * output_layer_gradient

            # Back prop error to Hidden layer
            hidden_layer_error = output_layer_delta.dot(output_layer_weights.T)
            hidden_layer_gradient = sigmoid_derivate(hidden_layer_activation)
            hidden_layer_delta = hidden_layer_error * hidden_layer_gradient

            # Update Weights
            output_weight_change = hidden_layer_activation.T.dot(output_layer_delta) * learning_rate
            output_weight_change += prev_output_weight_change * momentum_rate
            output_layer_weights += output_weight_change

            output_bias_change = np.sum(output_layer_delta, axis=0, keepdims=True) * learning_rate
            output_bias_change += prev_output_bias_change * momentum_rate
            output_layer_bias += output_bias_change

            hidden_weight_change = input.T.dot(hidden_layer_delta) * learning_rate
            hidden_weight_change += prev_hidden_weight_change * momentum_rate
            hidden_layer_weights += hidden_weight_change

            hidden_bias_change = np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate
            hidden_bias_change += prev_hidden_bias_change * momentum_rate
            hidden_layer_bias += hidden_bias_change

            # Remember n-1 for momentum
            prev_output_weight_change = output_weight_change
            prev_output_bias_change =  output_bias_change
            prev_hidden_weight_change = hidden_weight_change
            prev_hidden_bias_change = hidden_bias_change

        else:
            solved = True
            total_epochs += i
            number_solved += 1
            break

    if solved:
        print("Solved at epoch %d\nfinal model:" % i)
        print(model_output)



print("Solved %d out of %d, average epochs required for solution: %f" % (number_solved, a + 1, total_epochs/number_solved))
    
