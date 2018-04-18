# This is an AND test

import numpy as np


learning_rate = 1
epochs = 5000


def sigmoid_function(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivate(x):
    return x * (1 - x)


input = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
    ]
)

output = np.array([
    [0],
    [1],
    [1],
    [0]
])


input_layer_n = input.shape[1]

hidden_layer_n = 2

output_layer_n = output.shape[1]

# Initialise the hidden layer
hidden_layer_weights = np.random.uniform( size = (input_layer_n, hidden_layer_n) )
hidden_layer_bias = np.random.uniform( size = (1, hidden_layer_n) )

# Initialize the output layer
output_layer_weights =  np.random.uniform(size = (hidden_layer_n, output_layer_n))
output_layer_bias = np.random.uniform(size = (1, output_layer_n))


# Start of learning 
for i in range(epochs):

    #Forward!
    hidden_layer_input = input.dot(hidden_layer_weights) + hidden_layer_bias
    hidden_layer_activation = sigmoid_function(hidden_layer_input)
    output_layer_input = hidden_layer_activation.dot(output_layer_weights) + output_layer_bias
    model_output = sigmoid_function(output_layer_input)

    if i == 0:
        print("\nIntial!\nOutput") 
        print(model_output)

    '''
    print("\n\nHidden Layer Input")
    print(hidden_layer_input)
    print("\n\nHidden Layer activation")
    print(hidden_layer_activation)
    print("\n\nOutput Layer Inputs")
    print(output_layer_input)
    print("\nOutput")
    '''
    #Backprop
    error = output - model_output

    output_layer_gradient = sigmoid_derivate(model_output)
    output_layer_delta = error * output_layer_gradient

    # Back prop error to Hidden layer
    hidden_layer_error = output_layer_delta.dot(output_layer_weights.T)

    hidden_layer_gradient = sigmoid_derivate(hidden_layer_activation)
    
    hidden_layer_delta = hidden_layer_error * hidden_layer_gradient

    #Update Weights
    hidden_layer_weights += input.T.dot(hidden_layer_delta) * learning_rate
    hidden_layer_bias += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

    output_layer_weights += hidden_layer_activation.T.dot(output_layer_delta) * learning_rate
    output_layer_bias += np.sum(output_layer_delta, axis=0, keepdims=True) * learning_rate

    #print("Model output at epoch %d" % i)
    #print model_output

print("\nFinal!") 
print model_output

print("\nReal!")
print output
    
    