import argparse
import numpy as np

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

input_args = argparse.ArgumentParser()
input_args.add_argument('--folder', '-f', 
    help="Folder that contains the input files. Leave blank to read in, out, param from folder of execution", 
    type = str
)
input_args.add_argument('--epochs', '-e', help="Number of Epochs. Defaults to 3000", type = int, default = 3000)
input_args.add_argument('--number', '-n', help="Number of Iterations. Defaults to 1", type = int, default = 1)
input_args.add_argument('--generalise', '-g', 
    help="Split into Test/Training sample. Define the Test percentage as a float. Only applicable for Iris set", 
    type= float, 
    default = 0
)
input_args.add_argument('--online', '-o', help="Online Learning. Switch from Batch. Boolean flag", type = int, default = 0, nargs='?', const=1)
input_args.add_argument('--print_error', '-pe', help="Print Error per 100 epochs. Boolean flag", type = int, default = 0, nargs='?', const=1)

arguments =  input_args.parse_args()

'''
Each different type of models input files are put within a folder,
You can pass in a folder agrument to the script, if no folder argument is
passed in, then the input files are grabbed from the current folder.
'''
input_filename = "in.txt"
output_filename = "out.txt"
param_filename = "param.txt"
# IF you use auto complete on command line, the suffixed / will be included
if(arguments.folder):

    input_filename = arguments.folder.rstrip('/') + "/" + input_filename
    output_filename = arguments.folder.rstrip('/') + "/" + output_filename
    param_filename = arguments.folder.rstrip('/') + "/" +  param_filename

with open(input_filename, "r") as f:
    raw_input = []
    for line in f:
        if line.split()[0].find('.') > -1:
            raw_input.append( map(float, line.split()) )
        else:
            raw_input.append( map(int, line.split()) )
input = np.asarray(raw_input)

with open(output_filename, "r") as f:
    raw_output = []
    for line in f:
        if line.split()[0].find('.') > -1:
            raw_output.append( map(float, line.split()) )
        else:
            raw_output.append( map(int, line.split()) )

output = np.asarray(raw_output)

# Number of I/O samples (Used for population error calc)
io_pairs = input.shape[0]

'''
You can pass in a second argument to change the epoch count 
otherwise this defaults to 3000
'''
epochs = arguments.epochs


# Define Parameters
f = open(param_filename, "r")
lines = f.readlines()
input_layer_n = int(lines[0].rstrip('\n'))
hidden_layer_n = int(lines[1].rstrip('\n'))
output_layer_n = int(lines[2].rstrip('\n'))
learning_rate = float(lines[3].rstrip('\n'))
momentum_rate = float(lines[4].rstrip('\n'))
error_criterion  = float(lines[5].rstrip('\n'))
f.close()



def sigmoid_function(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivate(x):
    return x * (1 - x)

def online_learn():
    print("Not done yet!")

def batch_learn():
    number_solved = 0
    total_epochs = 0
    for a in range(arguments.number):
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
        prev_hidden_bias_change = 0
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
            if arguments.print_error:
                if(i % 100 == 0):
                    print("Epoch %d: %f" % (i, population_error) )
                    

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
            print("\nSolved at epoch %d\nfinal model:" % i)
            print(model_output)
            print("\n")
    if(number_solved > 0):
        print("Solved %d out of %d, average epochs required for solution: %f" % (number_solved, a + 1, total_epochs/number_solved))
    else:
        print("None solved out of %d iterations" %  (a + 1))



if arguments.online:
    online_learn()
else:
    batch_learn()





