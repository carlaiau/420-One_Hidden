import sys
import numpy as np

def sigmoid_function(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivate(x):
    return x * (1 - x)

def relu_function(x):                
    return np.maximum(0,x)

def relu_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

class One_Hidden:
    # Constructor 
    def __init__(self, folder="", model="s"):
        '''
        Each different type of models input files are put within a folder,
        You can pass in a folder agrument to the script, if no folder argument is
        passed in, then the input files are grabbed from the current folder.
        '''
        input_filename = "in.txt"
        output_filename = "out.txt"
        param_filename = "param.txt"
        if(folder != ""):
            input_filename = folder.strip('/') + "/" + input_filename
            output_filename = folder.strip('/') + "/" + output_filename
            param_filename = folder.strip('/') + "/" +  param_filename

        with open(input_filename, "r") as f:
            raw_input = []
            for line in f:
                if line.split()[0].find('.') > -1:
                    raw_input.append( map(float, line.split()) )
                else:
                    raw_input.append( map(int, line.split()) )
        self.input = np.asarray(raw_input)

        with open(output_filename, "r") as f:
            raw_output = []
            for line in f:
                if line.split()[0].find('.') > -1:
                    raw_output.append( map(float, line.split()) )
                else:
                    raw_output.append( map(int, line.split()) )
        self.output = np.asarray(raw_output)

        # Number of I/O samples (Used for population error calc)
        self.io_pairs = self.input.shape[0]
        # Define Parameters
        f = open(param_filename, "r")
        lines = f.readlines()
        self.input_layer_n = int(lines[0].rstrip('\n'))
        self.hidden_layer_n = int(lines[1].rstrip('\n'))
        self.output_layer_n = int(lines[2].rstrip('\n'))
        self.learning_rate = float(lines[3].rstrip('\n'))
        self.momentum_rate = float(lines[4].rstrip('\n'))
        self.error_criterion  = float(lines[5].rstrip('\n'))
        
        f.close()

        self.prev_output_weight_change = 0
        self.prev_output_bias_change = 0
        self.prev_hidden_weight_change= 0
        self.prev_hidden_bias_change = 0
        # Initialize the hidden layer
        self.hidden_layer_weights = np.random.uniform(size=(self.input_layer_n, self.hidden_layer_n))
        self.hidden_layer_bias = np.random.uniform(size=(1, self.hidden_layer_n))
        # Initialize the output layer
        self.output_layer_weights = np.random.uniform(size = (self.hidden_layer_n, self.output_layer_n))
        self.output_layer_bias = np.random.uniform(size=(1, self.output_layer_n))
        
        self.model = model
        self.epochs = 0

    def learn(self, epochs = 5000):
        solved = False
        epochs = int(epochs)
        for e in range(epochs):
            #Feed Forward
            hidden_layer_input = self.input.dot(self.hidden_layer_weights) + self.hidden_layer_bias

            if self.model == "s":
                hidden_layer_activation = sigmoid_function(hidden_layer_input)
            elif self.model == "r":
                hidden_layer_activation = relu_function(hidden_layer_input)

            output_layer_input = hidden_layer_activation.dot(self.output_layer_weights) + self.output_layer_bias

            if self.model == "s":
                self.model_output =  sigmoid_function(output_layer_input)
            elif self.model == "r":
                self.model_output =  relu_function(output_layer_input)
                
            #Determine Error
            error = self.output - self.model_output
            population_error = 0.5 * np.sum( error**2 ) / (self.output_layer_n * self.io_pairs)
                
            # Backpropogate
            if(population_error > self.error_criterion):
                if self.model == "s":
                    output_layer_gradient = sigmoid_derivate(self.model_output)
                    hidden_layer_gradient = sigmoid_derivate(hidden_layer_activation)
                
                elif self.model == "r":
                    output_layer_gradient = relu_derivative(self.model_output)
                    hidden_layer_gradient = relu_derivative(hidden_layer_activation)
                
                output_layer_delta = error * output_layer_gradient
                hidden_layer_error = output_layer_delta.dot(self.output_layer_weights.T)
                hidden_layer_delta = hidden_layer_error * hidden_layer_gradient
                    
                # Update Weights
                output_weight_change = hidden_layer_activation.T.dot(output_layer_delta) * self.learning_rate
                output_weight_change += self.prev_output_weight_change * self.momentum_rate
                self.output_layer_weights += output_weight_change

                output_bias_change = np.sum(output_layer_delta, axis=0, keepdims=True) * self.learning_rate
                output_bias_change += self.prev_output_bias_change * self.momentum_rate
                self.output_layer_bias += output_bias_change

                hidden_weight_change = self.input.T.dot(hidden_layer_delta) * self.learning_rate
                hidden_weight_change += self.prev_hidden_weight_change * self.momentum_rate
                self.hidden_layer_weights += hidden_weight_change

                hidden_bias_change = np.sum(hidden_layer_delta, axis=0, keepdims=True) * self.learning_rate
                hidden_bias_change += self.prev_hidden_bias_change * self.momentum_rate
                self.hidden_layer_bias += hidden_bias_change

                # Remember n-1 for momentum
                self.prev_output_weight_change = output_weight_change
                self.prev_output_bias_change =  output_bias_change
                self.prev_hidden_weight_change = hidden_weight_change
                self.prev_hidden_bias_change = hidden_bias_change
            else:
                # Solution found
                solved = True
                print("Error Criterion reached at epoch: %d" % (self.epochs + e + 1)  )
                self.epochs = 0 
                break

        if not solved:
            self.epochs += epochs
            
                        
    def print_weights(self):
        print("\nHidden Layer Weights")
        print(self.hidden_layer_weights)
        print("\nHidden Layer Bias")
        print(self.hidden_layer_bias)
        print("\nOutput Layer Weights")
        print(self.output_layer_weights)
        print("\nOutput Layer Bias")
        print(self.output_layer_bias)                        

    def print_model_output(self):
        print("\nModel Output")
        print(self.model_output)

    def reset_weights(self):
        self.prev_output_weight_change = 0
        self.prev_output_bias_change = 0
        self.prev_hidden_weight_change= 0
        self.prev_hidden_bias_change = 0
        # Initialize the hidden layer
        self.hidden_layer_weights = np.random.uniform(size=(self.input_layer_n, self.hidden_layer_n))
        self.hidden_layer_bias = np.random.uniform(size=(1, self.hidden_layer_n))
        # Initialize the output layer
        self.output_layer_weights = np.random.uniform(size = (self.hidden_layer_n, self.output_layer_n))
        self.output_layer_bias = np.random.uniform(size=(1, self.output_layer_n))
        self.historial_epochs_for_iteration = 0

if __name__== "__main__":
    folder = raw_input("Please input the model/folder you want to run from. Leave blank for current folder\n")
    model = raw_input("\nPlease choose the activation function you want to use\n"
    "s: Sigmoid\n"
    "r: Relu\n"
    "mr: Modified Relu\n"
    "mr2: Modified Relu 2\n\n")
    
    one_h = One_Hidden(folder, model)
    print("\nModel Initalised")
    option = 10
    while(option != "0"):
        try:
            option = raw_input(
                "\nPlease choose option\n" 
                "1: Teach (100 epochs)\n"
                "2: Teach (to criteria)\n" 
                "3: Test (only applicable to Iris set)\n" 
                "4: Show weights\n"
                "5: Show Output\n"
                "6: Reset weights\n"
                "0: Quit\n\n"
            )
            if(option == "0"):
                print ("Thanks for playing!")
                break
            
            if(option == "1"):
                one_h.learn(epochs = 100)
            
            if(option == "2"):
                one_h.learn()
            
            if(option == "3"):
                print("Still need to build tester")
            
            if(option == "4"):
                one_h.print_weights()
            
            if(option == "5"):
                one_h.print_model_output()
            
            if(option == "6"):
                one_h.reset_weights()
                
        except EOFError:
            print ("Thanks for playing!")
            break
