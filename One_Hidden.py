import sys
import numpy as np
np.set_printoptions(suppress=True) # Prevented Sci Notation as it was annoying

'''
All the functions that are used for feeding forward along with their derivative 
counter parts for back propogation 
'''
def sigmoid_function(x):
    return 1/(1 + np.exp(-x))

def sine_function(x):
    return np.sin(x)

def cosine_function(x):
    return np.cos(x)

def sigmoid_derivate(x):
    return x * (1 - x)

def relu_function(x, a):                
    return np.maximum(x, a * x)

def relu_derivative(x, a):
    if 1. * np.all(a < x):
        return 1
    return a



class One_Hidden:

    '''
    The Initalisation of the object.
    This is where all of the variables input into the UI build the inital state of the model
    '''
    def __init__(self, folder="", hidden_action="s", output_action="s", generalisation= False):
        '''
        Each different type of models input files are put within a folder,
        You can pass in a folder agrument to the script, if no folder argument is
        passed in, then the input files are grabbed from the current folder.
        '''
        self.generalisation = generalisation
        self.input_filename = "in.txt"
        self.output_filename = "out.txt"
        param_filename = "param.txt"
        if(folder != ""):
            self.input_filename = folder.strip('/') + "/" + self.input_filename
            self.output_filename = folder.strip('/') + "/" + self.output_filename
            param_filename = folder.strip('/') + "/" +  param_filename

        with open(self.input_filename, "r") as f:    
            
            '''
            Split Iris into training and test set
            first determine the number of samples, then grab 80% of these 
            randomly as the training set, and the other 20% as the test
            '''
            if self.generalisation:
                full_doc = f.readlines()
                indexs = np.arange(len(full_doc))
                np.random.shuffle(indexs)
                number_of_training = int(len(full_doc) * 0.8)

                training_indexs = indexs[:number_of_training]
                test_indexs = indexs[number_of_training:]
                training_input = []
                test_input = []
                for i in training_indexs:
                    if full_doc[i].split()[0].find('.') > -1:
                        training_input.append( map(float, full_doc[i].split()) )
                    else:
                        training_input.append( map(int, full_doc[i].split()) )

                for i in test_indexs:
                    if full_doc[i].split()[0].find('.') > -1:
                        test_input.append( map(float, full_doc[i].split()) )
                    else:
                        test_input.append( map(int, full_doc[i].split()) )
                self.input = np.asarray(training_input)
                self.test_input = np.asarray(test_input)

            else: # This isn't a generalisation set we don't need to worry about splitting
                raw_input = []
                for line in f:
                    if line.split()[0].find('.') > -1:
                        raw_input.append( map(float, line.split()) )
                    else:
                        raw_input.append( map(int, line.split()) )
                self.input = np.asarray(raw_input)


        with open(self.output_filename, "r") as f:

            '''
            Also split the output into test and training, using the same random indexs
            '''
            if self.generalisation:
                full_doc = f.readlines()
                training_output = []
                test_output = []

                for i in training_indexs:
                    if full_doc[i].split()[0].find('.') > -1:
                        training_output.append( map(float, full_doc[i].split()) )
                    else:
                        training_output.append( map(int, full_doc[i].split()) )

                for i in test_indexs:
                    if full_doc[i].split()[0].find('.') > -1:
                        test_output.append( map(float, full_doc[i].split()) )
                    else:
                        test_output.append( map(int, full_doc[i].split()) )
                self.output = np.asarray(training_output)
                self.test_output = np.asarray(test_output)     
            else:  # This isn't a generalisation set we don't need to worry about splitting
                raw_output = []
                for line in f:
                    if line.split()[0].find('.') > -1:
                        raw_output.append( map(float, line.split()) )
                    else:
                        raw_output.append( map(int, line.split()) )
                self.output = np.asarray(raw_output)

        # Number of I/O samples (Used for population error calc)
        self.io_pairs = self.input.shape[0]
        self.total_training_population_error = 0
        if self.generalisation:
            self.test_io_pairs = self.test_input.shape[0]
            self.total_test_population_error = 0
            self.all_solved_test_population_errors = []

        # Parameters read from param.txt
        f = open(param_filename, "r")
        lines = f.readlines()
        self.input_layer_n = int(lines[0].rstrip('\n'))
        self.hidden_layer_n = int(lines[1].rstrip('\n'))
        self.output_layer_n = int(lines[2].rstrip('\n'))
        self.learning_rate = float(lines[3].rstrip('\n'))
        self.momentum_rate = float(lines[4].rstrip('\n'))
        self.error_criterion  = float(lines[5].rstrip('\n'))
        
        f.close()

        # Initalisation of previous values for momentum calculation
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
        
        # Define which activation functions to used from UI input
        self.hidden_action = hidden_action.strip()
        self.output_action = output_action.strip()
        self.epochs = 0


    def learn(self, iterations = 1000, epochs = 10000):
        solved = False
        total_epochs = 0
        total_solved = 0
        for i in range(iterations):
            for e in range(epochs):

                '''
                Feed forward - Hidden Layer Activation
                '''
                hidden_layer_input = self.input.dot(self.hidden_layer_weights) + self.hidden_layer_bias

                if self.hidden_action == "r": # Relu
                    hidden_layer_activation = relu_function(hidden_layer_input, 0)
                elif self.hidden_action == "l": # Leaky Relu
                    hidden_layer_activation = relu_function(hidden_layer_input, 0.01)
                elif self.hidden_action == "v": # Very Leaky Relu
                    hidden_layer_activation = relu_function(hidden_layer_input, 0.15)
                elif self.hidden_action == "n": # Sine
                    hidden_layer_activation = sine_function(hidden_layer_input) 
                else: # Sigmoid
                    hidden_layer_activation = sigmoid_function(hidden_layer_input)
                

                '''
                Feed forward - Output Layer Activation
                '''
                output_layer_input = hidden_layer_activation.dot(self.output_layer_weights) + self.output_layer_bias

                if self.output_action == "n": # Sine
                    self.model_output =  sine_function(output_layer_input)   
                else: # Sigmoid
                    self.model_output =  sigmoid_function(output_layer_input)
                
                #Determine Error
                error = self.output - self.model_output
                population_error = 0.5 * np.sum( error**2 ) / (self.output_layer_n * self.io_pairs)
                    
                # We have not reached Error criterion yet
                if population_error > self.error_criterion:  
                    
                    '''
                    Backproporate - Hidden Layer Gradient
                    '''                   
                    if self.hidden_action == "r": # Relu
                        hidden_layer_gradient = relu_derivative(hidden_layer_activation, 0)
                    elif self.hidden_action == "l": # Leaky Relu
                        hidden_layer_gradient = relu_derivative(hidden_layer_activation, 0.01)
                    elif self.hidden_action == "v": # Very Leaky Relu
                        hidden_layer_gradient = relu_derivative(hidden_layer_activation, 0.15)
                    elif self.hidden_action == "n": # Sine
                        hidden_layer_gradient = cosine_function(hidden_layer_activation)
                    else: # Sigmoid
                        hidden_layer_gradient = sigmoid_derivate(hidden_layer_activation)

                    '''
                    Backproporate - Output Layer Gradient
                    '''                       
                    if self.output_action == "n": # Sine
                        output_layer_gradient = cosine_function(self.model_output)
                    else: # Sigmoid
                        output_layer_gradient = sigmoid_derivate(self.model_output)                        
                    
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

                    '''
                    Remember values for momentum in n + 1 iteration
                    '''
                    self.prev_output_weight_change = output_weight_change
                    self.prev_output_bias_change =  output_bias_change
                    self.prev_hidden_weight_change = hidden_weight_change
                    self.prev_hidden_bias_change = hidden_bias_change
                
                
                else: # We have reached Error criterion
                    
                    solved = True
                    if iterations > 1:
                        total_solved += 1
                        total_epochs += e + 1
                        print("Solved @ %d" % e)
                        self.total_training_population_error += population_error
                        if self.generalisation:
                            self.test_model(bulk = True)
                    else:
                        print("Error Criterion reached at epoch: %d" % (self.epochs + e + 1)  )
                        print("Final Training Population Error was %f" % population_error)
                        self.epochs = 0 
                    break

            ''' 
            Must reset weights back to random on each iteration.
            If this is a generalisation task we must also create a new random test/training split
            '''
            if iterations > 1:
                self.reset_weights()    
                if self.generalisation:
                    self.reset_sample()

        '''
        If this is was a bulk learning task, and we ran more than one iteration
        We need to output the overall averages of the bulk task.

        If the task is a generalisation task we also output the standard deviation
        across all of the iterations.
        '''
        if iterations > 1:
            print("\nTotal Solved: %d, average epochs to find solution: %d" % (total_solved, total_epochs/total_solved))
            print("Average Training Error")
            print(self.total_training_population_error / total_solved)
            if self.generalisation:
                print("Average Test Error")
                average_population_error = self.total_test_population_error / total_solved
                print(average_population_error)

                # Calculate and output variance
                # Minus mean from every sample
                self.all_solved_test_population_errors[:] = [ (x - average_population_error)**2 for x in self.all_solved_test_population_errors]
                std_dev = np.sqrt(sum(self.all_solved_test_population_errors) / (total_solved - 1))
                print("Test Standard Deviation")
                print(std_dev)

        elif not solved:
            self.epochs += epochs
            print("Not solved over %d epochs" % epochs) 
                        
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

    ''' 
    Called from the bulk training function to resample the input into different traning/test sets for
    the testing of generalisation, only applicable on the iris set
    '''
    def reset_sample(self):
        with open(self.input_filename, "r") as f:
            full_doc = f.readlines()
            indexs = np.arange(len(full_doc))
            np.random.shuffle(indexs)
            number_of_training = int(len(full_doc) * 0.8)
            training_indexs = indexs[:number_of_training]
            test_indexs = indexs[number_of_training:]
            training_input = []
            test_input = []
            for i in training_indexs:
                if full_doc[i].split()[0].find('.') > -1:
                    training_input.append( map(float, full_doc[i].split()) )
                else:
                    training_input.append( map(int, full_doc[i].split()) )
            for i in test_indexs:
                if full_doc[i].split()[0].find('.') > -1:
                    test_input.append( map(float, full_doc[i].split()) )
                else:
                    test_input.append( map(int, full_doc[i].split()) )
            self.input = np.asarray(training_input)
            self.test_input = np.asarray(test_input)
        
        with open(self.output_filename, "r") as f:
                full_doc = f.readlines()
                training_output = []
                test_output = []
                for i in training_indexs:
                    if full_doc[i].split()[0].find('.') > -1:
                        training_output.append( map(float, full_doc[i].split()) )
                    else:
                        training_output.append( map(int, full_doc[i].split()) )

                for i in test_indexs:
                    if full_doc[i].split()[0].find('.') > -1:
                        test_output.append( map(float, full_doc[i].split()) )
                    else:
                        test_output.append( map(int, full_doc[i].split()) )
                self.output = np.asarray(training_output)
                self.test_output = np.asarray(test_output)

    def test_model(self, bulk = False):
        hidden_layer_input = self.test_input.dot(self.hidden_layer_weights) + self.hidden_layer_bias   
        '''
        Feed forward - Hidden Layer Activation
        '''
        if self.hidden_action == "r": # Relu
            hidden_layer_activation = relu_function(hidden_layer_input, 0)
        elif self.hidden_action == "l": # Leaky Relu
            hidden_layer_activation = relu_function(hidden_layer_input, 0.01)
        elif self.hidden_action == "v": # Very Leaky Relu
            hidden_layer_activation = relu_function(hidden_layer_input, 0.15)
        elif self.hidden_action == "n": # Cosine
            hidden_layer_activation = sine_function(hidden_layer_input)    
        else: # Sigmoid
            hidden_layer_activation = sigmoid_function(hidden_layer_input)
        '''
        Feed forward - Output Layer Activation
        '''        
        output_layer_input = hidden_layer_activation.dot(self.output_layer_weights) + self.output_layer_bias
        if self.output_action == "n": # Sine
            test_model_output =  sine_function(output_layer_input)   
        else: # Sigmoid
            test_model_output =  sigmoid_function(output_layer_input)

        #Determine Error
        error = self.test_output - test_model_output
        population_error = 0.5 * np.sum( error**2 ) / (self.output_layer_n * self.test_io_pairs)
        self.total_test_population_error += population_error
        if not bulk:
            print("Model Output")
            print(test_model_output)

        print("Test Population Error")
        print(population_error)
        self.all_solved_test_population_errors.append(population_error)
    
'''
The Actual running of the application. 
Where we provide the UI from, and where we initalize the One_Hidden object
based on user input
'''
if __name__== "__main__":
    folder = raw_input("Please input the model/folder you want to run from. Leave blank for current folder\n")
    hidden_activation = raw_input("\nPlease choose the hidden layer activation function\n"
    "s: Sigmoid\n"
    "r: Relu\n"
    "l: Leaky Relu 0.01\n"
    "v: Leaky Relu 0.15a (This is our optimal)\n"
    "n: Sine\n\n")
    output_activation = raw_input("\nPlease choose the output layer activation function\n"
    "s: Sigmoid (This)\n"
    "n: Sine\n\n")
    generalisation = raw_input("\nDoes this model need to generalise (80:20 Training:Test)\n"
    "y\nn\n\n")
    if generalisation == "y":
        generalisation = True
    else:
        generalisation = False
    one_h = One_Hidden(folder, hidden_activation, output_activation, generalisation)

    print("\nModel Initalised")
    option = 10
    while option != "0":
        try:
            option = raw_input(
                "\nPlease choose option\n" 
                "1: Teach (100 epochs)\n"
                "2: Teach (to criteria)\n" 
                "3: Test (Iris will test generalisation on the 20% of unseen test data)\n" 
                "4: Show weights\n"
                "5: Show Output\n"
                "6: Reset weights\n"
                "7: bulk iterations\n"
                "0: Quit\n\n"
            )
            if option == "0":
                print ("Thanks for learning!")
                break
            
            if option == "1":
                one_h.learn(iterations = 1, epochs = 100)
            
            if option == "2":
                one_h.learn(iterations = 1, epochs = 5000)
            
            if option == "3":
                if folder == "6_iris":
                    one_h.test_model()
                else:
                    print("Can only test generalisation on Iris dataset")
            
            if option == "4":
                one_h.print_weights()
            
            if option == "5":
                one_h.print_model_output()
            
            if option == "6":
                one_h.reset_weights()

            if option == "7":
                try:
                    in_iterations = raw_input("\nhow Many iterations? Leave blank for default of 1000\n");
                except ValueError:
                    in_iterations = 1000
                else:
                    if in_iterations == '':
                        in_iterations = 1000
                    else:
                        in_iterations = int(in_iterations)
                try:
                    in_epochs = raw_input("\nhow Many Epochs? Leave blank for default of default of 5000\n");
                except ValueError:
                    in_epochs = 5000
                else:
                    if in_epochs == '':
                        in_epochs = 5000
                    else:
                        in_epochs = int(in_epochs)    
                one_h.learn(iterations =  in_iterations, epochs = in_epochs)
                
        except EOFError:
            print ("Thanks for learning!")
            break
