    '''
    def learn(self, epochs = 5000):
        solved = False
        epochs = int(epochs)
        for e in range(epochs):
            #Feed Forward
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

            output_layer_input = hidden_layer_activation.dot(self.output_layer_weights) + self.output_layer_bias

            if self.output_action == "n": # Sine
                self.model_output =  sine_function(output_layer_input)          
            else: # Sigmoid
                self.model_output =  sigmoid_function(output_layer_input)

            #Determine Error
            error = self.output - self.model_output
            population_error = 0.5 * np.sum( error**2 ) / (self.output_layer_n * self.io_pairs)
                
            # Backpropogate
            if(population_error > self.error_criterion):
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

                # Remember n-1 for momentum
                self.prev_output_weight_change = output_weight_change
                self.prev_output_bias_change =  output_bias_change
                self.prev_hidden_weight_change = hidden_weight_change
                self.prev_hidden_bias_change = hidden_bias_change
            else:
                # Solution found
                solved = True
                print("Error Criterion reached at epoch: %d" % (self.epochs + e + 1)  )
                print("Final Training Population Error was %f" % population_error)
                self.epochs = 0 
                break

        if not solved:
            self.epochs += epochs
            print("Not solved over %d epochs" % epochs)
    '''