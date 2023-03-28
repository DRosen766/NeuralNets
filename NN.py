import random
from math import exp

def initialize_model(num_inputs, num_hidden, num_outputs):
    model = list()
    hidden_layer = [{"weights":[random() for i in num_inputs + 1]} for j in num_hidden]
    model.append(hidden_layer)
    output_layer = [{"weights":[random() for i in num_hidden + 1]} for j in num_outputs]
    model.append(output_layer)
    return model

def activate(weights, inputs):
    bias = weights[-1]
    activation = bias
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

def transfer(activation):
    return (1.0 / (1.0 + exp(-activation)))

def forward_propogate(model, row):
    current_inputs = row
    for layer in model:
        next_inputs = []
        for neuron in layer:
            neuron["output"] = transfer(activate(neuron["weights"], current_inputs))
            next_inputs.append(neuron["output"])
        current_inputs = next_inputs
    return current_inputs

def transfer_derivative(output):
    return 1.0 / (1.0 - output)

def backpropogate_error(model, expected_outputs):
    # iterate backwards through network layers
    for i in reversed(range(len(model))):
        errors = list()
        # if not in the output layer
        if(i < len(model) - 1):
            # iterate through neurons
            for j in model[i]:
            # iterate through layer after
                error = 0.0
                # calculate error and add to errors list
                for neuron in model[i + 1]:
                    error += (neuron["weights"][j] * neuron["delta"])
                errors.append(error)
            
        # if in the output layer
        else:
            # iterate through neurons in output layer
            for j in range(len(model[i])):
                errors.append(model[i][j] - expected_outputs[j])
        # iterate through neurons in layer, assign deltas
        for j in range(len(model[i])):
            model[i][j]["delta"] = errors[j] * transfer_derivative(model[i][j]["output"])
    
def update_weights(model, learning_rate, row):
    inputs = row[:-1]
    # iterate through layers
    for layer_index in range(1, len(model)):
        # create list of neuron outputs in previous layer
        inputs = [neuron["output"] for neuron in model[layer_index-1]]
        # iterate through neurons in layer
        for neuron in model[layer_index]:
            # iterate through inputs
            for j in range(len(inputs)):
                # adjust weights
                neuron["weights"][j] -= learning_rate * \
                    neuron['delta'] * inputs[j]
            # adjust bias
            neuron["weights"][-1] -= learning_rate * neuron['delta']
            