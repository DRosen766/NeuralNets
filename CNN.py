import numpy as np
from numpy import random
from math import exp


# number of convolutional nodes
num_conv = 3

"""
:param num_inputs: shape of input layer as tuple (ros, columns)
:param num_outputs: shape of output layer as tuple (ros, columns)
"""


def initialize_cnn_model(input_shape, num_outputs):
    model = []
    conv_layer = []
    for _ in range(num_conv):
        conv_node = {
            "filter": np.matrix(
                [
                    [random.rand() for _ in range(input_shape[1])]
                    for _ in range(input_shape[0])
                ]
            ),
            "stride": 1,
        }
        conv_layer.append(conv_node)
    model.append(conv_layer)

    max_pool_layer = []
    for _ in range(num_conv):
        max_pool_node = {}
        max_pool_layer.append(max_pool_node)
    model.append(max_pool_layer)

    output_layer = []
    for _ in range(num_outputs):
        output_node = {"weights": random.rand(num_conv)}
        output_layer.append(output_node)
    model.append(output_layer)
    return model


def activate(weights, inputs):
    bias = weights[-1]
    activation = bias
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def transfer_derivative(output):
    return 1.0 / (1.0 - output)


"""
:param model: the cnn
:param inputs: input values to model
"""


def forward_propagate(model, inputs: np.matrix):
    # *execute convolutional layer on inputs
    # iterate through convolutional nodes
    for conv_node in model[0]:
        # store inputs for error backpropogation
        conv_node["inputs"] = inputs
        conv_node["output"] = np.zeros(inputs.shape[0] // len(conv_node["filter"]))
        # iterate through rows in the input matrix
        for r in range(
            0, inputs.shape[0] - len(conv_node["filter"]), conv_node["stride"]
        ):
            # iterate through columns in the input matrix
            for c in range(
                0, inputs.shape[1] - len(conv_node["filter"][0]), conv_node["stride"]
            ):
                filter_sum = 0.0
                # iterate through rows and columns in filter
                for filter_r in range(len(conv_node["filter"])):
                    for filter_c in range(len(conv_node["filter"][filter_r])):
                        filter_sum += (
                            conv_node["filter"][filter_r, filter_c]
                            * inputs[r + filter_r, c + filter_c]
                        )
                if len(conv_node["output"]) == 1:
                    conv_node["output"][0] = filter_sum
                else:
                    conv_node["output"][r, c] = filter_sum

    # *calculate maxpool results of each convolutional node
    # iterate through max pool nodes
    max_pool_outputs = []
    for max_pool_node_index in range(len(model[1])):
        # calculate maxpool of each conv_node output and store its location
        model[1][max_pool_node_index]["max_location"] = np.unravel_index(
            model[0][max_pool_node_index]["output"].argmax(),
            model[0][max_pool_node_index]["filter"].shape,
        )
        # the output of the max pool is the location of the conv_layers output with the largest value
        model[1][max_pool_node_index]["output"] = (
            model[0][max_pool_node_index]["output"][
                model[1][max_pool_node_index]["max_location"]
            ]
            if len(model[0][max_pool_node_index]["output"]) > 1
            else model[0][max_pool_node_index]["output"]
        )

        max_pool_outputs.append(model[1][max_pool_node_index]["output"])

    # *feed maxpool results into final layer
    # iterate through output nodes
    for output_node in model[2]:
        output_node["output"] = transfer(
            activate(output_node["weights"], max_pool_outputs)
        )
    return [output_node["output"] for output_node in model[-1]]

# NOTE: this code does not currently calculate dL_dX which is equivalent to dL_dO for the preceeding layer 
def backpropogate_error(model, target_outputs):
    # compare output to target_output
    for i in range(len(model[-1])):
        model[-1][i]["delta"] = (model[-1][i]["output"] - target_outputs[i]) * transfer_derivative(model[-1][i]["output"])

    # *calculate deltas for max_pool nodes
    # iterate through max_pool nodes
    for max_pool_node_index in range(len(model[1])):
        error_sum = 0
        # iterate through output nodes
        # sum all errors
        for output_node in model[-1]:
            error_sum += (
                output_node["delta"] * output_node["weights"][max_pool_node_index]
            )
        # assign delta for current max_pool node and corresponding convolutional node
        model[1][max_pool_node_index]["delta"] = error_sum

    # iterate through convolutional nodes
    for conv_node_index in range(len(model[0])):
        # calculate loss from delta of max pool node
        # one because the previous layer is a maxpool layer
        model[0][conv_node_index]["dL_dO"] = 1
        # *calculate local filter gradient
        # determine which output element needs to be "lossed"
        lossed_element = model[1][conv_node_index]["max_location"]
        # find the subsection of the input matrix corresponding to that output element
        lossed_subsection = model[0][conv_node_index]["inputs"][
            lossed_element[0] : lossed_element[0]
            + len(model[0][conv_node_index]["filter"]),
            lossed_element[1] : lossed_element[1]
            + len(model[0][conv_node_index]["filter"][0])
        ]

        # create matrix of deltas for filter
        model[0][conv_node_index]["dO_dF"] = np.zeros(model[0][conv_node_index]["filter"].shape)
        model[0][conv_node_index]["dO_dF"][
            lossed_element[0] : lossed_element[0] + lossed_subsection.shape[0],
            lossed_element[1] : lossed_element[1] + lossed_subsection.shape[1],
        ] += lossed_subsection
        # multiply by dL_dO
        # assign as dL_dF of node
        model[0][conv_node_index]["dL_dF"] = (
            model[0][conv_node_index]["dO_dF"] * model[0][conv_node_index]["dL_dO"]
        )


def update_weights(model, learning_rate):
    for conv_node in model[0]:
        # update filter elements based on dL_dF
        conv_node["filter"] -= learning_rate * conv_node["inputs"] * conv_node["dL_dF"]
    for output_node in model[-1]:
          # iterate through inputs
            for j in range(len(model[-2]["outputs"])):
                # adjust weights
                output_node["weights"][j] -= learning_rate * \
                    output_node['delta'] * model[-2]["outputs"][j]
model = initialize_cnn_model((2, 2), 1)

output = forward_propagate(model, np.eye(3))
backpropogate_error(model, [0])
print(output)