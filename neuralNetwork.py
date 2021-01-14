import numpy as np
from activations_fun import softmax


class neuralNetwork:
    def __init__(
        self,
        n_input,
        n_output,
        hidden_layers,
        learningrate,
        activation_function,
        bias,
    ):
        self.input = n_input
        self.output = n_output
        self.hidden = hidden_layers
        if type(self.hidden) is tuple:
            self.w = [
                np.random.normal(
                    0.0, pow(self.input, -0.5), (self.hidden[0], self.input)
                )
            ]
            for i in range(len(self.hidden) - 1):
                self.w.append(
                    np.random.normal(
                        0.0,
                        pow(self.hidden[i], -0.5),
                        (self.hidden[i + 1], self.hidden[i]),
                    )
                )
            self.w.append(
                np.random.normal(
                    0.0,
                    pow(self.hidden[-1], -0.5),
                    (self.output, self.hidden[-1]),
                )
            )
        else:
            self.w = [
                np.random.normal(
                    0.0, pow(self.input, -0.5), (self.hidden, self.input)
                )
            ]
            self.w.append(
                np.random.normal(
                    0.0, pow(self.hidden, -0.5), (self.output, self.hidden)
                )
            )
        self.add_bias = bias
        self.lr = learningrate
        self.activation_function = activation_function
        self.loss, self.val_loss = [], []
        self.acc, self.val_acc = [], []
        self.bias, self.bias_lr = [], 0.01
        self.sdw, self.sdb = [], []
        self.beta, self.epsilon = 0.9, 10 ** -8
        for i in range(len(self.w)):
            self.sdw.append(np.zeros(self.w[i].shape))
            self.bias.append(np.zeros((self.w[i].shape[0], 1)))
            self.sdb.append(np.zeros((self.w[i].shape[0], 1)))

    def feedforward(self, inputs):
        hidden_inputs = []
        hidden_outputs = []
        for i in range(len(self.w)):
            if i == 0:
                hidden_inputs.append((np.dot(self.w[i], inputs) + self.bias[i]))
            else:
                hidden_inputs.append(
                    (np.dot(self.w[i], hidden_outputs[i - 1]) + self.bias[i])
                )
            hidden_outputs.append(self.activation_function(hidden_inputs[i]))
        hidden_outputs.insert(0, inputs)
        return hidden_outputs

    def backward_propagation(self, output_errors, hidden_outputs, inputs):

        for i in range(1, len(self.w) + 1):
            if i == 1:
                error = output_errors
            else:
                error = np.dot(self.w[-(i - 1)].T, error)
            self.w[-i] += self.lr * np.dot(
                (error * hidden_outputs[-i] * (1.0 - hidden_outputs[-i])),
                hidden_outputs[-(i + 1)].T,
            )
            # self.sdw[-i] = (self.beta * self.sdw[-i]) + ((1 - self.beta) * np.dot((error * hidden_outputs[-i] * (1.0 - hidden_outputs[-i])), hidden_outputs[-(i + 1)].T)**2)
            # self.w[-i] += self.lr * np.dot((error * hidden_outputs[-i] * (1.0 - hidden_outputs[-i])), hidden_outputs[-(i + 1)].T) / ((self.sdw[-i])**0.5 + self.epsilon)
            if self.add_bias is True:
                self.bias[-i] += self.bias_lr * (
                    error * hidden_outputs[-i] * (1.0 - hidden_outputs[-i])
                )
                # self.sdb[-i] = (self.beta * self.sdb[-i]) + ((1 - self.beta) * (error * hidden_outputs[-i] * (1.0 - hidden_outputs[-i]))**2)
                # self.bias[-i] += self.bias_lr * (error * hidden_outputs[-i] * (1.0 - hidden_outputs[-i])) / ((self.sdb[-i])**0.5 + self.epsilon)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_outputs = self.feedforward(inputs)
        output_errors = targets - hidden_outputs[-1]
        self.backward_propagation(output_errors, hidden_outputs, inputs)

    def query(self, inputs_lists):
        value = np.array(inputs_lists, ndmin=2).T
        for layer in range(len(self.w)):
            if layer == len(self.w) - 1:
                value = softmax(np.dot(self.w[layer], value) + self.bias[layer])
            else:
                value = self.activation_function(
                    np.dot(self.w[layer], value) + self.bias[layer]
                )
        return value
