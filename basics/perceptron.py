import numpy as np

def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1/(1+np.exp(-x))

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

# TODO: Calculate the output
output = sigmoid(bias + np.dot(weights,inputs))

print('Output: {0}'.format(output))
