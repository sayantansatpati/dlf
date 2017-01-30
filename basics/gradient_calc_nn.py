import numpy as np

def sigmoid(x):
    """Signmoid"""
    return 1/(1+np.exp(-x))


def sigmoid_deriv(sig_moid):
    """Derivative of the Sigmoid"""
    return sig_moid * (1 - sig_moid)


learnrate = 0.5
x = np.array([1, 2])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5])

# Calculate one gradient descent step for each weight
# TODO: Calculate output of neural network
nn_output = sigmoid(np.dot(x,w))

# TODO: Calculate error of neural network
error = y - nn_output

# TODO: Calculate change in weights
del_w = learnrate * np.dot(error * sigmoid_deriv(nn_output), x.T)

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)