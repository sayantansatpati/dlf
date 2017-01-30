import numpy as np

def disp(a,msg):
    print '\n### {0}'.format(msg)
    print 'nDim: {0}, shape: {1}'.format(np.ndim(a), a.shape)
    print a[:5]


def sig(z):
    return 1 / (1 + np.exp(-z))


def deriv_sig_1(z):
    return sig(z)/(1 - sig())


def deriv_sig_2(sigmoid):
    return sigmoid/(1 - sigmoid)

# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
disp(X,"X")

# output dataset
y = np.array([[0, 0, 1, 1]]).T
disp(y,"y")

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1
disp(syn0,"syn0")

for iter in xrange(1):
    # forward propagation
    l0 = X
    l1 = sig(np.dot(l0, syn0))
    disp(l1, "l1")

    # how much did we miss?
    l1_error = y - l1
    disp(l1_error, "l1_error")

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)
    disp(l1_delta, "l1_delta")

    # update weights
    syn0 += np.dot(l0.T, l1_delta)
    disp(syn0, "syn0")

print "Output After Training:"
print l1
