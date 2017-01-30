import numpy as np
import matplotlib.pyplot as plt


def generate_data(n=100,m=1):
    """Generate OLR Dataset"""
    # One feature
    x = np.random.uniform(0, 10, n)
    # Gaussian noise
    b = np.random.normal(0, 0.1, n)

    # b == theta0, m == theta1
    y = m * x + b

    '''
    # Scatter
    plt.scatter(x,y)
    # Show Regression Line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.show()
    '''

    return x,y


def calculate_cost(X,y,w):
    return np.mean((y - np.dot(X,w)) ** 2)


def calc_gradient_descent(X,y,w,alpha=0.0005):
    hypothesis = np.dot(X,w.T)
    loss = y - np.dot(X,w.T)
    # Calc Gradients
    for i in range(0,X.shape[1]):
        w[i] = w[i] + alpha * np.dot(loss, X[:,i])

    return w


def disp(a,msg):
    print('\n### {0}'.format(msg))
    print('nDim: {0}, shape: {1}'.format(np.ndim(a), a.shape))
    print(a[:5])


if __name__ == '__main__':
    print('####################################')
    print('### Gradient Descent Using NumPy ###')
    print('####################################\n')

    np.random.seed(101)

    x,y = generate_data()
    print('### Generated Data\n')
    disp(x,"x")
    disp(y,"y")

    x0 = np.ones(x.shape[0])
    disp(x0, "x0")

    X = np.column_stack((x0,x))
    disp(X, "X")

    '''
    y = y[:,np.newaxis]
    print "y: ", y.shape, y.ndim, y[:5]
    '''

    w = np.zeros(X.shape[1])
    disp(w, "w")

    print('\n### Epochs: Train GD')

    epochs = 1000
    costs = []
    for i in range(0, epochs):
        w = calc_gradient_descent(X,y,w)

        c = calculate_cost(X,y,w)
        costs.append(c)

        print('Epoch: {0}, Cost: {1}, Weights: {2}'.format(i, c, w))

    #plt.plot(costs)
    #plt.show()





