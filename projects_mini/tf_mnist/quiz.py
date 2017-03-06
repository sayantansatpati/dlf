# Solution is available in the other "quiz_solution.py" tab
import tensorflow as tf

def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    # TODO: Return weights
    '''
    Initializing the weights with random numbers from a normal distribution is good practice.
    Randomizing the weights helps the model from becoming stuck in the same place every time you train it.
    You'll learn more about this in the next lesson, when you study gradient descent.

    Similarly, choosing weights from a normal distribution prevents any one weight from overwhelming other weights.
    You'll use the tf.truncated_normal() function to generate random numbers from a normal distribution.

    The tf.truncated_normal() function returns a tensor with random values from a normal distribution whose magnitude
    is no more than 2 standard deviations from the mean
    '''
    return tf.Variable(tf.truncated_normal((n_features, n_labels)))


def get_biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    '''
    Since the weights are already helping prevent the model from getting stuck, you don't need to randomize the bias.
    Let's use the simplest solution, setting the bias to 0.
    '''
    # TODO: Return biases
    return tf.Variable(tf.zeros(n_labels))


def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    # TODO: Linear Function (xW + b)
    return tf.matmul(input,w) + b