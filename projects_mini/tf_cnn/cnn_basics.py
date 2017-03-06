import tensorflow as tf


'''

### Dimensionality:

From what we've learned so far, how can we calculate the number of neurons of each layer in our CNN?

Given our input layer has a volume of W, our filter has a volume (height * width * depth) of F, we have a stride of S,
and a padding of P, the following formula gives us the volume of the next layer: (W−F+2P)/S+1.

### Example:

H = height, W = width, D = depth

We have an input of shape 32x32x3 (HxWxD)
20 filters of shape 8x8x3 (HxWxD)
A stride of 2 for both the height and width (S)
Valid padding of size 1 (P)

new_height = (input_height - filter_height + 2 * P)/S + 1
new_width = (input_width - filter_width + 2 * P)/S + 1

What's the shape of the output?

Answer: 14x14x20


~~ Parameters:

~ Without Parameter Sharing:

We're now going to calculate the number of parameters of the convolutional layer.
The answer from the last quiz will come into play here!

Without parameter sharing, each neuron in the output layer must connect to each neuron in the filter.
In addition, each neuron in the output layer must also connect to a single bias neuron.


(8 * 8 * 3 + 1) * (14 * 14 * 20) = 756560

~ With Parameter Sharing:

With parameter sharing, each neuron in an output channel shares its weights with every other neuron in that channel.
So the number of parameters is equal to the number of neurons in the filter, plus a bias neuron,
all multiplied by the number of channels in the output layer.


(8 * 8 * 3 + 1) * 20 = 3840 + 20 = 3860

That's 3840 weights and 20 biases. This should look similar to the answer from the previous quiz.
The difference being it's just 20 instead of (14 * 14 * 20).
Remember, with weight sharing we use the same filter for an entire depth slice.
Because of this we can get rid of 14 * 14 and be left with only 20.

'''

input = tf.placeholder(tf.float32, (None, 32, 32, 3))
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) # (height, width, input_depth, output_depth)
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1] # (batch, height, width, depth)
padding = 'VALID'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias