# Convolutional Neural Network with Tensorflow

# The goal of this notebook is to train a neural network model in order to read
# hand-written digits automatically. It uses the Tensorflow library, developed
# by Google.

# Although the notebook is divided into smaller steps, three main task will be
# of interest: network conception, optimization design and model training.

# Step 0: module imports

# Among necessary modules, there is of course Tensorflow; but also an utilitary
# for reading state-of-the-art data sets, like MNIST.

import math
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
# Alternative choice: from tensorflow.examples.tutorials.mnist import input_data
import time

# Step 1: data recovering

# Read in data using TF Learn's built in function to load MNIST data to the
# folder data/mnist

mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
# If alternative module import: mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

# Step 2: parameter definition

# Define paramaters for the model:
# - hidden layer depth (number of channel per convolutional and fully connected layer)
# - number of output classes
# - number of images per batch
# - number of epochs (one epoch = all images have been used for training)
# - decaying learning rate: fit the learning rate during training according to the convergence step (larger at the beginning, smaller at the end), the used formula is the following: min_lr + (max_lr-min_lr)*math.exp(-i/decay_speed), with i being the training iteration
# - dropout, i.e. percentage of nodes that are briefly removed during training process
# - printing frequency during training

L_C1 = 32
L_C2 = 64
L_FC = 512
N_CLASSES = 10
BATCH_SIZE = 150
N_EPOCHS = 5
MAX_LR = 0.003
MIN_LR = 0.0001
DECAY_SPEED = 1000.0
DROPOUT = 0.75
SKIP_STEP = 10

# Step 3: Prepare the checkpoint creation

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
make_dir('checkpoints')
make_dir('checkpoints/convnet_mnist')

# Step 4: create placeholders

# In Tensorflow, placeholders refer to variables that will be fed each time the
# model is run.

# Each image in the MNIST data is of shape 28*28*1 (greyscale) therefore, each
# image is represented with a 28*28*1 tensor; use None for shape so we can
# change the batch_size once we've built the tensor graph. The resulting output
# is a vector of N_CLASSES 0-1 values, the only '1' being the model prediction.

# As we work with a decaying learning rate, this quantity is managed within a
# placeholder. We'll be doing dropout for hidden layer so we'll need a
# placeholder for the dropout probability too.

with tf.name_scope("data"):
    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')
    # If alternative module import: X = tf.placeholder(tf.float32, [None, 784], name="X")
    Y = tf.placeholder(tf.float32, [None, N_CLASSES], name='Y')
# variable learning rate
lrate = tf.placeholder(tf.float32, name='learning_rate')
# dropout proportion
dropout = tf.placeholder(tf.float32, name='dropout')

# Step 5: model building

# The model is composed of the following steps:

# conv -> relu -> pool -> conv -> relu -> pool -> fully connected -> softmax
# - conv: convolution between an input neuron and an image filter
# - relu (REctified Linear Unit): neuron activation function
# - pool: max pooling layer, that considers the maximal value in a n*n patch
# - fully connected: full connection between two consecutive neuron layer, concretized by a matrix multiplication
# - softmax: neuron activation function, associated with output

# They represent its structure, and may be showed within graph with tensorboard
# command.

# First convolutional layer

with tf.variable_scope('conv1') as scope:
    # if alternative module import, reshape the image to [BATCH_SIZE, 28, 28, 1]
    # X = tf.reshape(X, shape=[-1, 28, 28, 1])
    # create kernel variable of dimension [5, 5, 1, 32]
    kernel = tf.get_variable('kernel',
                             [5, 5, 1, L_C1],
                             initializer=tf.truncated_normal_initializer())
    # create biases variable of dimension [32]
    biases = tf.get_variable('biases',
                             [L_C1],
                             initializer=tf.constant_initializer(0.0))
    
    # apply tf.nn.conv2d. strides [1, 1, 1, 1], padding is 'SAME'
    conv = tf.nn.conv2d(X, kernel, strides=[1, 1, 1, 1], padding='SAME')
    # apply relu on the sum of convolution output and biases
    conv1 = tf.nn.relu(conv+biases, name=scope.name)

# Output is of dimension BATCH_SIZE * 28 * 28 * 32.

# First pooling layer

with tf.variable_scope('pool1') as scope:
    # apply max pool with ksize [1, 2, 2, 1], and strides [1, 2, 2, 1], padding
    # 'SAME'
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

# Output is of dimension BATCH_SIZE x 14 x 14 x 32

# Second convolutional layer

with tf.variable_scope('conv2') as scope:
    # similar to conv1, except kernel now is of the size 5 x 5 x 32 x 64
    kernel = tf.get_variable('kernels', [5, 5, L_C1, L_C2], 
                        initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable('biases', [L_C2],
                        initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv + biases, name=scope.name)

# Output is of dimension BATCH_SIZE x 14 x 14 x 64

# Second pooling layer

with tf.variable_scope('pool2') as scope:
    # similar to pool1
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

# Output is of dimension BATCH_SIZE x 7 x 7 x 64

# Fully-connected layer

with tf.variable_scope('fc') as scope:
    # use weight of dimension 7 * 7 * 64 x 1024
    input_features = 7 * 7 * L_C2
    # create weights and biases
    w = tf.get_variable('weights', [input_features, L_FC],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [L_FC],
                        initializer=tf.constant_initializer(0.0))
    # reshape pool2 to 2 dimensional
    pool2 = tf.reshape(pool2, [-1, input_features])
    # apply relu on matmul of pool2 and w + b
    fc = tf.nn.relu(tf.matmul(pool2, w) + b, name='relu')
    # apply dropout
    fc = tf.nn.dropout(fc, dropout, name='relu_dropout')

# Output building

with tf.variable_scope('softmax_linear') as scope:
    # get logits without softmax you need to create weights and biases
    w = tf.get_variable('weights', [L_FC, N_CLASSES],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [N_CLASSES],
                        initializer=tf.random_normal_initializer())
    logits = tf.matmul(fc, w) + b
    Ypredict = tf.nn.softmax(logits)

# Step 6: loss function design

# Use cross-entropy loss function (-sum(Y_i * log(Yi)) ), normalised for
# batches of 100 images. TensorFlow provides the
# softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) (which is NaN).

with tf.name_scope('loss'):
    # cross-entropy between predicted and real values    
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(entropy, name="loss")

with tf.name_scope('accuracy'):
    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Ypredict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Step 7: Define training optimizer

# Use Adam optimizer with decaying learning rate to minimize cost.
with tf.name_scope("train"):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(lrate).minimize(loss, global_step=global_step)

# Final step: running the neural network

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # to visualize using TensorBoard (tensorboard --logdir="./graphs/convnet"
    # --port 6006)
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./graphs/convnet', sess.graph)
    ##### You have to create folders to store checkpoints
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)

    initial_step = global_step.eval(session=sess)

    # Train the model
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)
    start_time = time.time()
    for index in range(n_batches * N_EPOCHS): # train the model n_epochs times
        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
        learning_rate = MIN_LR + (MAX_LR - MIN_LR) * math.exp(-index/DECAY_SPEED)
        if index % SKIP_STEP == 0:
            loss_batch, accuracy_batch = sess.run([loss, accuracy], 
                                feed_dict={X: X_batch, Y:Y_batch, lrate:
                                           learning_rate, dropout: 1.0})
            print("""Step {}: loss = {:5.1f},\
            accuracy = {:1.3f}""".format(index, loss_batch, accuracy_batch))
            saver.save(sess, 'checkpoints/convnet_mnist/epoch', index)
        sess.run(optimizer, feed_dict={X: X_batch, Y: Y_batch, lrate:
                                       learning_rate, dropout: DROPOUT})
    print("Optimization Finished!")
    print("Total time: {:.2f} seconds".format(time.time() - start_time))

    # Test the model
    X_batch, Y_batch = mnist.test.next_batch(BATCH_SIZE)
    test_dict = {X: mnist.test.images, Y: mnist.test.labels, lrate:
                     learning_rate, dropout: DROPOUT}
    _, loss_batch, accuracy_batch = sess.run([optimizer, loss, accuracy], 
                                             feed_dict=test_dict)
    print("Accuracy={:1.3f}; loss={:1.3f}".format(accuracy_batch, loss_batch))
    
