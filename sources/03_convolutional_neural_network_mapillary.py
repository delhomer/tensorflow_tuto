# Raphael Delhome - september 2017

# Convolutional Neural Network with Tensorflow

# The goal of this script is to train a neural network model in order to read
# street scene images produced by Mapillary
# (https://www.mapillary.com/dataset/vistas)

# Four main task will be of interest: data recovering, network conception,
# optimization design and model training.

# Step 0: module imports

import logging
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import sys
import time
import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
f = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(f)
logger.addHandler(ch)

# Step 1: parameter definition

# Define paramaters for the model:
# - relative paths to data
# - image dimensions (width, height, number of channels)
# - hidden layer depth (number of channel per convolutional and fully connected layer)
# - number of output classes
# - number of images per batch
# - number of epochs (one epoch = all images have been used for training)
# - decaying learning rate: fit the learning rate during training according to the convergence step (larger at the beginning, smaller at the end), the used formula is the following: min_lr + (max_lr-min_lr)*math.exp(-i/decay_speed), with i being the training iteration
# - dropout, i.e. percentage of nodes that are briefly removed during training process
# - printing frequency during training

TRAINING_INPUT_PATH = os.path.join("data", "training", "input")
TRAINING_OUTPUT_PATH = os.path.join("data", "training", "output")
VALIDATION_INPUT_PATH = os.path.join("data", "validation", "input")
VALIDATION_OUTPUT_PATH = os.path.join("data", "validation", "output")
IMG_SIZE = (816, 612)
IMAGE_HEIGHT  = IMG_SIZE[1]
IMAGE_WIDTH   = IMG_SIZE[0]
NUM_CHANNELS  = 3 # Colored images (RGB)
L_C1 = 8
K_C1 = 5
STR_C1 = [1, 1, 1, 1]
KS_P1 = [1, 4, 4, 1]
STR_P1 = [1, 4, 4, 1]
L_C2 = 12
K_C2 = 5
STR_C2 = [1, 1, 1, 1]
KS_P2 = [1, 3, 3, 1]
STR_P2 = [1, 3, 3, 1]
L_FC = 1024
N_CLASSES = 66
BATCH_SIZE = 10
N_BATCHES = int(18000 / BATCH_SIZE) # TODO
N_EPOCHS = 1
START_LR = 0.01
MIN_LR = 0.0001
DECAY_SPEED = 1000.0
DROPOUT = 0.75
SKIP_STEP = 10

# Step 2: data recovering

with tf.variable_scope("training_data_pipe") as scope:
    # Reading image file paths
    train_filepaths = os.listdir(TRAINING_INPUT_PATH)
    train_filepaths.sort()
    train_filepaths = [os.path.join(TRAINING_INPUT_PATH, fp) for fp in train_filepaths]
    train_images = ops.convert_to_tensor(train_filepaths, dtype=tf.string,
                                         name="train_images")
    # Reading labels
    train_labels = pd.read_csv(os.path.join(TRAINING_OUTPUT_PATH,
                                            "labels.csv")).iloc[:,6:].values
    train_labels = ops.convert_to_tensor(train_labels, dtype=tf.int16,
                                         name="train_labels")
    # Create input queues
    train_input_queue = tf.train.slice_input_producer([train_images,
                                                       train_labels],
                                                      shuffle=False)
    # Process path and string tensor into an image and a label
    train_file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(train_file_content, channels=NUM_CHANNELS)
    train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    train_label = train_input_queue[1]
    # Collect batches of images before processing
    train_image_batch, train_label_batch, train_filename_batch = tf.train.batch(
        [train_image,
         train_label,
         train_input_queue[0]],
        batch_size=BATCH_SIZE,
        num_threads=4
    )

with tf.variable_scope("training_data_pipe") as scope:
    # Reading image file paths
    validation_filepaths = os.listdir(VALIDATION_INPUT_PATH)
    validation_filepaths.sort()
    validation_filepaths = [os.path.join(VALIDATION_INPUT_PATH, fp)
                            for fp in validation_filepaths]
    validation_images = ops.convert_to_tensor(validation_filepaths,
                                              dtype=tf.string,
                                              name="validation_images")
    # Reading labels
    validation_labels = pd.read_csv(os.path.join(VALIDATION_OUTPUT_PATH,
                                                 "labels.csv")).iloc[:,6:].values
    validation_labels = ops.convert_to_tensor(validation_labels,
                                              dtype=tf.int16,
                                              name="validation_images")
    # Create input queues
    validation_input_queue = tf.train.slice_input_producer([validation_images,
                                                            validation_labels],
                                                           shuffle=False)
    # Process path and string tensor into an image and a label
    validation_file_content = tf.read_file(validation_input_queue[0])
    validation_image = tf.image.decode_jpeg(validation_file_content,
                                            channels=NUM_CHANNELS)
    validation_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    validation_label = validation_input_queue[1]
    # Collect batches of images before processing
    validation_image_batch, validation_label_batch, validation_filename_batch =\
    tf.train.batch([validation_image,
                    validation_label,
                    validation_input_queue[0]],
                   batch_size=BATCH_SIZE,
                   num_threads=4)

# Step 3: Prepare the checkpoint creation

utils.make_dir('../checkpoints')
utils.make_dir('../checkpoints/convnet_mapillary')

# Step 4: create placeholders

with tf.name_scope("data"):
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH,
                                    NUM_CHANNELS], name='X')
    Y = tf.placeholder(tf.float32, [None, N_CLASSES], name='Y')

# Step 5: model building

# The model is composed of the following steps:
# conv -> relu -> pool -> conv -> relu -> pool -> fully connected -> sigmoid
# - conv: convolution between an input neuron and an image filter
# - relu (REctified Linear Unit): neuron activation function
# - pool: max pooling layer, that considers the maximal value in a n*n patch
# - fully connected: full connection between two consecutive neuron layer, concretized by a matrix multiplication
# - sigmoid: neuron activation function, associated with output (multilabel
# classification problem)

# They represent its structure, and may be showed within graph with tensorboard
# command.

# First convolutional layer

with tf.variable_scope('conv1') as scope:
    # Create kernel variable of dimension [K_C1, K_C1, NUM_CHANNELS, L_C1]
    kernel = tf.get_variable('kernel',
                             [K_C1, K_C1, NUM_CHANNELS, L_C1],
                             initializer=tf.truncated_normal_initializer())
    # Create biases variable of dimension [L_C1]
    biases = tf.get_variable('biases',
                             [L_C1],
                             initializer=tf.constant_initializer(0.0))
    # Apply the image convolution
    conv = tf.nn.conv2d(X, kernel, strides=STR_C1, padding='SAME')
    # Apply relu on the sum of convolution output and biases
    conv1 = tf.nn.relu(tf.add(conv, biases), name=scope.name)
    # Output is of dimension BATCH_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH * L_C1.

# First pooling layer

with tf.variable_scope('pool1') as scope:
    # Apply max pooling
    pool1 = tf.nn.max_pool(conv1, ksize=KS_P1, strides=STR_P1, padding='SAME')
    # Output is of dimension BATCH_SIZE x 612 x 816 x L_C1

# Second convolutional layer

with tf.variable_scope('conv2') as scope:
    # Similar to conv1, except kernel now is of the size 5 x 5 x L_C1 x L_C2
    kernel = tf.get_variable('kernels', [K_C2, K_C2, L_C1, L_C2], 
                        initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable('biases', [L_C2],
                        initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(pool1, kernel, strides=STR_C2, padding='SAME')
    conv2 = tf.nn.relu(tf.add(conv, biases), name=scope.name)
    # Output is of dimension BATCH_SIZE x 612 x 816 x L_C2

# Second pooling layer

with tf.variable_scope('pool2') as scope:
    # Similar to pool1
    pool2 = tf.nn.max_pool(conv2, ksize=KS_P2, strides=STR_P2, padding='SAME')
    # Output is of dimension BATCH_SIZE x 153 x 204 x L_C2

# Fully-connected layer

with tf.variable_scope('reshaping') as scope:
    # Reshape pool2 to 2 dimensional
    new_height = IMAGE_HEIGHT / (STR_C1[2]*STR_P1[2]*STR_C2[2]*STR_P2[2])
    new_width = IMAGE_WIDTH / (STR_C1[1]*STR_P1[1]*STR_C2[1]*STR_P2[1])
    input_features = new_height * new_width * L_C2
    reshaped = tf.reshape(pool2, [-1, input_features])

with tf.variable_scope('fc') as scope:
    # Create weights and biases
    w = tf.get_variable('weights', [input_features, L_FC],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [L_FC],
                        initializer=tf.constant_initializer(0.0))
    # Apply relu on matmul of reshaped and w + b
    fc = tf.nn.relu(tf.add(tf.matmul(reshaped, w), b), name='relu')
    # Apply dropout
    dropout = tf.placeholder(tf.float32, name='dropout')
    fc = tf.nn.dropout(fc, dropout, name='relu_dropout')

# Output building

with tf.variable_scope('sigmoid_linear') as scope:
    # Create weights and biases for the final fully-connected layer
    w = tf.get_variable('weights', [L_FC, N_CLASSES],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [N_CLASSES],
                        initializer=tf.random_normal_initializer())
    # Compute logits through a simple linear combination
    logits = tf.matmul(fc, w) + b
    # Compute predicted outputs with sigmoid function
    Ypredict = tf.nn.sigmoid(logits)

# Step 6: loss function design

# Use cross-entropy loss function (-sum(Y_i * log(Yi)) ), normalised for
# batches of BATCH_SIZE images. TensorFlow provides the
# sigmoid_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) (which is NaN).
# We use sigmoid instead of softmax as we are in a multilabel classification
# problem

with tf.name_scope('loss'):
    # Cross-entropy between predicted and real values    
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(entropy, name="loss")

with tf.name_scope('accuracy'):
    # A prediction is correct when the rounded predicted output is equal to Y
    correct_prediction = tf.equal(tf.round(Y), tf.round(Ypredict))
    # Accuracy of the trained model, between 0 (worst) and 1 (best)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Step 7: Define training optimizer

with tf.name_scope("train"):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    # Variable learning rate
    lrate = tf.train.exponential_decay(START_LR, global_step,
                                       decay_steps=1000, decay_rate=0.95,
                                       name='learning_rate')
    # Use Adam optimizer with decaying learning rate to minimize cost.
    optimizer = tf.train.AdamOptimizer(lrate).minimize(loss, global_step=global_step)

# Final step: running the neural network

with tf.Session() as sess:
    # Initialize the tensorflow variables
    # To visualize using TensorBoard
    # tensorboard --logdir="../graphs/convnet_mapillary" --port 6006)
    sess.run(tf.global_variables_initializer())
    # Declare a saver instance and a summary writer to store the trained network
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('../graphs/convnet_mapillary', sess.graph)
    initial_step = global_step.eval(session=sess)
    # Create folders to store checkpoints
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('../checkpoints/convnet_mapillary/checkpoint'))
    # If that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    # Initialize threads to begin batching operations
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Train the model
    start_time = time.time()
    epoches = []
    losses = []
    accuracies = []
    for index in range(initial_step, N_BATCHES * N_EPOCHS):
        X_batch, Y_batch, filename_batch = sess.run([train_image_batch,
                                                     train_label_batch,
                                                     train_filename_batch])
        if index % SKIP_STEP == 0:
            loss_batch, accuracy_batch = sess.run([loss, accuracy], 
                                feed_dict={X: X_batch, Y:Y_batch, dropout: 1.0})
            logger.info("""Step {}: loss = {:5.1f}, accuracy = {:1.3f}""".format(index, loss_batch, accuracy_batch))
            epoches.append(index)
            losses.append(loss_batch)
            accuracies.append(accuracy_batch)
        if (index+1) % N_BATCHES == 0:
            saver.save(sess, '../checkpoints/convnet_mapillary/epoch', index)
        sess.run(optimizer, feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT})
    logger.info("Optimization Finished!")
    logger.info("Total time: {:.2f} seconds".format(time.time() - start_time))
    
    # The results are stored as a pandas dataframe and saved on the file system.
    param_history = pd.DataFrame({"epoch":epoches, "loss":losses,
                                  "accuracy":accuracies})
    param_history = param_history.set_index("epoch")
    if initial_step == 0:
        param_history.to_csv("cnn_mapillary.csv", index=True)
    else:
        param_history.to_csv("cnn_mapillary.csv", index=True, mode='a', header=False)
    # Stop the threads used during the process
    coord.request_stop()
    coord.join(threads)
