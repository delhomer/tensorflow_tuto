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
# - hidden layer depth (number of channel per convolutional and fully connected
# layer), kernel dimension, conv layer stride, max pool layer ksize and stride
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
NETWORK_NAME = "cnn_mapillary"

# Step 2: data recovering

def prepare_data(height, width, n_channels, batch_size, dataset_type, scope_name):
    INPUT_PATH = os.path.join("data", dataset_type, "input")
    OUTPUT_PATH = os.path.join("data", dataset_type, "output")
    with tf.variable_scope(scope_name) as scope:
        # Reading image file paths
        filepaths = os.listdir(INPUT_PATH)
        filepaths.sort()
        filepaths = [os.path.join(INPUT_PATH, fp) for fp in filepaths]
        images = ops.convert_to_tensor(filepaths, dtype=tf.string,
                                       name=dataset_type+"_images")
        # Reading labels
        labels = (pd.read_csv(os.path.join(OUTPUT_PATH, "labels.csv"))
                  .iloc[:,6:].values)
        labels = ops.convert_to_tensor(labels, dtype=tf.int16,
                                       name=dataset_type+"_labels")
        # Create input queues
        input_queue = tf.train.slice_input_producer([images, labels],
                                                    shuffle=False)
        # Process path and string tensor into an image and a label
        file_content = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(file_content, channels=n_channels)
        image.set_shape([height, width, n_channels])
        label = input_queue[1]
        # Collect batches of images before processing
        return tf.train.batch([image, label, input_queue[0]],
                              batch_size=batch_size,
                              num_threads=4)

train_image_batch, train_label_batch, train_filename_batch = \
prepare_data(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, BATCH_SIZE, "training", "training_data_pipe")
validation_image_batch, validation_label_batch, validation_filename_batch =\
prepare_data(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, BATCH_SIZE,
                 "validation", "validation_data_pipe")

# Step 3: Prepare the checkpoint creation

utils.make_dir('../checkpoints')
utils.make_dir('../checkpoints/'+NETWORK_NAME)

# Step 4: create placeholders

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH,
                                NUM_CHANNELS], name='X')
Y = tf.placeholder(tf.float32, [None, N_CLASSES], name='Y')
dropout = tf.placeholder(tf.float32, name='dropout')

# Step 5: model building

CONV_LAYER_COUNTER = 0
POOL_LAYER_COUNTER = 0
FULLCON_LAYER_COUNTER = 0

# Convolutional layer
def conv_layer(input_layer, input_layer_depth, kernel_dim, layer_depth, conv_strides, counter):
    counter = counter + 1
    with tf.variable_scope('conv'+str(counter)) as scope:
        # Create kernel variable of dimension [K_C1, K_C1, NUM_CHANNELS, L_C1]
        kernel = tf.get_variable('kernel',
                                 [kernel_dim, kernel_dim,
                                  input_layer_depth, layer_depth],
                                 initializer=tf.truncated_normal_initializer())
        # Create biases variable of dimension [L_C1]
        biases = tf.get_variable('biases',
                                 [layer_depth],
                                 initializer=tf.constant_initializer(0.0))
        # Apply the image convolution
        conv = tf.nn.conv2d(input_layer, kernel, strides=conv_strides,
                            padding='SAME')
        # Apply relu on the sum of convolution output and biases
        # Output is of dimension BATCH_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH * L_C1.
        return tf.nn.relu(tf.add(conv, biases), name=scope.name), counter

# Max-pooling layer
def maxpool_layer(input_layer, pool_ksize, pool_strides, counter):
    counter = counter + 1
    with tf.variable_scope('pool'+str(counter)) as scope:
        return tf.nn.max_pool(input_layer, ksize=pool_ksize,
                               strides=pool_strides, padding='SAME'), counter
        # Output is of dimension BATCH_SIZE x 612 x 816 x L_C1

# Fully-connected layer
def reshape(height, width, str_c1, str_p1, str_c2, str_p2, last_layer_depth):
    new_height = int(height / (str_c1[2]*str_p1[2]*str_c2[2]*str_p2[2]))
    new_width = int(width / (str_c1[1]*str_p1[1]*str_c2[1]*str_p2[1]))
    return new_height * new_width * last_layer_depth

def fullconn_layer(input_layer, height, width, str_c1, str_p1, str_c2, str_p2,
                   last_layer_depth, fc_layer_depth, counter):
    counter = counter + 1
    with tf.variable_scope('fc'+str(counter)) as scope:
        fc_size = reshape(height, width, str_c1, str_p1, str_c2, str_p2,
                          last_layer_depth)
        reshaped = tf.reshape(input_layer, [-1, fc_size])
        # Create weights and biases
        w = tf.get_variable('weights', [fc_size, fc_layer_depth],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [fc_layer_depth],
                            initializer=tf.constant_initializer(0.0))
        # Apply relu on matmul of reshaped and w + b
        fc = tf.nn.relu(tf.add(tf.matmul(reshaped, w), b), name='relu')
        # Apply dropout
        return tf.nn.dropout(fc, dropout, name='relu_with_dropout'), counter

conv1, CONV_LAYER_COUNTER = conv_layer(X, NUM_CHANNELS, K_C1, L_C1, STR_C1,
                                       CONV_LAYER_COUNTER)
pool1, POOL_LAYER_COUNTER = maxpool_layer(conv1, KS_P1, STR_P1,
                                          POOL_LAYER_COUNTER)
conv2, CONV_LAYER_COUNTER = conv_layer(pool1, L_C1, K_C2, L_C2, STR_C2,
                                       CONV_LAYER_COUNTER)
pool2, POOL_LAYER_COUNTER = maxpool_layer(conv2, KS_P2, STR_P2,
                                          POOL_LAYER_COUNTER)
fc1, FULLCON_LAYER_COUNTER = fullconn_layer(pool2, IMAGE_HEIGHT, IMAGE_WIDTH,
                                            STR_C1, STR_P1, STR_C2, STR_P2,
                                            L_C2, L_FC, FULLCON_LAYER_COUNTER)

# Output building

with tf.variable_scope('sigmoid_linear') as scope:
    # Create weights and biases for the final fully-connected layer
    w = tf.get_variable('weights', [L_FC, N_CLASSES],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [N_CLASSES],
                        initializer=tf.random_normal_initializer())
    # Compute logits through a simple linear combination
    logits = tf.add(tf.matmul(fc1, w), b)
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
    # tensorboard --logdir="../graphs/"+NETWORK_NAME --port 6006)
    sess.run(tf.global_variables_initializer())
    # Declare a saver instance and a summary writer to store the trained network
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('../graphs/'+NETWORK_NAME, sess.graph)
    initial_step = global_step.eval(session=sess)
    # Create folders to store checkpoints
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('../checkpoints/'+NETWORK_NAME+'/checkpoint'))
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
            saver.save(sess, '../checkpoints/'+NETWORK_NAME+'/epoch', index)
        sess.run(optimizer, feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT})
    logger.info("Optimization Finished!")
    logger.info("Total time: {:.2f} seconds".format(time.time() - start_time))
    
    # The results are stored as a pandas dataframe and saved on the file system.
    param_history = pd.DataFrame({"epoch":epoches, "loss":losses,
                                  "accuracy":accuracies})
    param_history = param_history.set_index("epoch")
    if initial_step == 0:
        param_history.to_csv(NETWORK_NAME+".csv", index=True)
    else:
        param_history.to_csv(NETWORK_NAME+".csv",
                             index=True,
                             mode='a',
                             header=False)
    # Stop the threads used during the process
    coord.request_stop()
    coord.join(threads)
