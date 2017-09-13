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
import sys
import time

import bpmll # Multilabel classification loss
import tensorflow_layers
import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
f = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(f)
logger.addHandler(ch)

# Step 1: parameter definition

# image dimensions (width, height, number of channels)
IMG_SIZE = (816, 612)
IMAGE_HEIGHT  = IMG_SIZE[1]
IMAGE_WIDTH   = IMG_SIZE[0]
NUM_CHANNELS  = 3 # Colored images (RGB)
# hidden layer depth (number of channel per convolutional and fully connected
# layer), kernel dimension, conv layer stride, max pool layer ksize and stride
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
L_FC = 512
# number of output classes
N_CLASSES = 66
# number of images per batch
BATCH_SIZE = 10
N_BATCHES = int(18000 / BATCH_SIZE) # TODO
# number of epochs (one epoch = all images have been used for training)
N_EPOCHS = 1
# Starting learning rate (it moves following an exponential decay afterwards)
START_LR = 0.01
# dropout, i.e. percentage of nodes that are briefly removed during training
# process
DROPOUT = 0.75
# printing frequency during training
SKIP_STEP = 10
# Name of the convolutional neural network
NETWORK_NAME = "cnn_mapillary"

# Step 2: data recovering

train_image_batch, train_label_batch, train_filename_batch = \
tensorflow_layers.prepare_data(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,
                               BATCH_SIZE, "training", "training_data_pipe")
validation_image_batch, validation_label_batch, validation_filename_batch =\
tensorflow_layers.prepare_data(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,
                               BATCH_SIZE, "validation", "validation_data_pipe")

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

conv1, CONV_LAYER_COUNTER = tensorflow_layers.conv_layer(X, NUM_CHANNELS, K_C1,
                                                         L_C1, STR_C1,
                                                         CONV_LAYER_COUNTER)
pool1, POOL_LAYER_COUNTER = tensorflow_layers.maxpool_layer(conv1, KS_P1,
                                                            STR_P1,
                                                            POOL_LAYER_COUNTER)
conv2, CONV_LAYER_COUNTER = tensorflow_layers.conv_layer(pool1, L_C1, K_C2,
                                                         L_C2, STR_C2,
                                                         CONV_LAYER_COUNTER)
pool2, POOL_LAYER_COUNTER = tensorflow_layers.maxpool_layer(conv2, KS_P2,
                                                            STR_P2,
                                                            POOL_LAYER_COUNTER)
fc1, FULLCON_LAYER_COUNTER = tensorflow_layers.fullconn_layer(pool2,
                                                              IMAGE_HEIGHT,
                                                              IMAGE_WIDTH,
                                                              STR_C1, STR_P1,
                                                              STR_C2, STR_P2,
                                                              L_C2, L_FC,
                                                              FULLCON_LAYER_COUNTER,
                                                              dropout)

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

with tf.name_scope('loss'):
    # Cross-entropy between predicted and real values: we use sigmoid instead
    # of softmax as we are in a multilabel classification problem
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(entropy, name="loss")

with tf.name_scope('accuracy'):
    # A prediction is correct when the rounded predicted output is equal to Y
    correct_prediction = tf.equal(tf.round(Y), tf.round(Ypredict))
    # Accuracy of the trained model, between 0 (worst) and 1 (best)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    bpmll_acc = bpmll.bp_mll_loss(Y, Ypredict)

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
            loss_batch, accuracy_batch, acc2 = sess.run([loss, accuracy, bpmll_acc], 
                                feed_dict={X: X_batch, Y:Y_batch, dropout: 1.0})
            logger.info("""Step {}: loss = {:5.3f}, accuracy = {:1.3f} (bpmll: {})""".format(index, loss_batch, accuracy_batch, acc2))
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
