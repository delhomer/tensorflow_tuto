# Linear regression with Tensorflow

# This notebook will show how to make a linear regression with the help of the
# `Tensorflow` library. As an example, a set of OpenStreetMap elements gathered
# around Bordeaux will be used. The goal of the notebook is to predict the
# number of contributors for each element, starting from a set of other element
# characteristics.

# Step 0: module imports

# `matplotlib` will be used to plot regression results, `os` is necessary for
# relative path handling, then `pandas` is used to handle the input dataframe,
# and of course, `tensorflow` will be needed to do the regression.

import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf

# Step 1: data recovering and preparation

# The used data describes a set of OpenStreetMap elements, we admit it is
# available on the computer

data = pd.read_csv("/home/rde/data/osm-history/output-extracts/bordeaux-metropole/element-metadata.csv", index_col=0)
data.shape

# We have 2760999 individuals in this table, described by 17 different
# features. One can provide a short extract of this dataset:

print( data.sample(2).T )

# In this study, we will consider the number of contributors as the output to
# predict. We select a small set of features as predictors: the number of days
# between first and last modifications, the number of days since first
# modification, the number of days during which modifications arised, the last
# version, the number of change sets and the numbers of autocorrections and
# corrections.

data_x = data[["lifespan", "n_inscription_days", "n_activity_days", "version", "n_chgset", "n_autocorr", "n_corr"]].values
data_y = data[["n_user"]].values

print(data_x.shape)
print(data_y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                    test_size=0.1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Step 2: Parameter settings

learning_rate = 0.01
training_epochs = 15000
display_step = 50

# Step 3: Prepare the checkpoint creation

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
make_dir('checkpoints')
make_dir('checkpoints/linreg_osm')

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# Step 4: Tensorflow model design

# We need to tensors, *i.e.* one for inputs and one for outputs.

with tf.name_scope("data"):
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")

# The linear regression is defined through weights and biases, that are set as
# tensorflow variables, and injected into the output variable `predictions`. A
# hidden layer is added to

with tf.name_scope("linear_reg"):
    w = tf.get_variable('weights', [data_x.shape[1], 1], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0.0))
    predictions = tf.add(tf.matmul(X, w), b)

# The cost function is the sum of squares of differences between predictions
# and true outputs. A regularization term is added to this value.

with tf.name_scope('loss'):
    # loss function + regularization value
    loss = tf.reduce_sum(tf.square(predictions - Y)) + 0.01 * tf.nn.l2_loss(w)

# We use Adam optimizer to update the model variable (other choices:
# GradientDescentOptimizer, AdadeltaOptimizer, AdagradOptimizer...)

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# Step 5: Variable initialization

init = tf.global_variables_initializer()

# Final step: running the model

# First we have to open a new session, initialize the variable, and prepare the
# graph and checkpoint utilities:

session = tf.Session()
session.run(init)

saver = tf.train.Saver()
writer = tf.summary.FileWriter('./graphs/linreg', session.graph)
##### You have to create folders to store checkpoints
ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/linreg_osm/checkpoint'))
# if that checkpoint exists, restore from checkpoint
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(session, ckpt.model_checkpoint_path)

initial_step = global_step.eval(session=session)

# The model is ready to be trained. We proceed to as many training steps as
# indicated by the previous parametrization.

costs = list()
weights = list()
biases = list()
print("### Training step ###")
for epoch in range(initial_step, training_epochs):
    session.run(train_op, feed_dict={X: x_train, Y: y_train})
    #Display logs per epoch step
    if (epoch+1) % display_step == 0:
        training_cost, weight, bias = session.run([loss, w, b], feed_dict={X: x_train, Y: y_train})
        print("*** Epoch", '%04d' % (epoch+1), "cost={}\nn_user = {:.3f}*X1 + {:.3f}*X2 + {:.3f}*X3 + {:.3f}*X4 + {:.3f}*X5 + {:.3f}*X6 + {:.3f}*X7 + {:.3f} ***".format(training_cost, weight[0][0], weight[1][0], weight[2][0], weight[3][0], weight[4][0], weight[5][0], weight[6][0], bias[0]))
        costs.append(training_cost)
        weights.append(weight[:,0])
        biases.append(bias[0])
        saver.save(session, 'checkpoints/linreg_osm/epoch', epoch)

# The results are stored into a pandas dataframe and saved onto the file
# system.

param_history = pd.DataFrame(weights, columns=["lifespan", "n_inscription_days", "n_activity_days", "version", "n_chgset", "n_autocorr", "n_corr"])
param_history["bias"] = biases
param_history["loss"] = costs

if initial_step == 0:
    param_history.to_csv("linreg_osm.csv", index=False)
else:
    param_history.to_csv("linreg_osm.csv", index=False, mode='a', header=False)

# Then the data into this `csv` file is recovered. We do not consider the
# current `param_history` value, as it may represent only the last training
# steps, if the train just had been restored from a checkpoint. We use the
# results for plotting purpose.

param_history = pd.read_csv("linreg_osm.csv", index_col=False)

# Graphic display
f, ax = plt.subplots(3, 3, figsize=(12,6))
for i in range(param_history.shape[1]):
    ax[i % 3][int(i / 3)].plot(param_history.iloc[:,i])
    ax[i % 3][int(i / 3)].set_title(param_history.columns[i])
f.tight_layout()
f.show()

# Then the model is run with test data (this dataset was not used for model
# training). The goal is to evaluate the correspondance between true value of
# `y` and the model prediction.

print("### Test step ###")
cost, y_pred = session.run([loss, predictions], feed_dict={X: x_test, Y: y_test})
print("""Test cost = {}, i.e. +/- {:.3f} contributor(s) per OSM elements on
average""".format(cost, math.sqrt(cost/len(y_test))))

# A last plot is produced starting from the test step: it shows how good the
# predictions are.

plt.plot(y_test, y_pred, 'go')
output_min, output_max = int(min(y_pred)[0]), int(max(y_pred)[0])
plt.plot(range(output_min, output_max+2), range(output_min, output_max+2))
plt.xlabel("True values of y")
plt.ylabel("Model predictions")
plt.xlim(min(y_test)[0], max(y_test)[0]+2)
plt.ylim(output_min, output_max+2)
plt.tight_layout()
plt.show()

# Last the tensorflow session is closed.

session.close()
