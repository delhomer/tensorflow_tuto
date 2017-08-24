# Introduction to Tensorflow

# Tensorflow is a Python library designed for numerical computations, it is
# linked with data flow graphs as a smart representation of computation process
# (nodes are operations, edges are data arrays).

# Tensorflow is very similar to Numpy regarding the numerical operations. In
# this way we will handle here some constants and variables of different shapes
# (scalars, vectors, matrices). Some Tensorflow specificities will arise as well.

import tensorflow as tf

# Constant and basic operations

# Operations on scalar

a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
add = tf.add(a, b, name="addition")
mul = tf.multiply(a, b, name="multiplication")

sess = tf.Session()
a_, b_, x, y = sess.run([a, b, add, mul])
print("{0} + {1} = {2}\n{0} * {1} = {3}".format(a_, b_, x, y))
sess.close()

# Operations on matrices

m1 = tf.constant([[2,2]], name="m1")
m2 = tf.constant([[0,1],[2,3]], name="m2")
madd = tf.add(m1, m2, name="matrix_addition")
mmul = tf.matmul(m1, m2, name="matrix_multiplication")

with tf.Session() as sess:
    a_, b_, x, y = sess.run([m1, m2, madd, mmul])
    print("{0}\n+\n{1}\n=\n{2}\n\n{0}\n*\n{1}\n=\n{3}".format(a_, b_, x, y))

# Particular matrices
    
mzeros = tf.zeros([2,3], tf.int32)
mzeros_temp = tf.zeros_like(m2)
mones = tf.ones([2,3], tf.int32)
meights = tf.fill([2,3], 8)
with tf.Session() as sess:
    mzeros_, mzeros_t_, mones_, meights_ = sess.run([mzeros, mzeros_temp, mones, meights])
    print("""Matrice nulle:\n{0}\nMatrice nulle basée sur un
    modèle:\n{1}\nMatrice de '1':\n{2}\nMatrice de '8':\n{3}""".format(mzeros_,
                                                                       mzeros_t_,
                                                                       mones_,
                                                                       meights_))
# Random generator

mu = 10
sigma = 1
mrand = tf.random_normal([3,3], mean=mu, stddev=sigma)
with tf.Session() as sess:
    print("Matrice normale ({0},{1}):\n{2}".format(mu, sigma, sess.run(mrand)))

# Variables

v1 = tf.Variable(2, name="scalar")
v2 = tf.Variable([3.0, 4.0], name="vector")
v3 = tf.Variable(tf.zeros([2,6]), name="zeromatrix")

init = tf.variables_initializer([v1, v3], name="init_az")

with tf.Session() as sess:
    sess.run(init)
    print(v1)
    print(v1.eval())
    print(v3.eval())
    sess.run(v2.initializer)
    print(v2.eval())

new_v2 = v2.assign([30.0, 40.0])
with tf.Session() as sess:
    sess.run(v2.initializer)
    print(v2.eval())
    sess.run(new_v2)
    print(v2.eval())

# Placeholders

# Placeholders are a kind of variables that must be fed during a Tensorflow
# session. They are the ideal structure to insert input data into computation
# process.

# Placeholder definition

p1 = tf.placeholder(tf.float32, shape=[1,3], name="placeholder1")
p2 = tf.placeholder(tf.float32, shape=[3,1], name="placeholder2")

# Operation on placeholders to make an output

pmult = tf.matmul(p1, p2) + tf.ones(shape=[1,1])

# Executinon of the code within a session

with tf.Session() as sess:
    placeholder_mult = sess.run(pmult, feed_dict={p1:[[1,2,3]], p2:[[10],[20],[30]]})
    print("Sum of placeholder matrices:\n{}".format(placeholder_mult))

# Graph visualisation

# A dummy sequence of operations is designed, to elaborate a small example graph.
import numpy as np

p3 = tf.placeholder(tf.float32, shape=(), name="placeholder3")
p4 = tf.placeholder(tf.float32, shape=(), name="placeholder4")

padd = tf.add(pmult, p3)
pequals = tf.equal(padd, p4)
panswer = tf.reshape(pequals, [])

with tf.Session() as sess:
    writer = tf.summary.FileWriter('../graphs/intro', sess.graph)
    answer = sess.run(panswer, feed_dict={p1: [[1, 2, 3]], p2: [[10], [20], [30]], p3: 1000, p4: 1141})
    print("Does the fourth value equal the result obtained with the three others? {}".format(answer))

# Then the graph can be showed onto the local server, with the proper port. The
# shell command is the following one:

# tensorboard --logdir '../graphs/intro' --port 6006

# TD;LR
# Tensorflow:
# - allows to handle constants and variables of different shapes
# - represents operations in a graph
# - separates the graph conception from the numerical operations
# - needs the opening of a dedicated session to apply operations within a
# pre-built graph
# - and its tensorboard command show the graph and every tensorflow objects on
# the local server
