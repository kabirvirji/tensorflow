"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# remember to activate tensorflow like 
# source ~/tensorflow/bin/activate

import tensorflow as tf 
import importlib

# flag arguments to change batch size and such
# to test out different things

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size divides evenly into dataset sizes.')
tf.app.flags.DEFINE_integer('num_iter', 1000, 'Number of iterations')
tf.app.flags.DEFINE_integer('n_print', 100, 'Print accuracy at every n_print iteration')
tf.app.flags.DEFINE_string('datafile', 'MNIST', 'Data processing file')
tf.app.flags.DEFINE_string('modelfile', 'softmax_MNIST', 'Data processing file')

# load model

model_module = importlib.import_module(('models.' + FLAGS.modelfile))
model = model_module.model()
data_module = importlib.import_module(('data.' + FLAGS.datafile))
data = data_module.dataset()

# same for loop as the initial tutorial code but with the flag batch sizes

def train():
    for i in range(FLAGS.num_iter):
        # load training batch
        batch = data.train_next_batch(FLAGS.batch_size)
        # take training step with input target arg
        model.train_step(batch[0], batch[1])
        # print accuracy 
        if i % FLAGS.n_print == 0:
            print("Iteration", i, model.calculate_acc(data.test_x(), data.test_y()))
    print(model.calculate_acc(data.test_x(), data.test_y())) # last one is this one

def main(argv):
    train()

if __name__ == '__main__':
    tf.app.run()
