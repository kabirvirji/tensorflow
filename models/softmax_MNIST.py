import tensorflow as tf

class model(object):

    def __init__(self, n_dim=784, n_targets=10, optimizer=tf.train.AdamOptimizer()):
        self.n_dim = n_dim
        self.n_targets = n_targets

        # input and output nodes
        self.x = tf.placeholder(tf.float32, shape=[None, n_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, n_targets])

        self.weights = self._init_weights()

        # define model
        self.y_ = tf.matmul(self.x, self.weights[0]) + self.weights[1]

        # define softmax cross entropy loss
        self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_)
        self.cross_entropy = tf.reduce_mean(self.cost)
        self.optimizer = optimizer.minimize(self.cross_entropy)

        # start session and init var
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

    def _init_weights(self):
        # weights and bias
        W = tf.Variable(tf.zeros([self.n_dim, self.n_targets]))
        b = tf.Variable(tf.zeros([self.n_targets]))
        return W, b

    def train_step(self, inputs_x, targets_y):
        self.optimizer.run(feed_dict={self.x: inputs_x, self.y: targets_y})

    def calculate_acc(self, inputs_x, targets_y):
        correct_prediction = tf.equal(tf.argmax(self.y_,1), tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy.eval(feed_dict={self.x: inputs_x, self.y: targets_y})