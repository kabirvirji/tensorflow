from tensorflow.examples.tutorials.mnist import input_data

class dataset(object):

    def __init__(self):
        self.data = input_data.read_data_sets("data/MNIST_data/", one_hot=True)

    def train_next_batch(self, batch_size=100):
        return self.data.train.next_batch(batch_size)

    def train_x(self):
        return self.data.train.images

    def train_y(self):
        return self.data.train.labels

    def valid_next_batch(self, batch_size=100):
        return self.data.validation.next_batch(batch_size)

    def valid_x(self):
        return self.data.validation.images

    def valid_y(self):
        return self.data.validation.labels

    def test_next_batch(self, batch_size=100):
        return self.data.test.next_batch(batch_size)

    def test_x(self):
        return self.data.test.images

    def test_y(self):
        return self.data.test.labels