import tensorflow as tf

class Randomer(object):
    stddev = None

    @staticmethod
    def random_normal(wshape):
        return tf.random.normal(wshape, stddev=Randomer.stddev)

    @staticmethod
    def set_stddev(sd):
        Randomer.stddev = sd