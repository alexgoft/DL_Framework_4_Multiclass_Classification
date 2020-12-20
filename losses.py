import tensorflow as tf


def alex_loss(labels, logits):
    """
    Sigmoid cross entropy from logits with mean reduce.
    :param labels:
    :param logits:
    :return:
    """
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))
