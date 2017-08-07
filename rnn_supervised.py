import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


class LSTMSupervisedModel:
    def __init__(self, seq_len, input_dim, num_class, hidden_size=256,
                 learning_rate=1e-3):
        self.input_x = tf.placeholder(tf.float32, [None, seq_len, input_dim])
        self.input_y = tf.placeholder(tf.int64, [None])

        rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, state_keep_prob=0.7)

        _, rnn_output = tf.nn.dynamic_rnn(rnn_cell, self.input_x,
                                          dtype=tf.float32)
        self.logit = tf.layers.dense(rnn_output, units=num_class,
                                     activation=None)

        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.input_y, logits=self.logit)

        self.accuracy = tf.equal(self.input_y, tf.arg_max(self.logit, -1))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in
                      gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)


if __name__ == '__main__':
    seq_len = 28
    input_dim = 28
    num_class = 10
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False,
                                      reshape=False)

    with tf.Graph().as_default(), tf.Session() as sess:
        model = LSTMSupervisedModel(28, 28, 10)
        sess.run(tf.global_variables_initializer())

        for epoch_id in range(50):

            train_acc = []

            for iter_id in range(100):
                batch_xs, batch_ys = mnist.train.next_batch(600)
                batch_xs = batch_xs[:, :, :, 0]

                _, acc_ins = sess.run(
                    [model.train_op, model.accuracy],
                    feed_dict={
                        model.input_x: batch_xs,
                        model.input_y: batch_ys
                    }
                )
                train_acc += list(acc_ins)

            print(epoch_id, np.mean(train_acc), end=' ')

            test_acc = []
            for iter_id in range(100):
                batch_xs, batch_ys = mnist.test.next_batch(100)
                batch_xs = batch_xs[:, :, :, 0]

                acc_ins = sess.run(
                    model.accuracy,
                    feed_dict={
                        model.input_x: batch_xs,
                        model.input_y: batch_ys
                    }
                )

                test_acc += list(acc_ins)

            print(np.mean(test_acc))
