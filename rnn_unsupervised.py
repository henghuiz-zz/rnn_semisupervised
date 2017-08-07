import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def encoder_model(input_x, hidden_unit, sequence_length, scope='enc'):
    with tf.variable_scope(scope):
        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_unit)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, state_keep_prob=0.7)

        _, rnn_output = tf.nn.dynamic_rnn(
            rnn_cell, input_x, sequence_length=sequence_length,
            scope='RNN', dtype=tf.float32)
    return rnn_output


def decoder_model(input_x, encode_state, hidden_unit, sequence_length,
                  input_dim, scope='dec'):
    with tf.variable_scope(scope):
        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_unit)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, state_keep_prob=0.7)

        rnn_layer, _ = tf.nn.dynamic_rnn(
            rnn_cell, input_x, sequence_length=sequence_length,
            scope='RNN', dtype=tf.float32, initial_state=encode_state)

        fc1 = tf.layers.dense(rnn_layer, hidden_unit, activation=tf.nn.relu)
        output_layer = tf.layers.dense(fc1, input_dim, activation=None)

    return output_layer


class AutoEncoderModel:
    def __init__(self, sequence_length, input_dim, embed_dim):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
        self.input_l = tf.placeholder(tf.int64, [None])
        self.input_m = tf.placeholder(tf.float32, [None, sequence_length])
        mask = tf.expand_dims(self.input_m[:, 1:], -1)
        self.train_stepsize = tf.placeholder(tf.float32, [])

        self.encoded_state = encoder_model(self.input_x,
                                           embed_dim, self.input_l)
        self.prediction = decoder_model(self.input_x[:, :-1, :],
                                        self.encoded_state,
                                        embed_dim,
                                        self.input_l - 1,
                                        input_dim)

        self.loss = tf.losses.mean_squared_error(self.input_x[:, 1:, :],
                                                 self.prediction,
                                                 weights=mask)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.train_stepsize)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [
            (tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.train_step = optimizer.apply_gradients(capped_gvs)


if __name__ == '__main__':
    seq_len = 28
    input_dim = 28
    num_class = 10
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False,
                                      reshape=False)

    with tf.Graph().as_default(), tf.Session() as sess:
        model = AutoEncoderModel(seq_len + 2, input_dim + 2, embed_dim=256)
        sess.run(tf.global_variables_initializer())

        for epoch_id in range(100):

            step_size = 1e-3 if epoch_id < 50 else 1e-4

            train_loss = []

            for iter_id in range(100):
                batch_xs, _ = mnist.train.next_batch(600)
                batch_xs = batch_xs[:, :, :, 0]

                xs = np.zeros([600, seq_len+2, input_dim+2])
                xs[:, 1:-1, 2:] = batch_xs
                xs[:, 0, 0] = 1
                xs[:, -1, 1] = 1

                ls = (seq_len + 2) * np.ones([600])
                ms = (input_dim + 2) * np.ones([600, seq_len+2])

                loss_ins, _ = sess.run(
                    [model.loss, model.train_step],
                    feed_dict={
                        model.input_x: xs,
                        model.input_l: ls,
                        model.input_m: ms,
                        model.train_stepsize: step_size
                    }
                )

                train_loss.append(loss_ins)

            print(epoch_id, np.mean(train_loss), end=' ')

            test_loss = []
            for iter_id in range(100):
                batch_xs, _ = mnist.test.next_batch(100)
                batch_xs = batch_xs[:, :, :, 0]

                xs = np.zeros([100, seq_len + 2, input_dim + 2])
                xs[:, 1:-1, 2:] = batch_xs
                xs[:, 0, 0] = 1
                xs[:, -1, 1] = 1

                ls = (seq_len + 2) * np.ones([100])
                ms = (input_dim + 2) * np.ones([100, seq_len + 2])

                loss_ins = sess.run(
                    [model.loss],
                    feed_dict={
                        model.input_x: xs,
                        model.input_l: ls,
                        model.input_m: ms,
                        model.train_stepsize: step_size
                    }
                )

                test_loss.append(loss_ins)

            print(np.mean(test_loss))