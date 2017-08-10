import tensorflow as tf
import numpy as np
import pickle
from input_pipeline import supervised_batch


def _scale_l2(x, norm_length):
    alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(
        tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit


def rnn_loss(x, y, hidden_size, num_class, scope='RNN', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, state_keep_prob=0.7)

        _, rnn_output = tf.nn.dynamic_rnn(
            rnn_cell, x, dtype=tf.float32, scope='RNN')
        logit = tf.layers.dense(rnn_output, units=num_class, name='fc1',
                                activation=None)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logit)
        return logit, loss


class LSTMSupervisedModel:
    def __init__(self, seq_len, input_dim, num_class, hidden_size=256,
                 learning_rate=1e-3):
        self.input_x = tf.placeholder(tf.float32, [None, seq_len, input_dim])
        self.input_y = tf.placeholder(tf.int64, [None])
        self.perturb_norm_length = tf.placeholder(tf.float32, [])

        self.logit, o_loss = rnn_loss(
            self.input_x, self.input_y, hidden_size, num_class
        )

        grad, = tf.gradients(
            o_loss, self.input_x,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        perturb = _scale_l2(grad, self.perturb_norm_length)

        _, self.loss = rnn_loss(
            self.input_x + perturb, self.input_y, hidden_size, num_class,
            reuse=True
        )

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

    load_dict = pickle.load(open('data/semi_supervised.p', 'rb'))
    train_x, train_y = load_dict['x_labelled'], load_dict['y_labelled']
    test_x, test_y = load_dict['x_test'], load_dict['y_test']

    with tf.Graph().as_default(), tf.Session() as sess:
        model = LSTMSupervisedModel(28, 28, 10)
        sess.run(tf.global_variables_initializer())

        for epoch_id in range(10000):
            perturb_norm_length = 3

            train_acc = []

            for batch_xs, batch_ys in supervised_batch(500, train_x, train_y):
                _, acc_ins = sess.run(
                    [model.train_op, model.accuracy],
                    feed_dict={
                        model.input_x: batch_xs,
                        model.input_y: batch_ys,
                        model.perturb_norm_length: perturb_norm_length
                    }
                )
                train_acc += list(acc_ins)


            test_acc = []

            for batch_xs, batch_ys in supervised_batch(1000, test_x, test_y):
                acc_ins = sess.run(
                    model.accuracy,
                    feed_dict={
                        model.input_x: batch_xs,
                        model.input_y: batch_ys
                    }
                )

                test_acc += list(acc_ins)

            print('\r', epoch_id, np.mean(train_acc),
                  np.mean(test_acc), end='', flush=True)
