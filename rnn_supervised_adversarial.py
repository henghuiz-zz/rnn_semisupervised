import tensorflow as tf
import numpy as np
import pickle
from input_pipeline import semisupervised_batch
from util import _scale_l2


def rnn_loss(x, y, hidden_size, num_class, scope='rnn', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, state_keep_prob=0.9)

        _, rnn_output = tf.nn.dynamic_rnn(
            rnn_cell, x, dtype=tf.float32, scope='RNN')
    with tf.variable_scope('fc', reuse=reuse):
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

    save_path = 'model_ck/rnn_ck/'

    with tf.Graph().as_default(), tf.Session() as sess:
        model = LSTMSupervisedModel(28, 28+2, 10)
        # rnn_variable = tf.get_collection(
        #     tf.GraphKeys.TRAINABLE_VARIABLES, 'rnn')
        #
        # rnn_variable_saver = tf.train.Saver(rnn_variable)
        sess.run(tf.global_variables_initializer())

        # ck_state = tf.train.get_checkpoint_state(save_path)
        # rnn_variable_saver.restore(sess, ck_state.model_checkpoint_path)

        for epoch_id in range(10000):
            perturb_norm_length = 5

            train_acc = []

            for batch_xs, batch_ys, _, _ in semisupervised_batch(500, train_x, train_y):
                _, acc_ins = sess.run(
                    [model.train_op, model.accuracy],
                    feed_dict={
                        model.input_x: batch_xs[:, 1:-1, :],
                        model.input_y: batch_ys,
                        model.perturb_norm_length: perturb_norm_length
                    }
                )
                train_acc += list(acc_ins)
            print('\r', epoch_id, 'train', np.mean(train_acc), end='', flush=True)

            if epoch_id % 200 == 0:
                test_acc = []

                for batch_xs, batch_ys, _, _ in semisupervised_batch(1000, test_x, test_y):
                    acc_ins = sess.run(
                        model.accuracy,
                        feed_dict={
                            model.input_x: batch_xs[:, 1:-1, :],
                            model.input_y: batch_ys
                        }
                    )

                    test_acc += list(acc_ins)
                print('\r', epoch_id, 'train', np.mean(train_acc), 'test', np.mean(test_acc))
