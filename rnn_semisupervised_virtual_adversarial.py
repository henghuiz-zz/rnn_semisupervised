import tensorflow as tf
import numpy as np
import pickle
from input_pipeline import semisupervised_batch
from util import _scale_l2, _kl_divergence_with_logits


GRADIENT_METHOD = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N


def rnn_loss(x, hidden_size, num_class, scope='rnn', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size)

        _, rnn_output = tf.nn.dynamic_rnn(
            rnn_cell, x, dtype=tf.float32, scope='RNN')

        logit = tf.layers.dense(rnn_output, units=num_class, name='fc1',
                                activation=None)
        return logit


class LSTMSupervisedModel:
    def __init__(self, seq_len, input_dim, num_class,
                 hidden_size=256,
                 learning_rate=1e-3,
                 num_power_iteration=1,
                 small_constant_for_finite_diff=0.1,
                 perturb_norm_length=1,
                 weight_lambda=0.2):
        self.input_x = tf.placeholder(tf.float32, [None, seq_len, input_dim])
        self.input_y = tf.placeholder(tf.int64, [None])

        self.logit = rnn_loss(self.input_x, hidden_size, num_class)
        self.accuracy = tf.equal(self.input_y, tf.argmax(self.logit, -1))

        loss_1 = tf.losses.sparse_softmax_cross_entropy(
            self.input_y, self.logit)

        logits = tf.stop_gradient(self.logit)
        d = tf.random_normal(shape=tf.shape(self.input_x))

        for _ in range(num_power_iteration):
            d = _scale_l2(d, small_constant_for_finite_diff)
            d_logits = rnn_loss(
                self.input_x + d, hidden_size, num_class, reuse=True)
            kl = _kl_divergence_with_logits(logits, d_logits, num_class)
            d, = tf.gradients(kl, d, aggregation_method=GRADIENT_METHOD)
            d = tf.stop_gradient(d)

        perturb = _scale_l2(d, perturb_norm_length)
        vadv_logits = rnn_loss(
                self.input_x + perturb, hidden_size, num_class, reuse=True)
        self.uns_loss = _kl_divergence_with_logits(
            logits, vadv_logits, num_class)

        self.sup_loss = loss_1 + weight_lambda * self.uns_loss

        sup_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        sup_gvs = sup_opt.compute_gradients(self.sup_loss)
        sup_capped_gvs = [
            (tf.clip_by_value(grad, -1., 1.), var) for grad, var in sup_gvs]
        self.sup_train_op = sup_opt.apply_gradients(sup_capped_gvs)

        uns_optimizer = tf.train.AdamOptimizer(
            learning_rate=weight_lambda*learning_rate)
        uns_gvs = uns_optimizer.compute_gradients(self.uns_loss)
        uns_capped_gvs = [
            (tf.clip_by_value(grad, -1., 1.), var) for grad, var in uns_gvs]
        self.uns_train_op = uns_optimizer.apply_gradients(uns_capped_gvs)


if __name__ == '__main__':
    seq_len = 28
    input_dim = 28
    num_class = 10

    load_dict = pickle.load(open('data/semi_supervised.p', 'rb'))
    unlabelled_x = load_dict['x_unlabelled']
    labelled_x, test_x = load_dict['x_labelled'], load_dict['x_test']
    labelled_y, test_y = load_dict['y_labelled'], load_dict['y_test']

    save_path = 'model_ck/rnn_ck/'

    with tf.Graph().as_default(), tf.Session() as sess:
        model = LSTMSupervisedModel(28, 28, 10)
        sess.run(tf.global_variables_initializer())

        for epoch_id in range(10000):
            train_acc = []

            for batch_xs, batch_ys, _, _ in semisupervised_batch(
                    100, labelled_x, labelled_y):
                _, acc_ins = sess.run(
                    [model.sup_train_op, model.accuracy],
                    feed_dict={
                        model.input_x: batch_xs[:, 1:-1, 1:-1],
                        model.input_y: batch_ys
                    }
                )
                train_acc += list(acc_ins)
            # pick a batch of unlabbel example

            train_acc = np.mean(train_acc)
            if train_acc > 0.99:
                num_unlabelled = 20
            elif train_acc > 0.95:
                num_unlabelled = 10
            elif train_acc > 0.9:
                num_unlabelled = 5
            else:
                num_unlabelled = 2

            for _ in range(num_unlabelled):
                id_selected = np.random.choice(len(unlabelled_x), size=500)

                _ = sess.run(
                    model.uns_train_op,
                    feed_dict={
                        model.input_x: unlabelled_x[id_selected, :, :],
                    }
                )

            print('\r', epoch_id, 'train', train_acc, end='', flush=True)

            if epoch_id % 50 == 0:
                # start testing
                test_acc = []

                for batch_xs, batch_ys, _, _ in semisupervised_batch(
                        1000, test_x, test_y):
                    acc_ins = sess.run(
                        model.accuracy,
                        feed_dict={
                            model.input_x: batch_xs[:, 1:-1, 1:-1],
                            model.input_y: batch_ys
                        }
                    )

                    test_acc += list(acc_ins)
                test_acc = np.mean(test_acc)

                print('\r', epoch_id, 'train', train_acc, 'test', test_acc)
