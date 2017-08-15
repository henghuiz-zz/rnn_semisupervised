import tensorflow as tf
import numpy as np
import pickle
from input_pipeline import supervised_batch


class LSTMSupervisedModel:
    def __init__(self, seq_len, input_dim, num_class, hidden_size=256,
                 learning_rate=1e-3):
        self.input_x = tf.placeholder(tf.float32, [None, seq_len, input_dim])
        self.input_y = tf.placeholder(tf.int64, [None])

        rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, state_keep_prob=0.9)

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

    load_dict = pickle.load(open('data/semi_supervised.p', 'rb'))
    train_x, train_y = load_dict['x_labelled'], load_dict['y_labelled']
    test_x, test_y = load_dict['x_test'], load_dict['y_test']

    with tf.Graph().as_default(), tf.Session() as sess:
        model = LSTMSupervisedModel(28, 28, 10)
        sess.run(tf.global_variables_initializer())

        for epoch_id in range(10000):

            train_acc = []

            for batch_xs, batch_ys in supervised_batch(500, train_x, train_y):

                _, acc_ins = sess.run(
                    [model.train_op, model.accuracy],
                    feed_dict={
                        model.input_x: batch_xs,
                        model.input_y: batch_ys
                    }
                )
                train_acc += list(acc_ins)
            print('\r', epoch_id, 'train', np.mean(train_acc), end='', flush=True)

            if epoch_id % 200 == 0:
                test_acc = []

                for batch_xs, batch_ys in supervised_batch(100, test_x, test_y):

                    acc_ins = sess.run(
                        model.accuracy,
                        feed_dict={
                            model.input_x: batch_xs,
                            model.input_y: batch_ys
                        }
                    )

                    test_acc += list(acc_ins)
                print('\r', epoch_id, 'train', np.mean(train_acc), 'test', np.mean(test_acc))
