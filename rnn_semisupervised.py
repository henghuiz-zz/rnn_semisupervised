import tensorflow as tf
import pickle
import numpy as np

from input_pipeline import semisupervised_batch, supervised_batch, unsupervised_batch


def classifier(input_x, sequence_length, num_class, hidden_size=256, scope='cls'):
    with tf.variable_scope(scope):
        rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, state_keep_prob=0.7)

        _, rnn_output = tf.nn.dynamic_rnn(
            rnn_cell, input_x, sequence_length=sequence_length, dtype=tf.float32, scope='RNN')
        logit = tf.layers.dense(rnn_output, units=num_class, activation=None)
    return logit


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


class SemisupervisedModel:
    def __init__(self, sequence_length, input_dim, embed_dim, num_class):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
        self.input_y = tf.placeholder(tf.int64, [None])
        self.input_l = tf.placeholder(tf.int64, [None])
        self.input_m = tf.placeholder(tf.float32, [None, sequence_length])
        mask = tf.expand_dims(self.input_m[:, 1:], -1)
        self.train_stepsize = tf.placeholder(tf.float32, [])

        self.predict_logit = classifier(self.input_x, self.input_l, num_class)
        self.sup_loss = tf.losses.sparse_softmax_cross_entropy(self.input_y, self.predict_logit)

        broadcast_helper = tf.ones([1, sequence_length - 1], dtype=tf.float32)
        self.context = tf.tensordot(tf.expand_dims(self.predict_logit, -1), broadcast_helper, axes=[[-1], [0]])
        self.context = tf.transpose(self.context, [0, 2, 1])

        self.encoded_state = encoder_model(self.input_x, embed_dim, self.input_l)

        decoder_input = tf.concat([self.input_x[:, :-1, :], self.context], axis=-1)

        self.prediction = decoder_model(decoder_input,
                                        self.encoded_state,
                                        embed_dim,
                                        self.input_l - 1,
                                        input_dim)

        self.uns_loss = tf.losses.mean_squared_error(self.input_x[:, 1:, :],
                                                 self.prediction,
                                                 weights=mask)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.train_stepsize)
        gvs = optimizer.compute_gradients(self.uns_loss)
        capped_gvs = [
            (tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.uns_train = optimizer.apply_gradients(capped_gvs)

        optimizer2 = tf.train.AdamOptimizer(learning_rate=self.train_stepsize)
        gvs2 = optimizer2.compute_gradients(self.uns_loss + self.sup_loss)
        capped_gvs2 = [
            (tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs2]
        self.sup_train = optimizer.apply_gradients(capped_gvs2)

        self.accuracy = tf.equal(self.input_y, tf.arg_max(self.predict_logit, -1))


if __name__ == '__main__':
    seq_len = 28
    input_dim = 28
    num_class = 10
    load_dict = pickle.load(open('data/semi_supervised.p', 'rb'))
    unlabelled_x, labelled_x, test_x = load_dict['x_unlabelled'], load_dict['x_labelled'], load_dict['x_test']
    labelled_y, test_y = load_dict['y_labelled'], load_dict['y_test']

    with tf.Graph().as_default(), tf.Session() as sess:
        model = SemisupervisedModel(seq_len + 2, input_dim + 2, 200, num_class)
        sess.run(tf.global_variables_initializer())

        for epoch_id in range(100):
            step_size = 1e-3 if epoch_id < 50 else 1e-4

            # first train on labelled data
            train_acc = []
            train_loss = []
            for batch_x, batch_l, batch_m in unsupervised_batch(100, unlabelled_x):
                for batch_xs, batch_ys, batch_ls, batch_ms in semisupervised_batch(100, labelled_x, labelled_y):
                    _, acc_ins = sess.run(
                        [model.sup_train, model.accuracy],
                        feed_dict={
                            model.input_x: batch_xs,
                            model.input_y: batch_ys,
                            model.input_l: batch_ls,
                            model.input_m: batch_ms,
                            model.train_stepsize: step_size
                        }
                    )
                    train_acc += list(acc_ins)

                loss_ins, _ = sess.run(
                    [model.uns_loss, model.uns_train],
                    feed_dict={
                        model.input_x: batch_x,
                        model.input_l: batch_l,
                        model.input_m: batch_m,
                        model.train_stepsize: step_size
                    }
                )
                print('\r', np.mean(acc_ins), np.mean(loss_ins), end='', flush=True)
                train_loss.append(loss_ins)

            print('\r', epoch_id, np.mean(train_acc), np.mean(train_loss), end=' ')

            # finally, test on test data
            test_acc = []

            for batch_xs, batch_ys, batch_l, _ in semisupervised_batch(100, test_x, test_y):
                acc_ins = sess.run(
                    model.accuracy,
                    feed_dict={
                        model.input_x: batch_xs,
                        model.input_y: batch_ys,
                        model.input_l: batch_l
                    }
                )

                test_acc += list(acc_ins)
            print(epoch_id, np.mean(test_acc))



#
#     input_x = tf.placeholder(tf.float32, [None, seq_len, input_dim])
#
#     logit = classifier(input_x, 3)
#
#     broadcast_helper = tf.ones([1, seq_len - 1], dtype=tf.float32)
#     context = tf.tensordot(tf.expand_dims(logit, -1), broadcast_helper, axes=[[-1], [0]])
#     context = tf.transpose(context, [0, 2, 1])
#
#     print(context.get_shape())