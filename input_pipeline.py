import numpy as np


def supervised_batch(batch_size, x, y, suffle=True):
    shuffle_indices = np.arange(len(y))
    if suffle:
        np.random.shuffle(shuffle_indices)
    start_ind = 0
    while start_ind < len(y):
        selected_index = shuffle_indices[start_ind: start_ind + batch_size]
        batch_y = y[selected_index]
        batch_x = x[selected_index]

        yield batch_x, batch_y
        start_ind += batch_size


def semisupervised_batch(batch_size, x, y, suffle=True):
    shuffle_indices = np.arange(len(y))
    if suffle:
        np.random.shuffle(shuffle_indices)
    start_ind = 0
    while start_ind < len(y):
        selected_index = shuffle_indices[start_ind: start_ind + batch_size]
        batch_y = y[selected_index]
        batch_x_pre = x[selected_index]

        batch_x = np.zeros([batch_x_pre.shape[0], batch_x_pre.shape[1] + 2, batch_x_pre.shape[2] + 2])
        batch_x[:, 1:-1, 2:] = batch_x_pre
        batch_x[:, 0, 0] = 1
        batch_x[:, -1, 1] = 1

        batch_l = (batch_x_pre.shape[1] + 2) * np.ones([batch_x_pre.shape[0]])
        batch_m = (batch_x_pre.shape[2] + 2) * np.ones([batch_x_pre.shape[0], batch_x_pre.shape[1] + 2])

        yield batch_x, batch_y, batch_l, batch_m
        start_ind += batch_size


def unsupervised_batch(batch_size, x, suffle=True):
    shuffle_indices = np.arange(len(x))
    if suffle:
        np.random.shuffle(shuffle_indices)
    start_ind = 0
    while start_ind < len(x):
        selected_index = shuffle_indices[start_ind: start_ind + batch_size]
        batch_x_pre = x[selected_index]

        batch_x = np.zeros([batch_x_pre.shape[0], batch_x_pre.shape[1] + 2, batch_x_pre.shape[2] + 2])
        batch_x[:, 1:-1, 2:] = batch_x_pre
        batch_x[:, 0, 0] = 1
        batch_x[:, -1, 1] = 1

        batch_l = (batch_x_pre.shape[1] + 2) * np.ones([batch_x_pre.shape[0]])
        batch_m = (batch_x_pre.shape[2] + 2) * np.ones([batch_x_pre.shape[0], batch_x_pre.shape[1] + 2])

        yield batch_x, batch_l, batch_m
        start_ind += batch_size
