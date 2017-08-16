import pickle
import numpy as np

from tensorflow.contrib.keras.python.keras.datasets import mnist
from sklearn.model_selection import StratifiedShuffleSplit

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255

# randomly get 1,000 training example
sss = StratifiedShuffleSplit(n_splits=1, test_size=100/60000, random_state=0)

unlabel_index, label_index = sss.split(x_train, y_train).__next__()

x_label, x_unlabel = x_train[label_index], x_train[unlabel_index]
y_label = y_train[label_index]

save_dict = {
    'x_labelled': x_label,
    'x_unlabelled': x_unlabel,
    'y_labelled': y_label,
    'x_test': x_test,
    'y_test': y_test
}

pickle.dump(save_dict, open('data/semi_supervised.p', 'wb'))
