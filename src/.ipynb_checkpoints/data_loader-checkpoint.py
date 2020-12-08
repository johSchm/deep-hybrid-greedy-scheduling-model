import pandas as pd 
from tqdm import tqdm 
import numpy as np
import tensorflow as tf 
import sklearn.model_selection as sms


def load_data(x_path, y_path):

    df = pd.DataFrame(pd.read_pickle(x_path))
    x = []
    for example in tqdm(range(df.shape[0])):
        x.append([df['j' + str(job)][example] for job in range(df.shape[1])])
    x = np.array(x, np.float32)

    df = pd.DataFrame(pd.read_pickle(y_path))
    y = []
    for example in tqdm(range(df.shape[0])):
        y_m = []
        for m in range(df.shape[1]):
            j_length = len(df['m' + str(m)][example].keys())
            y_m.append([df['m' + str(m)][example]['j' + str(j)] for j in range(j_length)])
        y.append(y_m)
    y = np.array(y, np.float32)

    # one hot encoding of labels y
    if y.shape[-1] == 3:
        M = config_param.get("data_shape")[1][1]
        J = y.shape[-2]
        y_data_one_hot = np.zeros([len(y), M, J, J])
        for i in tqdm(range(len(y))):
            for m in range(M):
                for j in range(J):
                    one_hot = np.zeros(J)
                    idx = int(y[i, m, j, 0])
                    one_hot[idx] = 1 if idx != -1 else 0
                    y_data_one_hot[i, m, j] = one_hot
        y = np.transpose(y_data_one_hot, [0, 1, 3, 2])

    return x, y


def generator(dataset):
    while True:
        for data in dataset:
            yield data[0], data[1]
            
            
def preprocess(x, y, batch_size=8, split=(0.8, 0.1, 0.1)):

    x_train, x_test, y_train, y_test = sms.train_test_split(
        x, y, test_size=split[1]+split[2], random_state=42)
    x_test, x_val, y_test, y_val = sms.train_test_split(
        x_test, y_test, test_size=split[2], random_state=42)
        
    properties = {
        'x_mean': np.mean(x_train[:, :, :2]),
        'x_stdd': np.std(x_train[:, :, :2]),
        'x_min': np.amin(x_train[:, :, :2]),
        'x_max': np.amax(x_train[:, :, :2]),

        'x_pt_mean': np.mean(x_train[:, :, 0]),
        'x_pt_stdd': np.std(x_train[:, :, 0]),
        'x_pt_min': np.amin(x_train[:, :, 0]),
        'x_pt_max': np.amax(x_train[:, :, 0]),

        'x_dd_mean': np.mean(x_train[:, :, 1]),
        'x_dd_stdd': np.std(x_train[:, :, 1]),
        'x_dd_min': np.amin(x_train[:, :, 1]),
        'x_dd_max': np.amax(x_train[:, :, 1]),
        
        'train_size': x_train.shape[0],
        'val_size': x_val.shape[0],
        'test_size': x_test.shape[0],
    }

    x_train = tf.convert_to_tensor(x_train, tf.float32)
    y_train = tf.convert_to_tensor(y_train, tf.float32)
    x_val = tf.convert_to_tensor(x_val, tf.float32)
    y_val = tf.convert_to_tensor(y_val, tf.float32)
    x_test = tf.convert_to_tensor(x_test, tf.float32)
    y_test = tf.convert_to_tensor(y_test, tf.float32)

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train = train.batch(batch_size, drop_remainder=True).shuffle(100)
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val = val.batch(batch_size, drop_remainder=True).shuffle(100)
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test = test.batch(batch_size, drop_remainder=True).shuffle(100)

    return generator(train), generator(val), generator(test), properties
