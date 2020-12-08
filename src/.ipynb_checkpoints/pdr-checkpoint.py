import numpy as np
import tensorflow as tf


def random_sample():
    y_random = np.zeros([3, 9, 9])
    jobs = np.arange(9)
    np.random.shuffle(jobs)
    for j in range(9):
        pick_t = jobs[j]
        pick_m = np.random.randint(3)
        y_random[pick_m, j, pick_t] = 1
    y_random = np.expand_dims(y_random, axis=0)
    return tf.convert_to_tensor(y_random, tf.float32)


# earliest due date 
def edd(x):
    _x = x.numpy()[0]
    y_pick = np.zeros([3, 9, 9])
    m = 0
    for _ in range(9):
        t = np.sum(y_pick[m, :, :], axis=(-1,-2))
        j = np.argmin(_x[:,1])
        _x[j] = np.full(7, np.inf)
        m += 1 if m != 2 else -2
        y_pick[m, j, t.astype(int)] = 1
    y_pick = np.expand_dims(y_pick, axis=0)
    return tf.convert_to_tensor(y_pick, tf.float32)


# shortest processing time 
def spt(x):
    _x = x.numpy()[0]
    y_pick = np.zeros([3, 9, 9])
    m = 0
    for _ in range(9):
        t = np.sum(y_pick[m, :, :], axis=(-1,-2))
        j = np.argmin(_x[:,0])
        _x[j] = np.full(7, np.inf)
        m += 1 if m != 2 else -2
        y_pick[m, j, t.astype(int)] = 1
    y_pick = np.expand_dims(y_pick, axis=0)
    return tf.convert_to_tensor(y_pick, tf.float32)
