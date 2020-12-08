import tensorflow as tf


def count_setup_changes(x, y):
    cumulative_penalties = np.zeros(tf.shape(y)[0])
    for m in tf.range(tf.shape(y)[1]):
        Y_m, R_m = y[:, m, :, :], x[:, :, 2:]
        omega_m = tf.matmul(R_m, Y_m, transpose_a=True)
        for b, setup_batch_elem in enumerate(tf.unstack(omega_m, axis=0)):
            prior_valid_setup = tf.zeros(tf.shape(R_m)[-1], tf.float32)
            for setup in tf.unstack(setup_batch_elem, axis=-1):
                if tf.math.greater_equal(tf.reduce_sum(setup), tf.constant(0.5)):
                    if tf.reduce_any(tf.not_equal(setup, prior_valid_setup)):
                        cumulative_penalties[b] += 1
                    prior_valid_setup = setup
    return tf.convert_to_tensor(cumulative_penalties, tf.float32)


def setup_change_penalty(x, y, setup_change_penalty=10.0):
    cumulative_penalties = []
    for m in tf.range(tf.shape(y)[1]):
        cumulative_penalties.append([])
        Y_m, R_m = y[:, m, :, :], x[:, :, 2:]
        omega_m = tf.matmul(R_m, Y_m, transpose_a=True)
        for b, setup_batch_elem in enumerate(tf.unstack(omega_m, axis=0)):
            if len(cumulative_penalties[m]) <= b:
                cumulative_penalties[m].append(0.0)
            prior_valid_setup = tf.zeros(tf.shape(R_m)[-1], tf.float32)
            for setup in tf.unstack(setup_batch_elem, axis=-1):
                if tf.math.greater_equal(tf.reduce_sum(setup), tf.constant(0.5)):
                    if tf.reduce_any(tf.not_equal(setup, prior_valid_setup)):
                        if tf.math.greater_equal(tf.reduce_sum(prior_valid_setup), tf.constant(0.5)):
                            cumulative_penalties[m][b] += setup_change_penalty
                    prior_valid_setup = setup
    return tf.transpose(tf.convert_to_tensor(cumulative_penalties, tf.float32), [1,0])


def makespan(x, y):
    """ Computes makespan.
    :param x: un-normalized
    :return mean of makespan over all machines.
    """
    total_ms = []
    penalties = setup_change_penalty(x, y, 10.0)
    y = tf.transpose(y, [0, 1, 3, 2])
    for m in tf.range(3):
        processing_times = tf.matmul(y[:, m, :, :], x)
        cumulated_pts = tf.reduce_sum(processing_times, axis=-2)
        total_ms.append(cumulated_pts[:, 0])
    return tf.transpose(tf.convert_to_tensor(total_ms), [1,0]) + penalties


def lateness(x, y):
    """ Computes tardinesss.
    :param x: un-normalized
    :return mean of makespan over all machines.
    """
    total_ln = []
    penalties = setup_change_penalty(x, y, 10.0)
    y = tf.transpose(y, [0, 1, 3, 2])
    for m in tf.range(3):
        # the resulting vector contains the due dates of the allocated jobs in the correct order
        d_on_m = tf.matmul(y[:, m, ...], x[:, :, 1, tf.newaxis])
        d_on_m = tf.squeeze(d_on_m)
        
        # computes the cum sum of all processing times w.r.t.,to the machine allocation
        # thus the resulting matrix contains the intrinsic time per machine
        p_on_m = tf.squeeze(tf.matmul(y[:, m, ...], x[:, :, 0, tf.newaxis]))
        p_cumsum = tf.cumsum(p_on_m, axis=-1)
        
        # now before this can be substracted from d, we have to filter the job specific
        # intrinsic time points and set the rest null
        p_on_m_mask = tf.nn.tanh(p_on_m)
        p_cumsum_masked = tf.multiply(p_on_m_mask, p_cumsum)

        # add the penalties to total tardiness
        tardiness = tf.reduce_sum(p_cumsum_masked - d_on_m, axis=-1)
        
        # compute the offset between the due date of each job and the current intrinsic time point
        total_ln.append(tardiness)
    
    total_ln = tf.reshape(tf.convert_to_tensor(total_ln), [3, tf.shape(x)[0]])
    late = tf.transpose(total_ln, [1,0]) + penalties
    return tf.reduce_sum(late)
