import math
import tensorflow as tf
from metrics import setup_change_penalty


class ExpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, k=0.1):
        super(ExpDecay, self).__init__()
        self.k = tf.cast(k, tf.float32)
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)

    def __call__(self, step):
        return tf.cast(self.initial_learning_rate * tf.math.exp(-self.k*step), tf.float32)

    def get_config(self):
        config = {
            'k': self.k,
            'initial_learning_rate': self.initial_learning_rate
        }
        p = super(ExpDecay, self).get_config()
        return dict(list(p.items()) + list(config.items()))

    
def pick_time(schedule, target_m):
    target_m = tf.cast(target_m, tf.float32)
    T = []
    for b, m in enumerate(tf.unstack(target_m[:, tf.newaxis], axis=0)):
        t = tf.reduce_sum(schedule[b, tf.cast(m[0], tf.int32), :, :], axis=-2)
        T.append(t)
    T = 1.0 - tf.convert_to_tensor(T, tf.float32)
    r = tf.range(9, 0.0, -1.0, dtype=tf.float32)
    T = tf.math.argmax(T * r[tf.newaxis, :], axis=-1)
    return T


def pick_machine(inputs, outputs, y_true=None, i=tf.constant(0), differentiable=False, beta=tf.constant(12.0, tf.float32)):
    """ This method select the most urgent machine (with the least remaining runtime)
    based on the cumsum of the schedule to this point.
    """
    if tf.math.less(i, tf.constant(3, tf.int32)):
        return tf.cast(tf.fill((tf.shape(outputs)[0], 1), i)[:, 0], tf.float32)
    p = tf.matmul(outputs, inputs[:, tf.newaxis, :, 0, tf.newaxis], transpose_a=True)
    p = tf.reduce_sum(tf.squeeze(p, axis=-1), axis=-1)
    p += setup_change_penalty(inputs, outputs)
    p = tf.cast(tf.math.argmin(p, axis=-1), tf.float32)
    if y_true is not None:
        y_true_selections = tf.reduce_sum(y_true, axis=(-2, -1))
        y_pred_new_selections = tf.reduce_sum(outputs, axis=(-2, -1))
        y_pred_new_selections += tf.one_hot(tf.cast(p, tf.int32), 3)
        diff = y_true_selections - y_pred_new_selections
        # less than zero does not work practically since there is little noise in 
        # the predictions, such that its better to use a small negative epsilon
        if tf.math.reduce_any(tf.math.less(diff, -0.01)):
            return tf.cast(tf.math.argmax(diff, axis=-1), tf.float32)
    return p


@tf.function
def update_schedule(schedule, outputs, ms, prior_mask=None, nb_machines=3, nb_jobs=9):
    
    # since each time step is a prob distr, the sum is 1
    # when we subtract this from 1, we get the time steps, which are still free 
    # batch x machines x jobs x time
    T = 1.0 - tf.reduce_sum(schedule, axis=-2)
    
    # now pick the next possible time step by multiplying with the range vector 
    # and draw a arg min
    # batch x machines x 1
    r = tf.range(nb_jobs, 0.0, -1.0, dtype=tf.float32)
    T = tf.math.argmax(T * r[tf.newaxis, tf.newaxis, :], axis=-1)

    # compute a one hot vector from that index
    # batch x machines x time
    T = tf.one_hot(tf.cast(T, tf.int32), nb_jobs)

    # the machine index is already given, round it
    # batch x machines
    M = tf.one_hot(tf.cast(tf.math.round(ms), tf.int32), nb_machines)

    # merge the masks, by multiplying
    # batch x machines x jobs x time 
    mask = T[..., tf.newaxis, :] * M[..., tf.newaxis, tf.newaxis]

    # add prior mask to avoid invalid results, since the input is padded 
    # but this does not prevent the model from predicting the same job again
    # thus we keep the entire mask history in memeory and multiply it on the prediction
    # to destroy the distribution but keep it valid
    outputs *= prior_mask
    
    # add mask the output
    schedule += outputs[:, tf.newaxis, :, tf.newaxis] * mask

    return schedule, mask


@tf.function
def get_machine_state(inputs, outputs):
    outputs = outputs[...,tf.newaxis]
    machine_state = tf.matmul(outputs, inputs, transpose_a=True)
    machine_state = tf.squeeze(machine_state, axis=-2)
    machine_state = tf.reshape(machine_state[:, 2:], [tf.shape(inputs)[0], 5])
    return machine_state


@tf.function
def set_padding_by_output(inputs, outputs, prior_mask=None):
    # inputs: batch x job x props; outpus: batch x job
    if prior_mask is not None:
        outputs *= 1-prior_mask
    mask = tf.one_hot(tf.math.argmax(outputs, axis=-1), tf.shape(outputs)[-1])
    padded = inputs - inputs * mask[..., tf.newaxis]
    return padded, mask[:, tf.newaxis, tf.newaxis, :]


def iterative_optimize(optimizer, model, x, y_true, data_properties, training=True):
    
    schedule = tf.zeros([tf.shape(x)[0], 3, 9, 9])
    machine_state = tf.ones([tf.shape(x)[0], 5]) 
    enc_mask = tf.zeros([tf.shape(x)[0], 1, 1, 9]) 
    
    total_loss = []
    job_queue = x
    acc_m = []

    # normalize the job queue time value for the model
    job_queue_norm = job_queue.numpy()
    x_mean = data_properties.get('x_mean')
    x_stdd = data_properties.get('x_stdd')
    job_queue_norm[:, :, :2] = (job_queue_norm[:, :, :2] - x_mean) / x_stdd
        
    for j in tf.range(9):
        
        # select a machine 
        m = pick_machine(job_queue, schedule, y_true, j)
        
        # select a time point
        t = pick_time(schedule, m)
        
        m = tf.cast(m, tf.int32)
        t = tf.cast(t, tf.int32)
        y = [y_true[b, m[b], :, t[b]] for b in tf.range(tf.shape(y_true)[0])]
        y = tf.convert_to_tensor(y, tf.float32)
        acc_m.append(m)
        
        with tf.GradientTape() as tape:
            output, [jq_emb, ms_emb], attn = model(
                [job_queue_norm, machine_state, enc_mask],
                training=training)
            loss_value = tf.keras.losses.categorical_crossentropy(y, output)
            
        grads = tape.gradient(loss_value, model.trainable_variables)
    
        total_loss.append(loss_value)
        
        if training:
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
                
        # update the job queue
        job_queue, mask = set_padding_by_output(job_queue, output, enc_mask[:, 0, 0, :])
        
        # get the next machine state 
        machine_state = get_machine_state(job_queue, output)
        
        # add the job distribution to the schedule
        schedule, schedule_mask = update_schedule(
            schedule, output, m, 1-enc_mask[:, 0, 0, :])
        schedule = tf.reshape(schedule, [tf.shape(x)[0], 3, 9, 9])
        
        enc_mask += mask
        
    m = tf.transpose(tf.convert_to_tensor(acc_m, tf.float32), [1, 0])
    total_loss = tf.reduce_mean(tf.convert_to_tensor(total_loss), axis=-1)
    
    return total_loss, grads, schedule, m, [jq_emb, ms_emb], attn
