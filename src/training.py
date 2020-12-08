import tensorflow as tf
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.metrics import FalseNegatives
from tensorflow.keras.metrics import FalsePositives
from tensorflow.keras.metrics import TrueNegatives
from tensorflow.keras.metrics import TruePositives
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
from datetime import datetime
from pathlib import Path
from utils import iterative_optimize
from metrics import lateness, makespan, count_setup_changes
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
import os 


CKPT_SAVE_INV_FREQ = 10
TB_LOG_INV_FREQ = 500


class TrainingController:
    """ Custom training loop with custom loss.
    """
    
    def __init__(self,
                 model,
                 optimizer,
                 log_file_dir=None,
                 data_properties=None):
        """ Init. method.
        :param log_file_dir: If this is not None, then the training performance is stored in that log file directory.
        :param model: The model used for training.
        :param optimizer: Optimizer to be used for the weight updates.
        """
        self._data_properties = data_properties
        self._log_file_dir = log_file_dir
        self._optimizer = optimizer
        self._model = model
        
        self._tp_obj = TruePositives()
        self._tn_obj = TrueNegatives()
        self._fp_obj = FalsePositives()
        self._fn_obj = FalseNegatives()
        self._pre_obj = Precision()
        self._rec_obj = Recall()
        self._setup_changes = {
            'train': [],
            'valid': []
        }
        self._loss_tt = {
            'train': [],
            'valid': []
        }
        self._loss_ms = {
            'train': [],
            'valid': []
        }
        self._loss_total = {
            'train': [],
            'valid': []
        }
        self._acc = {
            'train': [],
            'valid': []
        }
        self._tn = {
            'train': [],
            'valid': []
        }
        self._tp = {
            'train': [],
            'valid': []
        }
        self._fn = {
            'train': [],
            'valid': []
        }
        self._fp = {
            'train': [],
            'valid': []
        }
        self._rec = {
            'train': [],
            'valid': []
        }
        self._pre = {
            'train': [],
            'valid': []
        }

    
    def _tb_update(self, grads, y_true, y_pred, m_idx, emb, attn, epoch, batch, batch_size, prefix='train/'):
        step = epoch * batch_size + batch

        if grads is not None:
            for var, grad in zip(self._model.trainable_variables, grads):
                tf.summary.histogram(prefix + var.name + '/gradient', grad, step=step)
        
        if attn is not None:
            self._plot_attention_weights(attention=attn,
                                         step=step,
                                         prefix=prefix+'layer{}/enc_pad/'.format(0),
                                         description='x: input jobs | y: output jobs')

        m_idx = tf.tile(m_idx[:, :, tf.newaxis, tf.newaxis], [1, 1, tf.shape(m_idx)[1], 1])
        tf.summary.image(prefix + "selected_machine", m_idx, step=step)

        for var in self._model.trainable_variables:
            tf.summary.histogram(prefix + var.name + '/weight', var, step=step)

        for m in tf.range(tf.shape(y_true)[1]):
            tf.summary.image(prefix + "y_true_m{}".format(m), tf.expand_dims(y_true[:, m, :, :], -1), step=step)
            tf.summary.image(prefix + "y_pred_m{}".format(m), tf.expand_dims(y_pred[:, m, :, :], -1), step=step)

    @staticmethod
    def _plot_attention_weights(attention, step, description='x: input, y: output', prefix='train/'):
        for head in range(attention.shape[1]):
            data = []
            for attn_matrix in tf.unstack(attention, axis=0):
                attn_matrix = attn_matrix.numpy()
                cmap = cm.get_cmap('Greens')
                norm = Normalize(vmin=attn_matrix.min(), vmax=attn_matrix.max())
                data.append(cmap(norm(attn_matrix)))
            tf.summary.image(prefix + "head{}".format(head),
                             np.array(data, np.float32)[:, head, :, :, :],
                             step=step, description=description)
        
    def train(self,
              train_data,
              val_data=None,
              epochs=1,
              steps_per_epoch=100,
              checkpoint_path=None,
              validation_steps=10):
        """ Custom training loop with custom loss.
        :param train_data: training data set
        :param val_data: validation data set
        :param epochs: number of training epochs
        :param steps_per_epoch: steps per epochs (required if generator used).
        If set to None, the number will be computed automatically.
        :param checkpoint_path: save checkpoints epoch-wise if directory provided.
        :param validation_steps:
        :return accumulated loss and accuracy
        """
        log_path = self._log_file_dir + '/' + datetime.now().strftime("%y%m%d-%H%M%S")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        writer = tf.summary.create_file_writer(log_path)
        writer.set_as_default()
        for step in tf.range(steps_per_epoch * epochs, dtype=tf.int64):
            tf.summary.scalar('learning_rate', self._optimizer.lr(tf.cast(step, tf.float32)), step=step)
        for epoch in range(epochs):
            for batch, (x, y_true) in enumerate(train_data):
                if batch == 0:
                    self._target_shape = x.shape
                if batch >= steps_per_epoch:
                    break
                    
                loss_total, grads, y_pred, m, emb, attn = iterative_optimize(optimizer=self._optimizer,
                                                                             model=self._model,
                                                                             x=x,
                                                                             y_true=y_true,
                                                                             data_properties=self._data_properties,
                                                                             training=True)  

                loss_tt = lateness(x, y_pred)
                loss_ms = makespan(x, y_pred)
                setup_changes = count_setup_changes(x, y_pred)
                self._update_metric('train', y_true, y_pred, (loss_tt, loss_ms, loss_total), setup_changes, batch)
                self._print('train', epoch, epochs, batch, steps_per_epoch)
                    
                if batch % TB_LOG_INV_FREQ == 0:
                    self._tb_update(grads, y_true, y_pred, m, emb, attn, epoch, batch, steps_per_epoch, 'train/')
                    self._log('train', epoch * steps_per_epoch + batch)

            self._validation_loop(val_data, validation_steps, epoch)

            self._empty_metric()

            if checkpoint_path and (epoch % CKPT_SAVE_INV_FREQ == 0):
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                self._model.save_weights(checkpoint_path.format(epoch=epoch))

        writer.close()

    def _empty_metric(self):
        """ This will empty all metric dict, to avoid memory overflows.
        """
        for key in ['train', 'valid']:
            
            self._loss_tt.get(key).clear()
            self._loss_ms.get(key).clear()
            self._loss_total.get(key).clear()

            self._acc.get(key).clear()

            self._setup_changes.get(key).clear()
            
            self._tp_obj.reset_states()
            self._tp.get(key).clear()

            self._tn_obj.reset_states()
            self._tn.get(key).clear()

            self._fp_obj.reset_states()
            self._fp.get(key).clear()

            self._fn_obj.reset_states()
            self._fn.get(key).clear()

            self._pre_obj.reset_states()
            self._pre.get(key).clear()

            self._rec_obj.reset_states()
            self._rec.get(key).clear()

    def _print(self, key: str, epoch: int, epochs_max: int, batch: int, batch_max: int):
        """ Prints the performance results in the console.
        :param key:
        :param epoch:
        :param epochs_max:
        :param batch:
        :param batch_max:
        """
        mean_loss = tf.reduce_mean(self._loss_total.get(key))
        mean_acc = tf.reduce_mean(self._acc.get(key))
        mean_pre = tf.reduce_mean(self._pre.get(key))
        mean_rec = tf.reduce_mean(self._rec.get(key))

        if key == 'train':
            tf.print('\r[Train] [E {0}/{1}] [B {2}/{3}] Loss: {4} Acc: {5} Pre: {6} Rec: {7}'
                     .format(epoch + 1, epochs_max, batch + 1, batch_max,
                             mean_loss, mean_acc, mean_pre, mean_rec), end='')
        else:
            tf.print('\n[Valid] [E {0}/{1}] [B {2}/{3}] Loss: {4} Acc: {5} Pre: {6} Rec: {7}\n'
                     .format(epoch, epochs_max, batch + 1, batch_max,
                             mean_loss, mean_acc, mean_pre, mean_rec))

    def _validation_loop(self, validation_data, validation_steps: int, epoch: int):
        """ Looping through the validation set and ouputs the validation performance
        results in a final step.
        :param validation_data:
        :param validation_steps:
        """
        for batch, (x, y_true) in enumerate(validation_data):
            if batch >= validation_steps:
                break
                optimizer=optimizer,
                                                                            
            loss_total, grads, y_pred, m, emb, attn = iterative_optimize(optimizer=self._optimizer,
                                                                         model=self._model,
                                                                         x=x,
                                                                         y_true=y_true,
                                                                         data_properties=self._data_properties,
                                                                         training=False)  
            loss_tt = lateness(x, y_pred)
            loss_ms = makespan(x, y_pred)
            setup_changes = count_setup_changes(x, y_pred)
            self._update_metric('valid', y_true, y_pred, (loss_tt, loss_ms, loss_total), setup_changes, batch)
            
            if batch % (TB_LOG_INV_FREQ * 0.1) == 0:
                self._tb_update(None, y_true, y_pred, m, emb, attn, epoch, batch, validation_steps, 'valid/')
                self._log('valid', epoch * validation_steps + batch)

        self._print('valid', 0, 0, validation_steps - 1, validation_steps)

    def _update_metric(self,
                       key: str,
                       y_true: tf.Tensor,
                       y_pred: tf.Tensor,
                       loss: tuple,
                       setup_changes: tf.Tensor,
                       step=0):
        """ Updates the metrics.
        :param key:
        :param y_true:
        :param y_pred:
        :param loss:
        :param grads:
        """
        loss_tt, loss_ms, loss_total = loss

        self._loss_tt.get(key).append(loss_tt)
        self._loss_ms.get(key).append(loss_ms)
        self._loss_total.get(key).append(loss_total)

        self._setup_changes.get(key).append(setup_changes)
        
        self._tp_obj.update_state(y_true, y_pred)
        self._tp.get(key).append(self._tp_obj.result())

        self._tn_obj.update_state(y_true, y_pred)
        self._tn.get(key).append(self._tn_obj.result())

        self._fp_obj.update_state(y_true, y_pred)
        self._fp.get(key).append(self._fp_obj.result())

        self._fn_obj.update_state(y_true, y_pred)
        self._fn.get(key).append(self._fn_obj.result())

        self._pre_obj.update_state(y_true, y_pred)
        self._pre.get(key).append(self._pre_obj.result())

        self._rec_obj.update_state(y_true, y_pred)
        self._rec.get(key).append(self._rec_obj.result())

        shape = tf.shape(y_true)
        y_true = tf.squeeze(tf.transpose(y_true, [0, 2, 1, 3]))
        y_pred = tf.squeeze(tf.transpose(y_pred, [0, 2, 1, 3]))
        y_pred = tf.reshape(y_pred, [shape[0], shape[2], -1])
        y_true = tf.reshape(y_true, [shape[0], shape[2], -1])

        self._acc.get(key).append(categorical_accuracy(y_true, y_pred))

    def _log(self, key: str, epoch: int):
        """ Logs the training progress in a log file. If the log file dir parameter is set.
        :param key:
        :param epoch:
        """
        if not self._log_file_dir:
            return
        if not os.path.exists(self._log_file_dir):
            os.mkdir(self._log_file_dir)

        tf.summary.scalar(key+'/tardiness', data=tf.reduce_mean(self._loss_ms.get(key)), step=epoch)
        tf.summary.scalar(key+'/makespan', data=tf.reduce_mean(self._loss_tt.get(key)), step=epoch)
        tf.summary.scalar(key+'/loss', data=tf.reduce_mean(self._loss_total.get(key)), step=epoch)
        tf.summary.scalar(key+'/acc', data=tf.reduce_mean(self._acc.get(key)), step=epoch)
        tf.summary.scalar(key+'/setup_changes', data=tf.reduce_mean(self._setup_changes.get(key)), step=epoch)
        tf.summary.scalar(key+'/tp', data=tf.reduce_mean(self._tp.get(key)), step=epoch)
        tf.summary.scalar(key+'/fp', data=tf.reduce_mean(self._fp.get(key)), step=epoch)
        tf.summary.scalar(key+'/tn', data=tf.reduce_mean(self._tn.get(key)), step=epoch)
        tf.summary.scalar(key+'/fn', data=tf.reduce_mean(self._fn.get(key)), step=epoch)
        tf.summary.scalar(key+'/pre', data=tf.reduce_mean(self._pre.get(key)), step=epoch)
        tf.summary.scalar(key+'/rec', data=tf.reduce_mean(self._rec.get(key)), step=epoch)
