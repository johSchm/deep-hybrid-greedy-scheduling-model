import tensorflow as tf 
from attention import MHSA


class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=512,
                 ffn_size=2048,
                 **kwargs):
        super(PointWiseFeedForwardNetwork, self).__init__(**kwargs)
        self.ffn_size = ffn_size
        self.hidden_size = hidden_size
        self.dense1 = None
        self.dense2 = None

    def build(self, input_shape):
        w_init = tf.keras.initializers.HeNormal
        b_init = tf.constant_initializer(0.01)
        
        self.dense1 = tf.keras.layers.Dense(self.ffn_size,
                                            activation='relu',
                                            kernel_initializer=w_init,
                                            bias_initializer=b_init,
                                            name=self.name+'/dense0')
        
        self.dense2 = tf.keras.layers.Dense(self.hidden_size,
                                            kernel_initializer=w_init,
                                            bias_initializer=b_init,
                                            name=self.name+'/dense1')

    @tf.function
    def call(self, inputs, training=None):
        outputs = self.dense2(self.dense1(inputs))
        return outputs

    def get_config(self):
        config = {
            'ffn_size': self.ffn_size,
            'hidden_size': self.hidden_size
        }
        p = super(PointWiseFeedForwardNetwork, self).get_config()
        return dict(list(p.items()) + list(config.items()))
    

class JobEmbedding(tf.keras.layers.Layer):
    
    def __init__(self,
                 embedding_size=512, 
                 dropout_rate=0.2, 
                 epsilon=1e-6, 
                 **kwargs):
        super(JobEmbedding, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.dense1 = None
        self.dense2 = None
        self.dense3 = None
        self.layer_norm = None

    def build(self, input_shape):
        batch, seq_len, properties = input_shape
        w_init = tf.keras.initializers.HeNormal
        b_init = tf.constant_initializer(0.01)
        
        self.dense1 = tf.keras.layers.Dense(self.embedding_size / 2,
                                            kernel_initializer=w_init,
                                            bias_initializer=b_init,
                                            name=self.name+'/job_emb_dense1')
        
        self.dense2 = tf.keras.layers.Dense(self.embedding_size / 2,
                                            kernel_initializer=w_init,
                                            bias_initializer=b_init,
                                            name=self.name+'/job_emb_dense2')
        
        self.dense3 = tf.keras.layers.Dense(self.embedding_size,
                                            activation=tf.keras.activations.relu,
                                            kernel_initializer=w_init,
                                            bias_initializer=b_init,
                                            name=self.name+'/job_emb_dense3')
        
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon,
                                                             name=self.name+'/job_emb_ln')

    @tf.function
    def call(self, inputs, training=None):
        
        # embed the processing times and due dates 
        pt_dd = tf.transpose(tf.unstack(inputs, axis=-1)[:2], [1, 2, 0])
        pt_dd_emb = self.dense1(pt_dd)
        
        # embed the setups 
        r = tf.transpose(tf.unstack(inputs, axis=-1)[2:], [1, 2, 0])
        r_emb = self.dense2(r)

        # fuse embedding
        embedding = tf.concat([pt_dd_emb, r_emb], axis=-1)
        embedding = self.dense3(embedding)
        embedding = self.dropout(embedding, training=training)
        embedding = self.layer_norm(embedding)
        
        return embedding

    def get_config(self):
        conf = {
            'embedding_size': self.embedding_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon
        }
        p = super(JobEmbedding, self).get_config()
        return dict(list(p.items()) + list(conf.items()))
    

class MaschineStateEmbedding(tf.keras.layers.Layer):
    
    def __init__(self,
                 embedding_size=512, 
                 ffn_size=2048,
                 dropout_rate=0.2, 
                 epsilon=1e-6, 
                 **kwargs):
        super(MaschineStateEmbedding, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.dropout = None
        self.ffn = None
        self.embedding = None

    def build(self, input_shape):
        w_init = tf.keras.initializers.HeNormal
        b_init = tf.constant_initializer(0.01)
        
        self.ffn = PointWiseFeedForwardNetwork(self.embedding_size,
                                               self.ffn_size,
                                               name=self.name+'/ms_emb_dense')
        
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon,
                                                             name=self.name+'/ms_emb_ln')

    @tf.function
    def call(self, inputs, training=None):
        embedding = self.ffn(inputs)
        embedding = self.dropout(embedding, training=training)
        embedding = self.layer_norm(embedding)
        return embedding

    def get_config(self):
        conf = {
            'embedding_size': self.embedding_size,
            'ffn_size': self.ffn_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon
        }
        p = super(MaschineStateEmbedding, self).get_config()
        return dict(list(p.items()) + list(conf.items()))

    
class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=512,
                 num_attention_heads=8,
                 ffn_size=2048,
                 dropout_rate=0.2,
                 epsilon=1e-6,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.ffn_size = ffn_size
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.attention = None
        self.attn_dropout = None
        self.attn_layer_norm = None
        self.ffn = None
        self.ffn_dropout = None
        self.ffn_layer_norm = None

    def build(self, input_shape):
        self.attention = MHSA(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads)
        
        self.attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon,
                                                                  name='encoder_layer_attn_ln1')

        self.ffn = PointWiseFeedForwardNetwork(self.hidden_size,
                                               self.ffn_size,
                                               name=self.name+'/encoder_layer_ff')

        self.ffn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
        self.ffn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon,
                                                                 name=self.name+'/encoder_layer_ff_ln')

    @tf.function
    def call(self, inputs, training=None):
        query, key, value, mask = inputs
        attn, attn_weights = self.attention(inputs=(query, key, value, mask))
        attn = self.attn_dropout(attn, training=training)
        attn = self.attn_layer_norm(query + attn)

        ffn = self.ffn(attn)
        ffn = self.ffn_dropout(ffn, training=training)
        ffn = self.ffn_layer_norm(ffn + attn)

        return ffn, attn_weights

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'ffn_size': self.ffn_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon
        }
        base = super(EncoderLayer, self).get_config()
        return dict(list(base.items()) + list(config.items()))
    

class SchedulingModel(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=256,
                 num_attention_heads=4,
                 ffn_size=2048,
                 dropout_rate=0.1,
                 epsilon=1e-6,
                 num_machines=3,
                 **kwargs):
        super(SchedulingModel, self).__init__(**kwargs)
        self.num_machines = num_machines 
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.ffn_size = ffn_size
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.dropout_fusion_1 = None
        self.dropout_fusion_2 = None
        self.dropout_fusion_3 = None
        self.dense_fusion = None
        self.dense_logits = None
        self.ln_fusion_1 = None
        self.ln_fusion_2 = None
        self.ln_fusion_3 = None
        self.ln_attn = None
        self.job_emb = None
        self.num_jobs = None 

    def build(self, input_shape):
        self.num_jobs = input_shape[0][1]
        w_init = tf.keras.initializers.HeNormal
        b_init = tf.constant_initializer(0.01)
        
        self.job_emb = JobEmbedding(embedding_size=self.hidden_size,
                                    dropout_rate=self.dropout_rate,
                                    epsilon=self.epsilon)
        
        self.machine_emb = MaschineStateEmbedding(ffn_size=self.ffn_size,
                                                  embedding_size=self.hidden_size,
                                                  dropout_rate=self.dropout_rate,
                                                  epsilon=self.epsilon)
        
        self.encoder = EncoderLayer(num_attention_heads=self.num_attention_heads,
                                    hidden_size=self.hidden_size,
                                    dropout_rate=self.dropout_rate,
                                    epsilon=self.epsilon)

        self.dropout_fusion_1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout_fusion_2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout_fusion_3 = tf.keras.layers.Dropout(self.dropout_rate)
        
        self.ln_fusion_1 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon,
                                                              name=self.name+'/ln_fusion_1')
        self.ln_fusion_2 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon,
                                                              name=self.name+'/ln_fusion_2')
        self.ln_fusion_3 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon,
                                                              name=self.name+'/ln_fusion_3')
        
        self.dense_fusion = tf.keras.layers.Dense(self.num_jobs * self.hidden_size,
                                                  activation='relu',
                                                  kernel_initializer=w_init,
                                                  bias_initializer=b_init,
                                                  name=self.name+'/dense_fusion')

        self.dense_logits = tf.keras.layers.Dense(1,
                                                  kernel_initializer=w_init,
                                                  bias_initializer=b_init,
                                                  name=self.name+'/dense_logits')

        self.ffn1 = PointWiseFeedForwardNetwork(self.num_jobs * self.hidden_size,
                                                self.ffn_size,
                                                name=self.name+'/fusion_ff1')

        self.ffn2 = PointWiseFeedForwardNetwork(self.num_jobs * self.hidden_size,
                                                self.ffn_size,
                                                name=self.name+'/fusion_ff2')
                
        # U E_J^T: HxH HxJ = HxJ
        self.U = tf.keras.layers.Dense(self.hidden_size,
                                       kernel_initializer=w_init,
                                       bias_initializer=b_init,
                                       name=self.name+'/U')
        
        # V E_M^T: HxH Hx1 = Hx1
        self.V = tf.keras.layers.Dense(self.hidden_size,
            kernel_initializer=w_init,
            bias_initializer=b_init,
            name=self.name+'/V')
        
        self.ln_attn = tf.keras.layers.LayerNormalization(epsilon=self.epsilon,
                                                          name=self.name+'/ln_attn')

    @tf.function
    def call(self, inputs, training=None):
        job_queue, machine_state, enc_padding_mask = inputs
        
        # embedding
        job_queue_emb = self.job_emb(job_queue)
        machine_state_emb = self.machine_emb(machine_state)
        
        # job encoder
        enc_outputs, enc_attns = self.encoder(
            inputs=(job_queue_emb, job_queue_emb, job_queue_emb, enc_padding_mask))

        # tile the machine embedding to the number of jobs
        # to use a residual connection with the fusion layer
        enc_outputs_res = enc_outputs + tf.tile(machine_state_emb[:, tf.newaxis, :],
                                                [1, self.num_jobs, 1])
        
        # fuse job and machine embeddings with the attention mechanism presented in the paper
        U = tf.transpose(self.U(enc_outputs), [0, 2, 1])
        V = tf.transpose(self.V(machine_state_emb)[:, tf.newaxis, :], [0, 2, 1])
        W = (U * V)/tf.math.sqrt(tf.cast(self.hidden_size, tf.float32))
        W = tf.transpose(W, [0, 2, 1])
        
        enc_outputs = self.ln_attn(W + enc_outputs_res)
        enc_outputs = tf.reshape(enc_outputs, [
            tf.shape(job_queue)[0], self.num_jobs * self.hidden_size])

        # batch x jobs x hidden
        output_fusion_1 = self.ffn1(enc_outputs)
        output_fusion_1 = self.dropout_fusion_1(output_fusion_1)
        output_fusion_1 = self.ln_fusion_1(output_fusion_1 + enc_outputs)
        
        # batch x jobs x hidden
        output_fusion_2 = self.ffn2(output_fusion_1)
        output_fusion_2 = self.dropout_fusion_2(output_fusion_2)
        output_fusion_2 = self.ln_fusion_2(output_fusion_2 + output_fusion_1)
        
        # batch x jobs x hidden
        output_fusion_3 = self.dense_fusion(output_fusion_2)
        output_fusion_3 = self.dropout_fusion_3(output_fusion_3)
        output_fusion_3 = self.ln_fusion_3(output_fusion_3 + output_fusion_2)

        # batch x jobs x 1
        output = tf.reshape(output_fusion_3,
                            [tf.shape(job_queue)[0], self.num_jobs, self.hidden_size])
        output = self.dense_logits(output)
        output = tf.squeeze(output, axis=-1)
        output = tf.nn.softmax(output)
        
        return output, [job_queue_emb, machine_state_emb], enc_attns

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'ffn_size': self.ffn_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'num_machines': self.num_machines
        }
        base = super(SchedulingModel, self).get_config()
        return dict(list(base.items()) + list(config.items()))
