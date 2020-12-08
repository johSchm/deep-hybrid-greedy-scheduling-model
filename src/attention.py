import tensorflow as tf


class ScaledDotProductAttention(tf.keras.layers.Layer):
    """ Scaled dot production attention.
    https://github.com/luozhouyang/transformers-keras/blob/master/transformers_keras/layers.py
    """

    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, training=None):
        query, key, value, mask = inputs
        score = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(query)[-1], tf.float32)
        score = score / tf.math.sqrt(dk)
        if mask is not None:
            # note this assumes a broadcastable shape betw score and mask
            # hence score will get auto reshaped and takes the shape of mask
            score += mask * -1e9
        attn_weights = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(attn_weights, value)
        return context, attn_weights

    def get_config(self):
        return super().get_config()
    

class MHSA(tf.keras.layers.Layer):
    """ Multi Head self attention.
    https://github.com/luozhouyang/transformers-keras/blob/master/transformers_keras/layers.py
    """

    def __init__(self,
                 hidden_size=512,
                 num_attention_heads=8,
                 keep_shape_query=False,
                 keep_shape_key=False,
                 keep_shape_value=False,
                 **kwargs):
        super(MHSA, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0
        self.query_weight = None
        self.key_weight = None
        self.value_weight = None
        self.attention = None
        self.dense = None

    def build(self, input_shape):
        w_init = tf.keras.initializers.HeNormal
        b_init = tf.constant_initializer(0.01)
        
        self.query_weight = tf.keras.layers.Dense(self.hidden_size,
                                                  kernel_initializer=w_init,
                                                  bias_initializer=b_init,
                                                  name=self.name+'/Q')
        self.key_weight = tf.keras.layers.Dense(self.hidden_size,
                                                kernel_initializer=w_init,
                                                bias_initializer=b_init,
                                                name=self.name+'/K')
        self.value_weight = tf.keras.layers.Dense(self.hidden_size,
                                                  kernel_initializer=w_init,
                                                  bias_initializer=b_init,
                                                  name=self.name+'/V')

        self.attention = ScaledDotProductAttention()

        self.dense = tf.keras.layers.Dense(self.hidden_size,
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init,
                                           name=self.name+'/dense')

    @tf.function
    def split_heads(self, x, keep_shape=False):
        shape = (tf.shape(x)[0], -1, self.num_attention_heads, x.shape[-1] // self.num_attention_heads)
        x_head = tf.transpose(tf.reshape(x, shape=shape), perm=[0, 2, 1, 3])
        return x_head

    @tf.function
    def call(self, inputs, training=None):
        query, key, value, mask = inputs
        query_shape = tf.shape(query)

        query = self.split_heads(self.query_weight(query))
        key = self.split_heads(self.key_weight(key))
        value = self.split_heads(self.value_weight(value))

        context, attn_weights = self.attention(inputs=(query, key, value, mask))

        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, query_shape)

        output = self.dense(context)
        return output, attn_weights

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))
