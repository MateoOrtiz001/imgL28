from tensorflow.keras.layers import Add, Multiply, Conv2D, BatchNormalization, Concatenate, Layer, Input, MaxPooling2D
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf

class SpatialAttentionBlock(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv = Conv2D(
            filters=1, 
            kernel_size=7, 
            activation='sigmoid', 
            padding='same',
            name="spatial_attention_conv"
        )
        super(SpatialAttentionBlock, self).build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)

        concat = Concatenate(axis=3)([avg_pool, max_pool])

        attention = self.conv(concat)
        return Multiply()([inputs, attention])

    def get_config(self):
        config = super().get_config()
        return config

@register_keras_serializable()
def residualBlock(x, filters):
    shortcut = x
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])  # Conexión residual
    return x

@register_keras_serializable()
def residualBlockCB(x, filters):
    shortcut = x
    x = Conv2D(filters, (1,1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x,shortcut])
    return x
