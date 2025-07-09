import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.regularizers import Regularizer

@register_keras_serializable()
def orthogonal_conv_regularizer(kernel):
    """
    kernel: Tensor (kh, kw, in_ch, out_ch), estándar en Keras Conv2D
    Calcula ‖KᵀK − I‖²_F como regularizador de ortogonalidad.
    """
    # Primero, reacomodamos el kernel: (kh, kw, in_ch, out_ch) → (out_ch, kh, kw, in_ch)
    kernel = tf.transpose(kernel, [3, 0, 1, 2])  # (out_ch, kh, kw, in_ch)

    # Aplanamos cada filtro en un vector
    filters = tf.reshape(kernel, [tf.shape(kernel)[0], -1])  # (out_ch, kh * kw * in_ch)

    # Producto entre filtros: K Kᵀ
    gram = tf.matmul(filters, filters, transpose_b=True)  # (out_ch, out_ch)

    # Diferencia con la identidad
    identity = tf.eye(tf.shape(gram)[0])
    diff = gram - identity

    # Frobenius norm
    loss = tf.reduce_mean(tf.square(diff))

    return loss


@register_keras_serializable()
class OrthogonalConvRegularizer(Regularizer):
    def __init__(self, strength=1e-4):
        self.strength = strength

    def __call__(self, x):
        return self.strength * orthogonal_conv_regularizer(x)

    def get_config(self):
        return {'strength': self.strength}