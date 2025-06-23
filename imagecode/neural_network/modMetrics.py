from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf

def to_01(x):
    return (x + 1.0) / 2.0  # de [-1, 1] a [0, 1]

@register_keras_serializable()
def psnr(y_true, y_pred):
    return tf.image.psnr(to_01(y_true), to_01(y_pred), max_val=1.0)

@register_keras_serializable()
def ssim(y_true, y_pred):
    return tf.image.ssim(to_01(y_true), to_01(y_pred), max_val=1.0)
