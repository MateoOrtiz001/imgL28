from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
from preprocess import *

def to_01(x):
    return (x + 1.0) / 2.0  # de [-1, 1] a [0, 1]

@register_keras_serializable()
def psnr(y_true, y_pred):
    l_channel = y_true[..., :1]
    y_true_rgb = lab_to_rgb_tensor(l_channel, y_true[..., 1:])
    y_pred_rgb = lab_to_rgb_tensor(l_channel, y_pred)
    return tf.image.psnr(y_true_rgb, y_pred_rgb, max_val=1.0)

@register_keras_serializable()
def ssim(y_true, y_pred):
    l_channel = y_true[..., :1]
    y_true_rgb = lab_to_rgb_tensor(l_channel, y_true[..., 1:])
    y_pred_rgb = lab_to_rgb_tensor(l_channel, y_pred)
    return tf.image.ssim(y_true_rgb, y_pred_rgb, max_val=1.0)