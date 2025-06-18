import tensorflow as tf
from preprocess import *

@tf.function
def combined_loss(y_true, y_pred):
    # y_true = [L, ab_true]
    L = y_true[..., :1]
    ab_true = y_true[..., 1:]
    ab_pred = y_pred

    mae = tf.reduce_mean(tf.abs(ab_true - ab_pred), axis=[1, 2, 3])

    lab_true = tf.concat([L, ab_true], axis=-1)
    lab_pred = tf.concat([L, ab_pred], axis=-1)

    rgb_true = lab_to_rgb_tensor(lab_true)
    rgb_pred = lab_to_rgb_tensor(lab_pred)

    perceptual = tf.reduce_mean(tf.square(rgb_true - rgb_pred), axis=[1, 2, 3])

    return mae + 0.1 * perceptual
