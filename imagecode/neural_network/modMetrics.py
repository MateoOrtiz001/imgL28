from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf

@register_keras_serializable()
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)
    
@register_keras_serializable()
def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)