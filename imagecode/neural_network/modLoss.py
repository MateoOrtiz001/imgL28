import tensorflow as tf
#from tensorflow_models.vision.augment import gaussian_filter2d
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import Loss
from preprocess import *

@register_keras_serializable()
class CustomCombinedLoss(Loss):
    def __init__(self, guidance_weight=0.2):  # Peso ajustable
        super().__init__()
        self.guidance_weight = guidance_weight
        self.mobilenet = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        self.mobilenet.trainable = False
        
    def call(self, y_true, y_pred):
        ab_pred = y_pred
        
        x_L = y_true[..., :1]
        ab_true = y_true[..., 1:]
        
        # 1. Pérdida principal (igual que antes)
        mae = tf.reduce_mean(tf.abs(ab_true - ab_pred))
        ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(self.to_01(ab_true), self.to_01(ab_pred), max_val=1.0))
        
        y_true_rgb = lab_to_rgb_tensor(x_L, ab_true)
        y_pred_rgb = lab_to_rgb_tensor(x_L, ab_pred)
        true_features = self.mobilenet(y_true_rgb)
        pred_features = self.mobilenet(y_pred_rgb)
        perceptual_loss = tf.cast(tf.reduce_mean(tf.square(true_features - pred_features)), tf.float32)
        
        main_loss = 0.6 * mae + 0.3 * ssim_loss + 0.1 * perceptual_loss
        
        return main_loss
    
    def to_01(self, x):
        return tf.clip_by_value((x + 1.0) / 2.0, 0.0, 1.0)  
