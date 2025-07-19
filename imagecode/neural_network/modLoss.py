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
        # Configurar ResNet50 para usar float32 explícitamente
        self.mobilenet = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        self.mobilenet.trainable = False
        # Forzar que el modelo use float32
        self.mobilenet = tf.keras.models.clone_model(self.mobilenet)
        self.mobilenet.set_weights(self.mobilenet.get_weights())
        
    def call(self, y_true, y_pred):
        # Asegurar que las entradas sean float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        ab_pred = y_pred
        
        x_L = y_true[..., :1]
        ab_true = y_true[..., 1:]
        print("x_L: ",x_L.dtype)
        print("\n")
        print("ab_true: ",ab_true.dtype)
        print("\n")
        print("y_pred: ",ab_pred.dtype)
        
        # 1. Pérdida principal (igual que antes) - asegurar que todas sean float32
        mae = tf.cast(tf.reduce_mean(tf.abs(ab_true - ab_pred)), tf.float32)
        ssim_loss = tf.cast(1.0 - tf.reduce_mean(tf.image.ssim(self.to_01(ab_true), self.to_01(ab_pred), max_val=1.0)), tf.float32)
        
        y_true_rgb = lab_to_rgb_tensor(x_L, ab_true)
        y_pred_rgb = lab_to_rgb_tensor(x_L, ab_pred)
        
        # Asegurar que las imágenes RGB sean float32
        y_true_rgb = tf.cast(y_true_rgb, tf.float32)
        y_pred_rgb = tf.cast(y_pred_rgb, tf.float32)
        
        true_features = self.mobilenet(y_true_rgb)
        pred_features = self.mobilenet(y_pred_rgb)
        
        # Asegurar que las características sean float32
        true_features = tf.cast(true_features, tf.float32)
        pred_features = tf.cast(pred_features, tf.float32)
        
        perceptual_loss = tf.cast(tf.reduce_mean(tf.square(true_features - pred_features)), tf.float32)
        
        main_loss = 0.6 * mae + 0.3 * ssim_loss + 0.1 * perceptual_loss
        
        return main_loss
    
    def to_01(self, x):
        return tf.clip_by_value((x + 1.0) / 2.0, 0.0, 1.0)  
