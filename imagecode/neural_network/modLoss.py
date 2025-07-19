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
        # Obtener el tipo de dato de y_pred (será float16 o float32 según la política)
        compute_dtype = y_pred.dtype
        
        # Convertir todo al mismo tipo de dato que están usando las predicciones
        y_true = tf.cast(y_true, compute_dtype)
        y_pred = tf.cast(y_pred, compute_dtype)
       
        ab_pred = y_pred
       
        x_L = y_true[..., :1]
        ab_true = y_true[..., 1:]
       
        # 1. Pérdida MAE
        mae = tf.reduce_mean(tf.abs(ab_true - ab_pred))
        
        # 2. Pérdida SSIM - convertir a float32 solo para SSIM ya que requiere float32
        ab_true_f32 = tf.cast(ab_true, tf.float32)
        ab_pred_f32 = tf.cast(ab_pred, tf.float32)
        ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(
            self.to_01(ab_true_f32), 
            self.to_01(ab_pred_f32), 
            max_val=1.0
        ))
        # Convertir resultado de vuelta al tipo de cómputo
        ssim_loss = tf.cast(ssim_loss, compute_dtype)
       
        # 3. Pérdida perceptual
        y_true_rgb = lab_to_rgb_tensor(x_L, ab_true)
        y_pred_rgb = lab_to_rgb_tensor(x_L, ab_pred)
        
        # Para ResNet50, convertir a float32 si es necesario
        if compute_dtype != tf.float32:
            y_true_rgb_f32 = tf.cast(y_true_rgb, tf.float32)
            y_pred_rgb_f32 = tf.cast(y_pred_rgb, tf.float32)
        else:
            y_true_rgb_f32 = y_true_rgb
            y_pred_rgb_f32 = y_pred_rgb
            
        true_features = self.mobilenet(y_true_rgb_f32)
        pred_features = self.mobilenet(y_pred_rgb_f32)
        perceptual_loss = tf.reduce_mean(tf.square(true_features - pred_features))
        # Convertir resultado de vuelta al tipo de cómputo
        perceptual_loss = tf.cast(perceptual_loss, compute_dtype)
       
        # Combinar pérdidas usando el mismo tipo de dato
        weights = tf.cast([0.6, 0.3, 0.1], compute_dtype)
        main_loss = weights[0] * mae + weights[1] * ssim_loss + weights[2] * perceptual_loss
       
        # La pérdida final debe retornarse en float32 para el optimizador
        return tf.cast(main_loss, tf.float32)
   
    def to_01(self, x):
        # Esta función ya espera float32
        return tf.clip_by_value((x + 1.0) / 2.0, 0.0, 1.0)
