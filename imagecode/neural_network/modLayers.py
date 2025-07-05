from tensorflow.keras.layers import Add, Multiply, Conv2D, BatchNormalization, Concatenate, Layer, Input, MaxPooling2D
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf

@register_keras_serializable()
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
class FullyConvGlobalAttention(Layer):
    """
    Módulo de atención global completamente convolucional
    que mantiene la capacidad de procesar imágenes de cualquier tamaño
    """
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.reduced_channels = max(self.channels // self.reduction_ratio, 8)
        
        # Todas las operaciones son convolucionales
        # Atención de canal (squeeze-and-excitation style)
        self.channel_squeeze = Conv2D(self.reduced_channels, (1, 1), activation='relu')
        self.channel_excite = Conv2D(self.channels, (1, 1), activation='sigmoid')
        
        # Atención espacial con contexto multi-escala
        self.spatial_conv1 = Conv2D(1, (1, 1), activation='sigmoid')
        self.spatial_conv3 = Conv2D(1, (3, 3), padding='same', activation='sigmoid')
        self.spatial_conv5 = Conv2D(1, (5, 5), padding='same', activation='sigmoid')
        
        # Combinación de atenciones espaciales
        self.spatial_combine = Conv2D(1, (1, 1), activation='sigmoid')
        
        super().build(input_shape)
    
    def call(self, inputs):
        # 1. Atención de canal (contexto global por canal)
        # Usar convolución global en lugar de GAP + Dense
        channel_context = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)  # (B, 1, 1, C)
        channel_att = self.channel_squeeze(channel_context)
        channel_att = self.channel_excite(channel_att)
        
        # Aplicar atención de canal
        channel_attended = inputs * channel_att
        
        # 2. Atención espacial multi-escala
        spatial_att1 = self.spatial_conv1(channel_attended)
        spatial_att3 = self.spatial_conv3(channel_attended)
        spatial_att5 = self.spatial_conv5(channel_attended)
        
        # Combinar atenciones espaciales
        spatial_combined = Concatenate()([spatial_att1, spatial_att3, spatial_att5])
        spatial_att = self.spatial_combine(spatial_combined)
        
        # 3. Aplicar atención espacial
        output = channel_attended * spatial_att
        
        return output

@register_keras_serializable()
class LightweightSemanticGuidance(Layer):
    """
    Guía semántica ligera usando solo convoluciones
    """
    def __init__(self, num_semantic_channels=32, **kwargs):
        super().__init__(**kwargs)
        self.num_semantic_channels = num_semantic_channels
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        
        # Extractor de características semánticas
        self.semantic_conv1 = Conv2D(self.num_semantic_channels, (3, 3), padding='same', activation='relu')
        self.semantic_conv2 = Conv2D(self.num_semantic_channels, (3, 3), padding='same', activation='relu')
        
        # Generador de mapas de color guidance
        self.color_guidance = Conv2D(2, (1, 1), activation='tanh')  # 2 canales para a*, b*
        
        # Combinador de features originales con guidance
        self.feature_fusion = Conv2D(self.channels, (1, 1), activation='relu')
        
        super().build(input_shape)
    
    def call(self, inputs):
        # 1. Extraer características semánticas
        semantic_features = self.semantic_conv1(inputs)
        semantic_features = self.semantic_conv2(semantic_features)
        
        # 2. Generar guidance de color
        color_guidance = self.color_guidance(semantic_features)
        
        # 3. Combinar features originales con guidance
        # Expandir guidance a las dimensiones de entrada
        guidance_expanded = tf.tile(color_guidance, [1, 1, 1, self.channels // 2])
        
        # Combinar con features originales
        combined = Concatenate()([inputs, guidance_expanded])
        output = self.feature_fusion(combined)
        
        return output, color_guidance

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
