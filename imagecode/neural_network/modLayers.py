from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Multiply
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
def spatialAttention(x):
    attention = Conv2D(1, (1, 1), activation='sigmoid')(x)
    return Multiply()([x, attention])

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
