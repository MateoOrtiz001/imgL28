from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Multiply

def spatialAttention(x):
    attention = Conv2D(1, (1, 1), activation='sigmoid')(x)
    return Multiply()([x, attention])

def residualBlock(x, filters):
    shortcut = x
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])  # Conexión residual
    return x

