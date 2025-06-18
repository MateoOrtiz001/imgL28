from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf

@register_keras_serializable()
@tf.function
def lab_to_rgb_tensor(lab):
    """
    Función generada por DeepSeek.
    Args:
        lab: tensor (batch, H, W, 3) con valores en [0, 1] (ya normalizado por 255)
    Returns:
        rgb: tensor (batch, H, W, 3)
    """
    L = lab[..., 0] * 100.0
    a = lab[..., 1] * 128.0
    b = lab[..., 2] * 128.0

    # Combinar
    lab = tf.stack([L, a, b], axis=-1)

    # Conversión Lab -> XYZ
    def f_inv(t):
        delta = 6/29
        return tf.where(t > delta, tf.pow(t, 3), 3 * tf.square(delta) * (t - 4/29))

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    x = 0.95047 * f_inv(fx)
    y = 1.00000 * f_inv(fy)
    z = 1.08883 * f_inv(fz)

    # XYZ → RGB
    r = x * 3.2406 + y * (-1.5372) + z * (-0.4986)
    g = x * (-0.9689) + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * (-0.2040) + z * 1.0570

    rgb = tf.stack([r, g, b], axis=-1)

    # Clamp a [0,1]
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    return rgb

@tf.function
def rgb_to_lab_tensor(rgb):
    """
    Función generada por DeepSeek.
    Args:
        rgb: tensor (batch, H, W, 3) con valores en [0, 1] (ya normalizado por 255)
    Returns:
        lab: tensor (batch, H, W, 3) con:
            L en [0,100] (sin normalizar)
            ab en [-128,128] (sin normalizar)
    """
    # Linearización sRGB
    mask = rgb <= 0.04045
    rgb_lin = tf.where(mask, rgb / 12.92, tf.pow((rgb + 0.055)/1.055, 2.4))

    # Matriz RGB → XYZ
    x = rgb_lin[..., 0] * 0.4124564 + rgb_lin[..., 1] * 0.3575761 + rgb_lin[..., 2] * 0.1804375
    y = rgb_lin[..., 0] * 0.2126729 + rgb_lin[..., 1] * 0.7151522 + rgb_lin[..., 2] * 0.0721750
    z = rgb_lin[..., 0] * 0.0193339 + rgb_lin[..., 1] * 0.1191920 + rgb_lin[..., 2] * 0.9503041

    # XYZ → Lab
    epsilon = 6/29
    x_n = x / 0.95047
    z_n = z / 1.08883
    
    def f(t):
        return tf.where(t > epsilon**3, tf.pow(t, 1/3), t/(3*epsilon**2) + 4/29)
    
    fx, fy, fz = f(x_n), f(y), f(z_n)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return tf.stack([L, a, b], axis=-1)