import numpy as np
from skimage import util
from algs.quantizationMaxLloyd import quantize_lloyd_max

def quantize_RGB(image,levelsR=16,levelsG=16,levelsB=16):
    R, G, B = [util.img_as_float(channel) for channel in np.dsplit(image, 3)]
    quantized_R_block = quantize_lloyd_max(R,levelsR)
    quantized_G_block = quantize_lloyd_max(G,levelsG)
    quantized_B_block = quantize_lloyd_max(B,levelsB)
    quantized_image_rgb = np.dstack((quantized_R_block, quantized_G_block, quantized_B_block))

    return quantized_image_rgb
