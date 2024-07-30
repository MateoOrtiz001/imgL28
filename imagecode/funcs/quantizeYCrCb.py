import numpy as np
from skimage import color, util
from algs.quantizationMaxLloyd import quantize_lloyd_max

def quantize_YCbCr(image):

  image_yuv = color.rgb2ycbcr(image)
  Y, Cb, Cr = [util.img_as_float(channel) for channel in np.dsplit(image_yuv, 3)]
  quantized_Y = np.zeros_like(Y)
  quantized_Cb = np.zeros_like(Cb)
  quantized_Cr = np.zeros_like(Cr)
  quantized_Y_block = quantize_lloyd_max(Y)
  quantized_Cb_block = quantize_lloyd_max(Cb,16)
  quantized_Cr_block = quantize_lloyd_max(Cr,16)
  quantized_image_yuv = np.dstack((quantized_Y_block, quantized_Cb_block, quantized_Cr_block))
  quantized_image_rgb = color.ycbcr2rgb(quantized_image_yuv)

  return quantized_image_rgb
