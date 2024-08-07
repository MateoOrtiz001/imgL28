import numpy as np
from skimage import color, util
from algs.quantizationMaxLloyd import quantize_lloyd_max

def quantize_YCbCr(image,levels_Y=32,levels_Cb=16,levels_Cr=16):

  image_yuv = color.rgb2ycbcr(image)
  Y, Cb, Cr = [util.img_as_float(channel) for channel in np.dsplit(image_yuv, 3)]
  quantized_Y_block = quantize_lloyd_max(Y,levels_Y)
  quantized_Cb_block = quantize_lloyd_max(Cb,levels_Cb)
  quantized_Cr_block = quantize_lloyd_max(Cr,levels_Cr)
  quantized_image_yuv = np.dstack((quantized_Y_block, quantized_Cb_block, quantized_Cr_block))
  quantized_image_rgb = color.ycbcr2rgb(quantized_image_yuv)

  return quantized_image_rgb
