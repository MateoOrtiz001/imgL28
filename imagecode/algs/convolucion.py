import numpy as np

def matrizConvolucion(matriz,kernel,normalizar = False):
#  kernel = np.array([[a,b,c],[d,e,f],[g,h,l]], dtype=np.float32)
  if normalizar:
    sumKernel = np.abs(kernel.sum())
    if sumKernel != 0:
      kernel = kernel / sumKernel

  aux = matriz * kernel
  index = aux.sum()
  return index