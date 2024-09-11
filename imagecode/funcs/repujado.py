import numpy as np
from algs.convolucion import matrizConvolucion

def repujado(image,levelR=2,levelG=2,levelB=2,canalR = True,canalG = True, canalB = True):
    imagen_np = np.array(image).astype(np.float32)
    R = imagen_np[:, :, 0]
    G = imagen_np[:, :, 1]
    B = imagen_np[:, :, 2]
    if canalR:
      convR = convolucionar_canal(R,levelR)
    if canalG:
      convG = convolucionar_canal(G,levelG)
    if canalB:
      convB = convolucionar_canal(B,levelB)
    output_image = np.dstack((convR, convG, convB))
    return output_image

def convolucionar_canal(aux,nivel):
    n = aux.shape[0]
    m = aux.shape[1]
#    n, m = aux.shape
    salida = np.zeros_like(aux)
    kernel = np.array([[-1*nivel,-1,0],[-1,1,1],[0,1,nivel]])
    for i in range(n):
        for j in range(m):
            if i == 0:
                if j == 0:
                    matrix = np.array([[aux[i][j], aux[i][j], aux[i+1][j]], [aux[i][j], aux[i][j], aux[i+1][j]], [aux[i][j+1], aux[i][j+1], aux[i+1][j+1]]])
                elif j == m-1:
                    matrix = np.array([[aux[i][j-1], aux[i][j-1], aux[i+1][j-1]], [aux[i][j], aux[i][j], aux[i+1][j]], [aux[i][j], aux[i][j], aux[i+1][j]]])
                else:
                    matrix = np.array([[aux[i][j-1], aux[i][j-1], aux[i+1][j-1]], [aux[i][j], aux[i][j], aux[i+1][j]], [aux[i][j+1], aux[i][j+1], aux[i+1][j+1]]])
            elif i == n-1:
                if j == 0:
                    matrix = np.array([[aux[i-1][j], aux[i][j], aux[i][j]], [aux[i-1][j], aux[i][j], aux[i][j]], [aux[i-1][j+1], aux[i][j+1], aux[i][j+1]]])
                elif j == m-1:
                    matrix = np.array([[aux[i-1][j-1], aux[i][j-1], aux[i][j-1]], [aux[i-1][j], aux[i][j], aux[i][j]], [aux[i-1][j], aux[i][j], aux[i][j]]])
                else:
                    matrix = np.array([[aux[i-1][j-1], aux[i][j-1], aux[i][j-1]], [aux[i-1][j], aux[i][j], aux[i][j]], [aux[i-1][j+1], aux[i][j+1], aux[i][j+1]]])
            else:
                if j == 0:
                    matrix = np.array([[aux[i-1][j], aux[i][j], aux[i][j+1]], [aux[i-1][j], aux[i][j], aux[i+1][j]], [aux[i-1][j+1], aux[i][j+1], aux[i+1][j+1]]])
                elif j == m-1:
                    matrix = np.array([[aux[i-1][j-1], aux[i][j-1], aux[i][j-1]], [aux[i-1][j], aux[i][j], aux[i+1][j]], [aux[i-1][j], aux[i][j], aux[i+1][j]]])
                else:
                    matrix = np.array([[aux[i-1][j-1], aux[i][j-1], aux[i+1][j-1]], [aux[i-1][j], aux[i][j], aux[i+1][j]], [aux[i-1][j+1], aux[i][j+1], aux[i+1][j+1]]])
            salida[i][j] = matrizConvolucion(matrix, kernel,True)
    
    return salida