#Vitor Giovani Dellinocente 9277875
#SCC0251 - Processamento de Imagens 2018/2
#T5 - Inpainting
import numpy as np
import imageio
import math

#   filtration(image, mask)
#   Função que filtra as imagens, zerando os valores maiores que 90% da magnitude
#   máxima da mascara e os valores menores que 1% da magnitude maxima da imagem
#   Parametros:
#       image -> imagem a ser filtrada
#       mask -> filtro
#   Retorno:
#       image -> imagem filtrada
def filtration(image, mask):
    magnitude_mask = np.absolute(np.amax(mask))
    magnitude_image = np.absolute(np.amax(image))
    image[np.logical_or(np.absolute(mask) >= 0.9*magnitude_mask, np.absolute(image) <= 0.01*magnitude_image)] = 0
    return image

#   convolution(image)
#   Função que faz a convulução entre uma imagem e um filtro médio, optei por faze-lo
#   no domínio das frequencias.
#   Parametros:
#       image -> imagem a ser filtrada
#   Retorno:
#       image -> imagem convoluida
def convolution(image):
    #Criação do filtro médio
    mean_filter = np.full((7,7), 1/49, dtype=np.float)
    f_mean_filter = np.fft.fft2(mean_filter, image.shape)
    return np.multiply(image, f_mean_filter)

#   normalize(image)
#   Função que normaliza dos valores da imagem para uint8 (0 - 255)
#   Parametros:
#       image -> imagem a ser normalizada
#   Retorno:
#       image -> imagem normalizada
def normalize(image):
    image = image.real
    imax = np.max(image)
    imin = np.min(image)
    image = 255*((image - imin)/(imax-imin))
    image = image.astype(dtype=np.uint8)
    return image

#   insert_pixels(g0, deteriorated, mask)
#   Função que insere os pixels onde há deformações na imagem
#   Parametros:
#       g0 -> imagem original
#       deteriorated -> gk, imagem que esta sendo restaurada, na sua k-esima iteração
#       mask -> filtro usado na restauração
#   Retorno:
#       imagem restaurada
def insert_pixels(g0, deteriorated, mask):
    return np.multiply(1-mask/np.max(mask), g0) + np.multiply(mask/np.max(mask), deteriorated)

#   calculate_error(original, image_filtrated)
#   Funcao que calcula o erro (RSME) entre duas imagens, a original e a filtrada
#   Parametros:
#       original -> imagem original
#       image_filtrated -> imagem filtrada
#   Retorno:
#       o erro entre as duas imagens
def calculate_error(original, image_filtrated):
    error = 0.0
    den = original.shape[0]*original.shape[1]
    error = np.sum(np.multiply((original-image_filtrated),(original-image_filtrated)))
    error = error/den
    return math.sqrt(error)

# Entrada de dados
original_name = str(input()).rstrip()
deteriorated_name = str(input()).rstrip()
mask_name = str(input()).rstrip()
T = int(input())

# Leitura das imagens, original e danificada(para realizarmos a restauração), e por ultimo o mascara usada no processo
original = imageio.imread(original_name)
deteriorated = imageio.imread(deteriorated_name)
g0 = imageio.imread(deteriorated_name)
mask = imageio.imread(mask_name)

f_mask =  np.fft.fft2(mask)

#  Iterações da restauração
for k in range(T):
    f_deteriorated = np.fft.fft2(deteriorated)
    f_deteriorated = filtration(f_deteriorated, f_mask)
    f_deteriorated = convolution(f_deteriorated)
    deteriorated = np.fft.ifft2(f_deteriorated)
    deteriorated = normalize(deteriorated)
    deteriorated = insert_pixels(g0,deteriorated, mask)

# Renormalização das imagens
original = normalize(original)
deteriorated = normalize(deteriorated)

#  Calculo do erro
print('%.5f' %calculate_error(original, deteriorated))
