#Vitor Giovani Dellinocente 9277875
#SCC0251 - Processamento de Imagens 2018/2
#T6 - Restauração
import numpy as np
import imageio
import math

#   calculate_error(original, image_filtrated)
#   Funcao que calcula o erro (RSME) entre duas imagens, a original e a filtrada
#   Parametros:
#       original -> imagem original
#       image_filtrated -> imagem filtrada
#   Retorno:
#       o erro entre as duas imagen
def calculate_error(Icomp, Iout):
    error = 0.0
    den = Icomp.shape[0]*Icomp.shape[1]
    error = np.sum(np.multiply((Icomp-Iout),(Icomp-Iout)))
    error = error/den
    return math.sqrt(error)
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

#   safe_power(neightbors, n, Q)
#   Funcao que faz o calculo da potencia de maneira segura, ou seja, em nenhum
#   momento haverá o número 0 sendo elevado a um expoente negativo. Além disso
#   essa funcao ja calcula o somatorio de g^Q ou g^Q+1, dependendo do parametro
#   Parametros:
#       neightbors -> vizinhanca do pixel a ser calculado
#       n -> tamanho da vizinhanca
#       Q -> ordem do filto
#   Retorno:
#       g^Q ou g^Q+1 calculado
def safe_power(neightbors, n, Q):
    sum = 0.0
    for i in range(n*n):
        if(neightbors[i] != 0):
            sum += math.pow(neightbors[i], Q)
    return sum

#   counter_harmonic_mean_filter(Inoisy, n, Q):
#   Função que restaura uma imagem a partir do Filtro da média contra-harmônica.
#   Parametros:
#       Inoisy -> Imagem danificada
#       n -> tamanho do filtro
#       Q -> ordem do filtro
#   Retorno:
#       Iout -> Imagem restaurada
def counter_harmonic_mean_filter(Inoisy, n, Q):
    pad = int(n/2)
    n_row = Inoisy.shape[0]
    n_col = Inoisy.shape[1]

    Iout = np.zeros((n_row, n_col), dtype=np.float64)
    Inoisy_wraped = np.pad(Inoisy,((pad, pad), (pad, pad)) ,'constant', constant_values=0)

    neightbors = make_neightbors_matrix(Inoisy_wraped, n_row, n_col, n)

    for i in range(Iout.shape[0]):
        for j in range(Iout.shape[1]):
            x = safe_power(neightbors[i, j], n, Q+1)
            y = safe_power(neightbors[i, j], n, Q)
            if(y == 0 or x == 0):
                Iout[i, j] = Inoisy[i, j]
            else:
                Iout[i, j] = x/y
    return Iout

#   stage_a(Inoisy, Inoisy_wraped, n, m, x, y, i, j):
#   Etapa A do metodo de restauração usando o Filtro adaptativo de mediana
#   Parametros:
#       Inoisy -> Imagem danificada
#       Inoisy_wraped -> Imagem com o pad realizado (extendida)
#       n -> tamanho do filtro
#       m -> tamanho máximo do filtro
#       x -> linha atual de Inoisy_wraped
#       y -> coluna autal de Inoisy_wraped
#       i -> linha atual de Inoisy
#       j -> coluna atual de Inoisy
#   Retorno:
#       Valor a ser inserido em Iout[i, j], ou seja, pixeal restaurado da imagem final
def stage_a(Inoisy, Inoisy_wraped, n, m, x, y, i, j):
    neightbors = Inoisy_wraped[x:x+n, y:y+n]
    z_med = np.median(neightbors)
    z_min = np.amin(neightbors)
    z_max = np.amax(neightbors)
    A1 = z_med - z_min
    A2 = z_med - z_max

    if (A1 > 0 and A2 < 0):
        return stage_b(Inoisy, x, y, z_min, z_med, z_max, i, j)
    else:
        if (int(n/2) + 1 < int(m/2)):
            return stage_a(Inoisy, Inoisy_wraped, n+1, m, x-1, y-1, i, j)
        else:
            return z_med

#   stage_b(Inoisy, x, y, z_min, z_med, z_max, i, j):
#   Etapa B do metodo de restauração usando o Filtro adaptativo de mediana
#   Parametros:
#       Inoisy -> Imagem danificada
#       x -> linha atual de Inoisy_wraped
#       y -> coluna autal de Inoisy_wraped
#       z_min -> valor minimo dos vizinhos do pixel da posição i, j
#       z_med -> valor da mediana dos vizinhos do pixel da posição i, j
#       z_max -> valor máximo dos vizinhos do pixel da posição i, j
#       i -> linha atual de Inoisy
#       j -> coluna atual de Inoisy
#   Retorno:
#       Valor a ser inserido em Iout[i, j], ou seja, pixeal restaurado da imagem final
def stage_b(Inoisy, x, y, z_min, z_med, z_max, i, j):
    B1 = float(Inoisy[i, j]) - z_min
    B2 = z_med - z_max
    if (B1 > 0 and B2 < 0):
        return Inoisy[i, j]
    else:
        return z_med

#   adaptative_median_filter(Inoisy, n, m):
#   Funcao que realiza o método de restauração a partir do Filtro adaptativo de mediana
#   Parametros:
#       Inoisy -> Imagem danificada
#       n -> tamanho do filtro
#       m -> tamanho máximo do filtro
#   Retorno:
#       Imagem restaurada
def adaptative_median_filter(Inoisy, n, m):
    pad = int(m/2)
    pad_aux = int(n/2)
    Iout = np.zeros(Inoisy.shape, dtype=np.float64)

    # Caso o tamanho maximo do filtro seja par
    if(m % 2 == 0):
        Inoisy_wraped = np.pad(Inoisy,((pad, pad), (pad, pad)) ,'edge')
        for i in range(Inoisy.shape[0]):
            for j in range(Inoisy.shape[1]):
                Iout[i, j] = stage_a(Inoisy, Inoisy_wraped, n, m, i + (m-n), j + (m-n), i, j)
    # Caso o tamanho máximo do filtro seja impar
    else:
        Inoisy_wraped = np.pad(Inoisy,((pad+ 1, pad + 1), (pad + 1, pad + 1)) ,'edge')
        for i in range(Inoisy.shape[0]):
            for j in range(Inoisy.shape[1]):
                Iout[i, j] = stage_a(Inoisy, Inoisy_wraped, n, m, i + (pad-pad_aux) + 1, j + (pad-pad_aux) + 1, i, j)

    return Iout

#   make_neightbors_matrix(Inoisy_wraped, n_row, n_col, n):
#   Funcao que monta uma matriz 3d que em cada posicao x, y, tem-se um vetor
#   com os vizinhos do pixel x, y
#   Parametros:
#       Inoisy_wraped -> Imagem danificada extendida
#       n_row -> numero de linhas da imagem danificada
#       n_col -> numero de colunas da imagem danificada
#       n -> tamanho do filtro
#   Retorno:
#       Matriz contendo os vizinhos de cada pixel
def make_neightbors_matrix(Inoisy_wraped, n_row, n_col, n):
    neightbors = np.zeros((n_row, n_col, (n*n)), dtype=np.float64)
    for i in range(n_row):
        for j in range(n_col):
            neightbors[i, j] = Inoisy_wraped[i:i+n, j:j+n].flatten()

    return neightbors

#   adaptative_local_noise_reduction_filter(Inoisy, n, sigma_noisy)
#   Funcao que realizado o método de restauração de imagens utilizando o Filtro adaptativo de redução de ruído local
#   Parametros:
#       Inoisy -> Imagem danificada
#       n -> tamanho do filtro
#       sigma_noisy -> valor da distribuição de ruído
#   Retorno:
#       Imagem restaurada
def adaptative_local_noise_reduction_filter(Inoisy, n, sigma_noisy):
    variance = math.pow(sigma_noisy, 2)

    pad = int(n/2)
    n_row = Inoisy.shape[0]
    n_col = Inoisy.shape[1]

    Iout = np.zeros(Inoisy.shape, dtype=np.float64)
    Inoisy_wraped = np.pad(Inoisy,((pad, pad), (pad, pad)) ,'wrap')

    neightbors = make_neightbors_matrix(Inoisy_wraped, n_row, n_col, n)

    Iout[:,:] = Inoisy[:,:] - (variance/np.var(neightbors[:,:]))*(Inoisy[:,:] - np.mean(neightbors[:,:]))

    return Iout

# Entrada dos nomes das imagens, original e com ruido, respectivamente
Icomp_name = str(input()).rstrip()
Inoisy_name = str(input()).rstrip()

# Entrada do método de restauração e do tamanho do filtro
method = int(input())
n = int(input())

#  Carregando as imagens
Icomp = imageio.imread(Icomp_name)
Inoisy = imageio.imread(Inoisy_name)

#  Escolha do método
if (method == 1):
    sigma_noisy = float(input())
    Iout = adaptative_local_noise_reduction_filter(Inoisy, n, sigma_noisy)
elif (method == 2):
    m = int(input())
    Iout = adaptative_median_filter(Inoisy, n, m)
elif (method == 3):
    Q = float(input())
    Iout = counter_harmonic_mean_filter(Inoisy, n, Q)


print('%.5f' %calculate_error(Icomp, Iout))
