#Vitor Giovani Dellinocente 9277875
#SCC0251 - Processamento de Imagens 2018/2
#T3 - Filtragem 1D
import numpy as np
import imageio
import math

#	calculateError(A, B)
#	Calcula o erro entre duas imagens
#	Parametros:
#		A -> Imagem referencia
#		B -> Imagem filtrada
#	Retorno:
#		Erro entre as imagens
def calculateError(A, B):
    error = 0.0
    den = A.shape[0]*A.shape[1]
    error = np.sum((A-B)*(A-B))
    error = error/den
    return math.sqrt(error)

#	espacialDomain(image, filter, n):
#	Faz a filtragem uma filtragem de uma imagem usando o domínio espacial
#	ou seja, passa-se um filtro por toda a imagem
#	Parametros:
#		image -> imagem a ser filtrada
#		filter -> filtro que será passado na imagem
#		n -> tamanho do filtro
#	Retorno:
#		new_image -> imagem filtrada
def espacialDomain(image, filter, n):
    new_image = np.zeros(image.shape[0])
    d = int(n/2)
    image = np.pad(image, (d, d), 'wrap')
    if (n % 2 == 0):
        for i in range(1, (image.shape - n + 1)):
            new_image[i - 1] += math.floor(np.sum(image[i:(i+n)]*filter))

    else:
        for i in range((image.shape[0] - n + 1)):
            new_image[i] += math.floor(np.sum(image[i:(i+n)]*filter))

    return new_image

#	createGaussianFilter(sigma, n):
#	Cria um filtro gaussiano baseando os valores em um sigma informado pelo usuário
#	Parametros:
#		sigma -> Parametro para o calculo do filtro
#		n -> tamanho do filtro
#	Retorno:
#		filter -> filtro gerado
def createGaussianFilter(sigma, n):
    filter = np.array(np.linspace(int(-n/2), int(n/2), n), dtype=np.float)
    for i in range(n):
        filter[i] = float((1/math.sqrt(2*math.pi))*math.exp(-math.pow(filter[i], 2)/(2*math.pow(sigma, 2))))

    norm = np.sum(filter)
    filter = filter/norm
    return filter

#	fourierTransform(img):
#	Faz a transformada de fourier no filtro ou na imagem (depende do parametro passado), ou seja, leva
#	o filtro e a imagem para o dominio das frequencias
#	Parametros:
#		img -> vetor que será levado ao dominio das frequencias
#	Retorno:
#		F -> vetor com a transformada realizada
def fourierTransform(img):
    F = np.zeros(img.shape[0], dtype=np.complex64)
    n = img.shape[0]

    x = np.arange(n)

    for i in range(n):
        F[i] = np.sum(np.multiply(img, np.exp((1j*2*np.pi*i*x) /n)))

    return F

#	invFourierTransform(img):
#	Faz o inverso da transformada de fourier no filtro ou na nova imagem, ou seja, leva
#	a imagem do dominio das frequencias para o dominio da propria imagem
#	Parametros:
#		F -> imagem no domínio das frequencias
#	Retorno:
#		img -> imagem filtrada
def invFourierTransform(F):
    img = np.zeros(F.shape[0], dtype=np.float32)
    n = F.shape[0]

    u = np.arange(n)

    for i in range(n):
        img[i] = np.real(np.sum(np.multiply(F, np.exp((1j*2*np.pi*u*i) /n))))

    return img/n

#	frequenceDomain(img, filter, n):
#	Faz a filtragem usando o vetor de frequencias da imagem e o filtro, ou seja
#	o filtro e a imagem são multiplicados ponto a ponto para realizar a operação de filtragem
#	Parametros:
#		img -> imagem a ser filtrada
#		filter -> filtro que será usado
#		n -> tamanho do filtro
#	Retorno:
#		imagem filtrada
def frequenceDomain(img, filter, n):
    newFilter = np.zeros(img.shape[0])
    newFilter[0:n] = filter[:]
    imgF = fourierTransform(img)
    filterF = fourierTransform(newFilter)
    newImage = np.multiply(imgF, filterF)
    return invFourierTransform(newImage)

# entrada de dados
img_name = str(input()).rstrip()
filter_type = int(input())
n = int(input())

#escolha do tipo de filtragem
if (filter_type == 1):
    filter = np.array(input().split(), dtype=np.float)
elif (filter_type == 2):
    sigma = float(input())
    filter = createGaussianFilter(sigma, n)

# entrada no tipo de dominio
domain = int(input())

original = imageio.imread(img_name)
x, y = original.shape
original = original.reshape(int(original.shape[0]*original.shape[1]))

# filtragem da imagem
if (domain == 1):
    image = espacialDomain(original, filter,n)
else:
    image = frequenceDomain(original, filter, n)


# colocando os valores da imagem em uint8
image = image * (255/np.amax(image))
image = image.astype(np.uint8)

original = np.reshape(original, (x,y))
image = np.reshape(image, (x, y))

# calculando o erro entre as imagens, original e filtrada
print('%.4f' %calculateError(original, image))
