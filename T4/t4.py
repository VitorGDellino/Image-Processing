#Vitor Giovani Dellinocente 9277875
#SCC0251 - Processamento de Imagens 2018/2
#T3 - Filtragem 2D
import numpy as np
import imageio
import math

#	arbitrary(image, filter):
#	Realiza a filtragem na imagem a partir de um filtro criado pelo usuário
#	Parametros:
#		image -> a imagem
#		filter -> o filtro
#	Return:
#		A imagem filtrada e no domínio das frequencias
def arbitrary(image, filter):
	f_image = np.fft.fft2(image)
	f_filter = np.zeros(image.shape)
	f_filter[0:filter.shape[0], 0:filter.shape[1]] = filter[:, :]
	f_filter = np.fft.fft2(f_filter)
	return np.multiply(f_image, f_filter)

#	normalize_filter(filter):
#	Normaliza o filtro laplaciana da gaussiana
#	Parametros:
#		filter -> o filtro gerado
#	Return:
#		o filtro normalizado
def normalize_filter(filter):
	sum_pos = np.sum(filter[np.where(filter > 0)])
	sum_neg = np.sum(filter[np.where(filter < 0)])

	for i in range (filter.shape[0]):
		for j in range (filter.shape[1]):
			if (filter[i, j] < 0):
				filter[i,j] =  filter[i,j]*(-sum_pos/sum_neg)

	return filter

#	generate_laplacian_gaussian_filter(n, sigma):
#	Gera um filtro laplaciana da gaussiana dado um tamanho e um parametro sigma
#	Parametros:
#		n -> tamanho do filtro
#		sigma -> valor sigma para gerar o filtro
#	Return:
#		o filtro gerado e normalizado
def generate_laplacian_gaussian_filter(n, sigma):
	X = np.empty(n)
	Y = np.empty(n)
	M = np.empty((n, n))
	X[:] = np.linspace(-5.0, 5.0, n)
	Y[:] = np.linspace(5.0, -5.0, n)
	for i in range(n):
		for j in range(n):
			M[i, j] = ((-1/(math.pi*math.pow(sigma, 4))) * (1 - (math.pow(X[j], 2) + math.pow(Y[i], 2))/(2*math.pow(sigma, 2))) * math.exp(-(math.pow(X[j], 2) + math.pow(Y[i], 2))/(2*math.pow(sigma, 2))))

	return normalize_filter(M)

#	laplacian_gaussian(image, n, sigma):
#	Realiza uma filtragem utilizando o filtro laplaciana da gaussiana
#	Parametros:
#		img -> uma imagem
#		n -> tamanho do filtro
#		sigma -> valor sigma para gerar o filtro
#	Return:
#		imagem ja filtrada e no domínio das frequencias
def laplacian_gaussian(image, n, sigma):
	f_image = np.fft.fft2(image)
	f_filter = np.zeros(image.shape)
	filter = generate_laplacian_gaussian_filter(n, sigma)
	f_filter[0:filter.shape[0], 0:filter.shape[1]] = filter[:, :]
	f_filter = np.fft.fft2(f_filter)

	return np.multiply(f_image, f_filter)

#	convolution(img, filter):
#	Realiza uma convolucao na image
#	Parametros:
#		img -> uma imagem
#		filter -> um filtro
#	Return:
#		imagem após a convolucao
def convolution(img, filter):
	pad = int(filter.shape[0]/2)
	dim_filter = filter.shape
	new_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
	img = np.pad(img, ((pad, pad), (pad, pad)),'constant', constant_values = 0)
	for i in range((img.shape[0] - dim_filter[0] + 1)):
		for j in range((img.shape[1] - dim_filter[1] + 1)):
			new_img[i, j] += np.sum(np.multiply(img[i:i+dim_filter[0],j:j+dim_filter[1]],filter))

	return new_img

#	sobel(img):
#	Realiza uma filtragem na imagem utilizando o operador sobel
#	Parametros:
#		img -> uma imagem
#	Return:
#		imagem ja filtrada e no domínio das frequencias
def sobel(img):
	Fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], np.float)
	Fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float)

	Ix = convolution(img, Fx)
	Iy = convolution(img, Fy)

	Iout = np.sqrt(np.multiply(Ix, Ix) + np.multiply(Iy, Iy))

	return np.fft.fft2(Iout)

#	first_cut(img):
#	Realiza um corte da imagem para deixa-la com 1/4 de seu tamanho original
#	Parametros:
#		img -> uma imagem
#	Return:
#		imagem cortada
def first_cut(img):
	return  img[0:int(img.shape[0]/2), 0:int(img.shape[1]/2)]

#	second_cut(img, cut_position):
#	Realiza um corte da imagem
#	Parametros:
#		img -> uma imagem
#		cut_position -> posicoes onde os cortes serao realizados
#	Return:
#		imagem cortada
def second_cut(img, cut_position):
	hi = int(img.shape[0] * cut_position[0])
	hf = int(img.shape[0] * cut_position[1])

	wi = int(img.shape[1] * cut_position[2])
	wf = int(img.shape[1] * cut_position[3])

	return img[hi:hf, wi:wf]

#	knn(Iout, dataset, k=1):
#	Realiza a classificação usando o algoritmo knn, com k = 1
#	Parametros:
#		Iout -> Imagem a ser classificada
#		dataset -> Conjunto de dados para comparar a imagem
#		k -> número de vizinhos,no caso, sempre 1
#	Return:
#		index -> indice do conjunto de labels que classifica a imagem
def knn(Iout, dataset, k=1):
	lowest_dist = float('inf')
	dist = 0.0
	for row in range(dataset.shape[0]):
		res = np.absolute(Iout - dataset[row])
		dist = np.sqrt(np.sum(np.multiply(res, res)))
		if dist < lowest_dist:
			lowest_dist = dist
			index = row

	return index

#Entrada de dados
img_name = str(input()).rstrip()
method = int(input())

if (method == 1):
	dim = np.array(input().split(), dtype=np.int)
	filter = np.empty((dim[0], dim[1]), dtype=np.float)

	for i in range(dim[0]):
		filter[i] = np.array(input().split(), dtype=np.float)

elif (method == 2):
	n = int(input())
	sigma = float(input())


cut_position = np.array(input().split(), dtype=np.float)

dataset_file_name = str(input()).rstrip()
labels_file_name = str(input()).rstrip()

original = imageio.imread(img_name)

#Realiza umas das filtragens escolhidas
if (method == 1):
	Iout = arbitrary(original, filter)
elif (method == 2):
    Iout = laplacian_gaussian(original, n, sigma)
elif (method == 3):
	Iout = sobel(original)

#Realiza o primeiro corte
Icut1 = first_cut(Iout)

#Realiza o segundo corte
Icut2 = second_cut(Icut1, cut_position)

dataset = np.load(dataset_file_name)
labels = np.load(labels_file_name)

#Faz a classificação da imagem
index = knn(Icut2.flatten(), dataset, k=1)

print(labels[index])
print(index)
