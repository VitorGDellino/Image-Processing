#	T1 Processamento de Imagens
#	Vitor Giovani Dellinocente 9277875

import numpy as np
import math
import random

#	getMaxOfSubmatrix(matrix, d, x, y)
#	Procura o maior valor de uma submatriz dada uma posição inicial
#	Parametros
#		matrix -> imagem
#		d -> tamanho da submatriz
#		x -> posição em x do inicio da submatriz
#		y -> posição em y do inicio da submatriz
#	Retorno
#		max -> maior valor da submatriz
def getMaxOfSubmatrix(matrix, d, x, y):
	max = np.amax(matrix[x*d:x*d + d, y*d:y*d + d])
	return max

#	sampling(matrix, C, N)
#	Faz a amostragem de uma matriz para criar uma imagem digitalizada
#	Parametros
#		matrix -> imagem
#		C -> tamanho da imagem original
#		N -> Tamanho da matriz digitalizada
#	Retorno
#		dig_img -> imagem digitalizada
def sampling(matrix, C, N):
	dig_img = np.zeros((N, N))
	d = int(C/N)
	for i in range(N):
		for j in range(N):
			dig_img[i, j] = getMaxOfSubmatrix(matrix, d, i, j)

	return dig_img

#	calculateError(A, B)
#	Calcula o erro entre duas imgens
#	Parametros
#		A -> imagem gerada
#		B -> imagem referencia
#	Retorno
#		error -> erro entre as imagnes
def calculateError(A, B):
	error = 0.0
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			error += math.pow(A[i, j] - B[i, j], 2)

	error = math.sqrt(error)
	return error

#	bitDislocation(matrix, B)
#	Desloca os valores de uma imagem em 8-B bits
#	Parametros
#		matrix -> imagem digitalizada
#		B -> numero de bits significativos
#	Retorno
#		imagem com valores shiftados
def bitDislocation(matrix, B):
	return np.int64(matrix) >> (8-B)

#	quantization(matrix)
#	Coloca os valores dos pixels da imagem entre 0 e 255 (uint8)
#	Parametros
#		matrix -> imagem digitalizada
#	Retorno
#		imagem quantizada
def quantization(matrix):
	max = np.amax(matrix)
	for i in range (matrix.shape[0]):
		for j in range (matrix.shape[1]):
			matrix[i, j] = matrix[i, j] * (255/max)
	return matrix.astype(np.uint8)

#	generateImageFunctionOne(matrix)
#	Cria uma imagem com valores gerados por uma dada função
#	Parametros
#		matrix -> imagem que sera criada
#	Retorno
#		matrix -> imagem criada
def generateImageFunctionOne(matrix):
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			matrix[i,j] = float(i + j)

	return matrix

#	generateImageFunctionTwo(matrix, Q)
#	Cria uma imagem com valores gerados por uma dada função
#	Parametros
#		matrix -> imagem que sera criada
#		Q -> parametro para funcao geradora
#	Retorno
#		matrix -> imagem criada
def generateImageFunctionTwo(matrix, Q):
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			matrix[i, j] = math.fabs(float(math.sin(i/Q)) + float(math.sin(j/Q)))

	return matrix

#	generateImageFunctionThree(matrix, Q)
#	Cria uma imagem com valores gerados por uma dada função
#	Parametros
#		matrix -> imagem que sera criada
#		Q -> parametro para funcao geradora
#	Retorno
#		matrix -> imagem criada
def generateImageFunctionThree(matrix, Q):
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			matrix[i, j] = math.fabs(float(i/Q) - float(math.sqrt(j/Q)))
	return matrix

#	generateImageFunctionTwo(matrix, S)
#	Cria uma imagem com valores aleatorios a partir de uma semente
#	Parametros
#		matrix -> imagem que sera criada
#		S -> semente para os numeros aleatorios
#	Retorno
#		matrix -> imagem criada
def generateImageFunctionFour(matrix, S):
	random.seed(S)
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			matrix[i, j] = random.random()
	return matrix

#	generateImageRandomWalk(matrix, C, S)
#	Cria uma imagem com valores em lugares aleatorios
#	Parametros
#		matrix -> imagem que sera criada
#		S -> semente para os numeros aleatorios
#		C -> tamanho da imagem
#	Retorno
#		matrix -> imagem criada
def generateImageRandomWalk(matrix, C, S):
	random.seed(S)
	x = y = 0
	matrix[x,y] = dx = dy = 1
	for i in range(int(1+(C*C)/2)):
		dx = random.randint(-1, 1)
		x = np.mod((x + dx), C)
		matrix[x, y] = 1
		dy = random.randint(-1, 1)
		y = np.mod((dy + y), C)
		matrix[x, y] = 1

	return matrix


filename = str(input()).rstrip()
img_reference = np.load(filename)
C = int(input())
ftype = int(input())
Q = int(input())
N = int(input())
B = int(input())
S = int(input())


#Gerar a imagem da cena
img_generated = np.zeros((C, C))

if ftype == 1:
	img_generated = generateImageFunctionOne(img_generated)
elif ftype == 2:
	img_generated = generateImageFunctionTwo(img_generated, Q)
elif ftype == 3:
	img_generated = generateImageFunctionThree(img_generated, Q)
elif ftype == 4:
	img_generated = generateImageFunctionFour(img_generated, S)
elif ftype == 5:
	img_generated = generateImageRandomWalk(img_generated, C, S)

#digitalizar a imagem gerada
img_generated = sampling(img_generated, C, N)
img_generated = quantization(img_generated)
img_generated = bitDislocation(img_generated, B)

#calcular erro entre as imagens
error = calculateError(img_generated, img_reference)
print (error)
