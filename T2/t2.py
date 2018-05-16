#Vitor Giovani Dellinocente 9277875
#SCC0251 - Processamento de Imagens 2018/2
#T2 - Realce e Superresolução

import numpy as np
import imageio
import math

#	calculateError(img, gen_img)
#	Calcula o erro entre duas imagens
#	Parametros:
#		img -> Imagem referencia
#		gen_img -> Iamagem gerada
#	Retorno:
#		Erro entre as imagens
def calculateError(img, gen_img):
	error = 0.0
	den = img.shape[0] * img.shape[1]
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			error += math.pow(img[i, j] - gen_img[i, j], 2)

	error = error/den

	return math.sqrt(error)


#	individualTransfer(img)
#	Faz realce da imagem usando o histograma de cada imagem
#	Parametros:
#		img -> imagem
#	Retorno:
#		imagem realçada
def individualTransfer(img):
	hist = np.histogram(img, 256, [0, 255])
	cumsum = np.cumsum(hist[0])
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			img[i, j] = (cumsum[img[i, j]]) * (255/(img.shape[0]*img.shape[1]))
	return img

#	jointTransfer(L1, L2, L3, L4)
#	Faz realce da imagem usando um unico histograma acumulado de todas as Imagens
#	Parametros:
#		L1 -> imagem 1
#		L2 -> imagem 2
#		L3 -> imagem 3
#		L4 -> imagem 4
#	Retorno:
#		Todas as imagens realçadas

def jointTransfer(L1, L2, L3, L4):
	hist1 = np.histogram(L1, 256, [0,255])
	hist2 = np.histogram(L2, 256, [0,255])
	hist3 = np.histogram(L3, 256, [0,255])
	hist4 = np.histogram(L4, 256, [0,255])

	cumsum1 = np.cumsum(hist1[0])
	cumsum2 = np.cumsum(hist2[0])
	cumsum3 = np.cumsum(hist3[0])
	cumsum4 = np.cumsum(hist4[0])

	cumsum = (cumsum1 + cumsum2 + cumsum3 + cumsum4)/4

	for i in range(L1.shape[0]):
		for j in range(L1.shape[1]):
			L1[i, j] = (cumsum[L1[i, j]]) * (255/(L1.shape[0]*L1.shape[1]))
			L2[i, j] = (cumsum[L2[i, j]]) * (255/(L2.shape[0]*L2.shape[1]))
			L3[i, j] = (cumsum[L3[i, j]]) * (255/(L3.shape[0]*L3.shape[1]))
			L4[i, j] = (cumsum[L4[i, j]]) * (255/(L4.shape[0]*L4.shape[1]))
	return L1, L2, L3, L4

#	gammaAdjust(img, gamma)
#	Faz o ajuste gama de uma imagem
#	Parametros:
#		img -> Imagem
#		gamma -> Parametro Gamma da formula de ajuste
#	Retorno:
#		imagem com ajustes
def gammaAdjust(img, gamma):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = math.floor(255*math.pow((img[i, j]/255.0), (1/gamma)))
    return img

#	superResolution(L1, L2, L3, L4)
#	Pega quatro imagens de uma resolução menor e gera uma imagem com resolução maior
#	Parametros:
#		L1 -> Imagem com resolução menor 1
#		L2 -> Imagem com resolução menor 2
#		L3 -> Imagem com resolução menor 3
#		L4 -> Imagem com resolução menor 4
#	Retorno:
#		Imagem gerada com resolução maior
def superResolution(L1, L2, L3, L4):
    x = L1.shape[0]
    y = L1.shape[1]

    H  = np.zeros((2*x, 2*y))

    for i in range(x):
        for j in range(y):
            H[2*i, 2*j] = L1[i, j]
            H[2*i + 1, 2*j] = L2[i, j]
            H[2*i, 2*j + 1] = L3[i, j]
            H[2*i + 1, 2*j + 1] = L4[i, j]

    return H


#Entrada dos parametros e nome das imagens
low_name = str(input()).rstrip()
high_name = str(input()).rstrip()
method = int(input())
gamma = float(input())

#Carrega imagens de baixa resolução
lowimg1 = imageio.imread(low_name + "1.png")
lowimg2 = imageio.imread(low_name + "2.png")
lowimg3 = imageio.imread(low_name + "3.png")
lowimg4 = imageio.imread(low_name + "4.png")

#Carrega imagem de referencia
highimgReference = imageio.imread(high_name + ".png")

if (method == 1):
    lowimg1 = individualTransfer(lowimg1)
    lowimg2 = individualTransfer(lowimg2)
    lowimg3 = individualTransfer(lowimg3)
    lowimg4 = individualTransfer(lowimg4)
elif (method == 2):
    lowimg1, lowimg2, lowimg3, lowimg4 = jointTransfer(lowimg1, lowimg2, lowimg3, lowimg4)
elif (method == 3):
    lowimg1 = gammaAdjust(lowimg1, gamma)
    lowimg2 = gammaAdjust(lowimg2, gamma)
    lowimg3 = gammaAdjust(lowimg3, gamma)
    lowimg4 = gammaAdjust(lowimg4, gamma)


highimg = superResolution(lowimg1, lowimg2, lowimg3, lowimg4)

print('%.4f' %calculateError(highimgReference, highimg))
