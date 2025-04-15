#pip install opencv-python

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Importa e converte para RGB
img = cv2.imread('Satelite.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convertendo para preto e branco (RGB -> Gray Scale -> BW)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
a = img_gray.max()
_, thresh = cv2.threshold(img_gray, a/2*1.7, a, cv2.THRESH_BINARY_INV)

tamanhoKernel = 5
kernel = np.ones((tamanhoKernel, tamanhoKernel), np.uint8)

# Filtro de ruído (blurring)
img_blur = cv2.blur(img_gray, ksize=(tamanhoKernel, tamanhoKernel))

# Detecção de bordas com Canny
edges_blur = cv2.Canny(image=img_blur, threshold1=a/2, threshold2=a/2)

#Operador morfologico de fechamento
closed = cv2.morphologyEx(edges_blur, cv2.MORPH_CLOSE, kernel)

min_area = 1000 #define que areas de contorno com menos de 1000 pixels serao desconsiderados
contours_filtered, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #pega todos os objetos brancos 
satellite_only = np.zeros_like(closed) #cria uma imagem nova preta para ser adicionado os bordas do satelite

#percorre todos os contornos filtrados anteriormente  e percorre um a um 
for cnt in contours_filtered:
    #se o contorno tiver mais de 1000 pixels 
    if cv2.contourArea(cnt) > min_area:
        #ele e desenhado na imagem preta
        cv2.drawContours(satellite_only, [cnt], -1, 255, thickness=cv2.FILLED)


# Contornos
contours, hierarchy = cv2.findContours(
    image=satellite_only,
    mode=cv2.RETR_TREE,
    method=cv2.CHAIN_APPROX_SIMPLE
)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx=-1,
                         color=(255, 0, 0), thickness=2)

# Lista de imagens e seus nomes
imagens = [
    (img, "Imagem Original (RGB)"),
    (img_blur, "Imagem com Blur"),
    (img_gray, "Imagem em Tons de Cinza"),
    (edges_blur, "Bordas com Canny (com Blur)"),
    (closed, "Fechamento"),
    (satellite_only, "Imagem sem ruido"),
    (final, "Contorno Detectado")
]

# Plotando as imagens
formatoX = math.ceil(len(imagens) ** 0.5)
formatoY = formatoX if (formatoX ** 2 - len(imagens)) <= formatoX else formatoX - 1

for i in range(len(imagens)):
    plt.subplot(formatoY, formatoX, i + 1)
    plt.imshow(imagens[i][0], 'gray')
    plt.title(imagens[i][1], fontsize=8)
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
