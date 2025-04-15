#pip install opencv-python

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Importa e converte para RGB
img = cv2.imread('aviao.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convertendo para preto e branco (RGB -> Gray Scale -> BW)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
a = img_gray.max()
_, thresh = cv2.threshold(img_gray, a / 2 * 1.7, a, cv2.THRESH_BINARY_INV)

# Kernel morfológico
tamanhoKernel = 5
kernel = np.ones((tamanhoKernel, tamanhoKernel), np.uint8)

# Filtro de ruído (blur)
img_blur = cv2.blur(img_gray, ksize=(tamanhoKernel, tamanhoKernel))

# Detecção de bordas com Canny
edges_gray = cv2.Canny(image=img_gray, threshold1=a / 2, threshold2=a / 2)

#Operador morfologico de fechamento
closed = cv2.morphologyEx(edges_gray, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #pega todos os objetos brancos 
plane_only = np.zeros_like(closed) #cria uma imagem nova preta para ser adicionado os bordas do aviao

#define o centro da imagem que e onde o aviao esta
image_center = (img.shape[1]//2, img.shape[0]//2) 
closest_contour = None
min_dist = 500

#para cada ponto selecionado nas bordas com canny
for cnt in contours:
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        dist = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
        #se a distancia do contorno for menor que a variavel de controle de distancia
        if dist < min_dist:
            min_dist = dist
            closest_contour = cnt

#desenha na imagem preta o resultado
plane_only = np.zeros_like(closed)
cv2.drawContours(plane_only, [closest_contour], -1, 255, cv2.FILLED)


# Detecção de contornos
contours, hierarchy = cv2.findContours(
    image=plane_only,
    mode=cv2.RETR_TREE,
    method=cv2.CHAIN_APPROX_SIMPLE
)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx=-1,
                         color=(255, 0, 0), thickness=2)

# Lista de imagens com nomes descritivos
imagens = [
    (img, "Imagem Original (RGB)"),
    (img_blur, "Imagem com Blur"),
    (img_gray, "Imagem em Tons de Cinza"),
    (edges_gray, "Bordas com Canny (sem Blur)"),
    (closed, "Fechamento"),
    (plane_only, "Imagem sem ruido"),
    (final, "Contornos Detectados")
]

# Plotando com títulos
formatoX = math.ceil(len(imagens) ** 0.5)
formatoY = formatoX if (formatoX ** 2 - len(imagens)) <= formatoX else formatoX - 1

for i in range(len(imagens)):
    plt.subplot(formatoY, formatoX, i + 1)
    plt.imshow(imagens[i][0], 'gray')
    plt.title(imagens[i][1], fontsize=8)
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
