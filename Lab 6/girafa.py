#pip install opencv-python

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Importa e converte para RGB
img = cv2.imread('GIRAFA.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convertendo para preto e branco (RGB -> Gray Scale -> BW)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
a = img_gray.max()
_, thresh = cv2.threshold(img_gray, a / 2 * 1.7, a, cv2.THRESH_BINARY_INV)

# Kernel para operações morfológicas
tamanhoKernel = 5
kernel = np.ones((tamanhoKernel, tamanhoKernel), np.uint8)


# Contornos
contours, hierarchy = cv2.findContours(
    image=thresh,
    mode=cv2.RETR_TREE,
    method=cv2.CHAIN_APPROX_SIMPLE
)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx=-1,
                         color=(255, 0, 0), thickness=2)

# Lista de imagens e títulos das operações
imagens = [
    (img, "Imagem Original (RGB)"),
    (img_gray, "Imagem em Tons de Cinza"),
    (thresh, "Imagem Binarizada (Threshold)"),
    (final, "Contornos Detectados")
]

# Plotando as imagens com os títulos
formatoX = math.ceil(len(imagens) ** 0.5)
formatoY = formatoX if (formatoX ** 2 - len(imagens)) <= formatoX else formatoX - 1

for i in range(len(imagens)):
    plt.subplot(formatoY, formatoX, i + 1)
    plt.imshow(imagens[i][0], 'gray')
    plt.title(imagens[i][1], fontsize=8)
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
