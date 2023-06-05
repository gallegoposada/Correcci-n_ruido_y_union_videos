#https://www.youtube.com/watch?v=y4v6K3-s3mE

import cv2 # OpenCV para computer vision
import numpy as np # Para cálculo de matrices
import matplotlib.pyplot as plt #Para graficar
import os #Habilita el manejo de directorios


Ruta = r'C:\Users\user\Desktop\Procesamiento de Imagenes\Parcial_2\GAL_COSTA.JPG'#Ubicación de la imagen desde el google drive
Imagen = cv2.imread(Ruta)#Lee
Imagen=Imagen[:,:,[2,1,0]]#Organiza
print('Mostrando imagen con ruido')
plt.imshow(Imagen.astype('uint8'),vmin=0, vmax=255) #Muestra
print('')

Img2 = cv2.fastNlMeansDenoisingColored(Imagen,None,15,15,7,21)
 
plt.subplot(121),plt.imshow(Imagen)
plt.subplot(122),plt.imshow(Img2)
plt.show()

I_Gris=cv2.cvtColor(Img2, cv2.COLOR_BGR2GRAY)
print('Imagen en grises')
plt.imshow(I_Gris.astype('uint8'),vmin=0, vmax=255,cmap='gray') #Grafica la imagen en campo de grises
plt.show()
print('')

umbral = 85
binaria = np.uint8((I_Gris<umbral)*255)


ImgSobelX = cv2.Sobel(I_Gris,cv2.CV_8U,1,0, ksize= 3)
ImgSobelY = cv2.Sobel(I_Gris,cv2.CV_8U,0,1, ksize= 3)

laplacianImg = cv2.Laplacian(I_Gris,cv2.CV_8U)

Resultado_Canny = cv2.Canny(I_Gris,50,240)# 130 diferencia mínima, 140 es diferencia màxima

print('Resultado Binaria')
plt.imshow(binaria.astype('uint8'),vmin=0, vmax=255,cmap='gray') #Grafica la imagen en campo de grises
plt.show()
print('')


print('Resultado Sobel X')
plt.imshow(ImgSobelX.astype('uint8'),vmin=0, vmax=255,cmap='gray') #Grafica la imagen en campo de grises
plt.show()
print('')

print('Resultado Sobel Y')
plt.imshow(ImgSobelY.astype('uint8'),vmin=0, vmax=255,cmap='gray') #Grafica la imagen en campo de grises
plt.show()
print('')

print('Resultado Laplacian')
plt.imshow(laplacianImg.astype('uint8'),vmin=0, vmax=255,cmap='gray') #Grafica la imagen en campo de grises
plt.show()
print('')

print('Resultado Canny')
plt.imshow(Resultado_Canny.astype('uint8'),vmin=0, vmax=255,cmap='gray') #Grafica la imagen en campo de grises
plt.show()
print('')




