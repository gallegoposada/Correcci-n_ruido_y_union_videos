from inspect import stack
from turtle import color
import cv2 # OpenCV para computer vision
import numpy as np # Para cálculo de matrices
import matplotlib.pyplot as plt #Para graficar
import os #Habilita el manejo de directorios
from math import sqrt
import skvideo.io



Video_1 = []
Video_2 = []
Video_3 = []

Video_unido = []
video_final = []

Ruta = r'C:\Users\user\Desktop\Procesamiento de Imagenes\Parcial_2\rhinos_1.avi'
Video1 = cv2.VideoCapture(Ruta)

Frames_number=0

while(Video1.isOpened()):
    ret, frame = Video1.read ()
    if ret == True:
        Frames_number=Frames_number+1
        Color=frame[:,:,[2,1,0]]
        [Fl_1,Cl_1,Ch_1]=Color.shape

        Video_1.append(Color)
    
    else:
        break
Video1.release()

print("# Frames video 1: ", Frames_number)

Ruta = r'C:\Users\user\Desktop\Procesamiento de Imagenes\Parcial_2\rhinos_2.avi'
Video2 = cv2.VideoCapture(Ruta)

Frames_number=0

while(Video2.isOpened()):
    ret, frame = Video2.read ()
    if ret == True:
        Frames_number=Frames_number+1
        Color=frame[:,:,[2,1,0]]
        [Fl_2,Cl_2,Ch_2]=Color.shape

        Video_2.append(Color)
    
    else:
        break
Video2.release()

print("# Frames video 2: ", len(Video_2))

Ruta = r'C:\Users\user\Desktop\Procesamiento de Imagenes\Parcial_2\rhinos_3.avi'
Video3 = cv2.VideoCapture(Ruta)

Frames_number=0

while(Video3.isOpened()):
    ret, frame = Video3.read ()
    if ret == True:
        Frames_number=Frames_number+1
        Color=frame[:,:,[2,1,0]]
        [Fl_3,Cl_3,Ch_3]=Color.shape

        Video_3.append(Color)
    
    else:
        break
Video3.release()

print("# Frames video 3: ", len(Video_3))

#print(Fl_1,Fl_2, Fl_3)
#print(Cl_1,Cl_2, Cl_3)


Video_unido = Video_1 + Video_2 + Video_3

print(len(Video_unido))

video_resultado = cv2.VideoWriter('Video.avi', cv2.VideoWriter_fourcc(*'DIVX'),33,(Cl_1,Fl_1))

Ruta = r'C:\Users\user\Desktop\Procesamiento de Imagenes\Parcial_2\rhinos_1.avi'
Video1 = cv2.VideoCapture(Ruta)
Ruta = r'C:\Users\user\Desktop\Procesamiento de Imagenes\Parcial_2\rhinos_2.avi'
Video2 = cv2.VideoCapture(Ruta)
Ruta = r'C:\Users\user\Desktop\Procesamiento de Imagenes\Parcial_2\rhinos_3.avi'
Video3 = cv2.VideoCapture(Ruta)

while(Video1.isOpened() and Video2.isOpened() and Video3.isOpened()):
    for i in range (0, len(Video_unido), 1):
        if i < 100:
            I_Gris=cv2.cvtColor(Video_unido[i], cv2.COLOR_BGR2GRAY)
            video_final[i]= video_final.append(I_Gris)
        
        if i > 170 and i < 250:
            Img = cv2.fastNlMeansDenoisingColored(Video_unido[i],None,10,10,7,21)
            video_final[i] = video_final.append(Img)
        

        
        if i >280 and i < 320:
            Img_2=cv2.cvtColor(Video_unido[i], cv2.COLOR_BGR2GRAY)
            Canny = cv2.Canny(Img_2,50,240)
            video_final[i] = video_final.append(Canny)

        else:
            video_final[i] = video_final.append(Video_unido[i])
    

        video_resultado.write(video_final[i])


video_resultado.release()
Video1.release()
Video2.release()
Video3.release()
