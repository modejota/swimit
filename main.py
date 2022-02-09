# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:02:20 2022

@author: modejota
"""

import cv2 as cv
from tkinter import Tk, filedialog, simpledialog

# De aquí, eliminar las que no se necesiten
import numpy as np
from skimage.transform import (hough_line, hough_line_peaks, hough_circle,
hough_circle_peaks)
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.data import astronaut
from skimage.io import imread, imsave
from skimage.color import rgb2gray, gray2rgb, label2rgb
from skimage import img_as_float
from skimage.morphology import skeletonize
from skimage import data, img_as_float
import matplotlib.pyplot as pylab
from matplotlib import cm
from skimage.filters import sobel, threshold_otsu
from skimage.feature import canny
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries, find_boundaries

Tk().withdraw()
try:
    vid = filedialog.askopenfilename(initialdir = "./",title = "Seleccione fichero",filetypes = [
        ("AVI file", ".avi"),                    
        ("MP4 file", ".mp4"),
        ("FLV file", ".flv"),
    ])
except(OSError,FileNotFoundError):
    print(f'No se ha podido abrir el fichero seleccionado.')
except Exception as error:
    print(f'Ha ocurrido un error: <{error}>')

if vid is not None:
    video = cv.VideoCapture(vid)

try:
    lane = simpledialog.askinteger(title="Calle", prompt="Calle (1-8):", minvalue=1, maxvalue=8)
except:
    print(f'Número de calle inválido.')

# Estadísticas sobre el vídeo
FRAMES = int(video.get(cv.CAP_PROP_FRAME_COUNT))
fps = video.get(cv.CAP_PROP_FPS)
original_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
original_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))             

# A diferenciar en función de la resolución del vídeo
alto_calle = 120
lane_xy = {"1": 690,"2": 590,"3": 490,"4": 390,"5": 290}    # Completar resto de calles
top = lane_xy[str(lane)]
bottom = top + alto_calle

actual_frame = 0

while video.isOpened():

    ok, frame = video.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb) 

    sframe = frame[...,1][top:bottom,:]
    cv.imshow('Cr from [YCrCb]', sframe)

    # BINARIZACION CON OTSU: nadador en blanco.
    # Se emborrona un poco antes para intentar mejorar el resultado.
    # Aplicando clausura parece mejorar un poco el resultado. (Kernel hasta 7)
    # Problemas con el lanzamiento al agua y un poquito de chapoteo en algunas ocasiones.
    # Parece que se obtiene mejor resultado que si trabajamos sobre HSV
    
    blur = cv.GaussianBlur(sframe,(5,5),0)
    _, thre = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    kernel = np.ones((7,7),np.uint8)
    th2 = cv.morphologyEx(thre,cv.MORPH_CLOSE,kernel)
    cv.imshow('Otsu clausura',th2)

    actual_frame = actual_frame + 1
    
    # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
    key = cv.waitKey(1) & 0xff
    if actual_frame == FRAMES or key == 27:
        break
    
video.release()
cv.destroyAllWindows()