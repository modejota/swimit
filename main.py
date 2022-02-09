# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:02:20 2022

@author: modejota
"""

from math import gamma
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

def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    # Calculate grayscale histogram
    hist = cv.calcHist([image],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
 
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
 
    return cv.LUT(src, table)

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
    # Parece que se obtiene mejor resultado que si trabajamos sobre HSV, el problema es el chapoteo
    
    blur = cv.GaussianBlur(sframe,(5,5),0)
    
    # Intentar ajustar brillo y contraste
    auto_result = automatic_brightness_and_contrast(blur)
    cv.imshow('Intento de ecualizacion',auto_result[0])
    _, thre = cv.threshold(auto_result[0],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    """
    # Intentar equalizar histograma adaptativamente añade ruido extra al binarizar
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(blur)
    cv.imshow('Intento de ecualizacion',cl1)
    _, thre = cv.threshold(cl1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    """

    """
    # Realizar correcion gamma
    gamma_corrected = gammaCorrection(blur,0.4)
    cv.imshow('Correcion gamma',gamma_corrected)
    _, thre = cv.threshold(gamma_corrected,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    """
    # Es probable que si el nadador que es mas claro fuese aun mas claro
    # Y el fondo que es oscuro, fuese aun más oscuro, la detección fuese más clara

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

