# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:02:20 2022

@author: modejota
"""

import cv2 as cv
from tkinter import Tk, filedialog, simpledialog
import numpy as np

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
    print(f'Número de calle inválido. Calle 3 por defecto.') # Valor provisional 
    lane = 3    # En el doc de Ángela dice que casi todas las calles 3 son buenas.

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
backSub = cv.createBackgroundSubtractorKNN()

def otsu_opencv(frame):
    blur = cv.GaussianBlur(frame,(5,5),0)
    _, thre = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    kernel = np.ones((7,7),np.uint8)
    filtered = cv.morphologyEx(thre,cv.MORPH_CLOSE,kernel)
    return filtered

while video.isOpened():

    ok, frame = video.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)[top:bottom,:] 
    
    # En Cr funciona bastante mejor que en Cb.
    # Se distingue al nadador de manera mucho más clara y con menos ruido, pero el problema es el chapoteo
    sframe = frame[...,1]
    cv.imshow('Cr from [YCrCb]', sframe)

    # Modificando OTSU (y su morfologia) probablemente se pueda mejorar más fácil que en HSV
    process = otsu_opencv(sframe)
    processed = backSub.apply(process)
    cv.imshow('BACKGROUND TEST', processed)

    actual_frame = actual_frame + 1
    
    # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
    key = cv.waitKey(1) & 0xff
    if actual_frame == FRAMES or key == 27: break
    
video.release()
cv.destroyAllWindows()

