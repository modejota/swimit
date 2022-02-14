# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:02:20 2022

@author: modejota
"""

import cv2 as cv
from tkinter import Tk, filedialog, simpledialog
import numpy as np
# import pybgs as bgs 
# # Paquete wrapper de una pechá de algoritmos de eliminación de fondos. No consigo instalarlo

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
background_subtr_KNN = cv.createBackgroundSubtractorKNN(detectShadows=False)
# El utilizar o no la detección de sombras no parece influir en la detección del nadador.
# No utilizarla proporciona una mejora en rendimiento

# MOG2 me proporciona mas ruido de primeras, por lo que KNN parece mas apropiado.


# Algoritmo GSOC (Google Summer of Code de 2017)
# Hay más "eliminadores de fondos" en "subpaquetes" de OpenCV, pero segun papers y el propio OpenCV este es el que mejor resultado proporciona
background_subtr_GSOC = cv.bgsegm.createBackgroundSubtractorGSOC()
background_subtr_CNT = cv.bgsegm.createBackgroundSubtractorCNT()

# Del paquete que soy incapaz de instalar. No parece proporcionar mejores resultados para nuestra casuística segun papers
# background_subtr_GSOC_subsense = bgs.SuBSENSE()

def aplicar_morfologia(frame):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,3))
    processing = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel,iterations = 1)
    processed = cv.morphologyEx(processing, cv.MORPH_OPEN, kernel,iterations = 1)
    return processed

# Otra alternativa, confia en que haya grises para intentar obtener mejorar resultado (necesita detectShadows =True)
# Pero no encuentro diferencia significativa respecto de GSOC
def knn_post_processed(frame):
    fg = background_subtr_KNN.apply(frame)
    fg_blur = cv.GaussianBlur(fg,(7,3),0)
    _, fg_th = cv.threshold(fg_blur,15,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    fg_M = aplicar_morfologia(fg_th)
    return fg_M


while video.isOpened():

    ok, frame = video.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)[top:bottom,:] 
    
    # En Cr funciona bastante mejor que en Cb.
    # Se distingue al nadador de manera mucho más clara y con menos ruido, pero el problema es el chapoteo
    sframe = frame[...,1]
    cv.imshow('Cr from [YCrCb]', sframe)
    

    """
    # Aplicar morfologia antes de eliminar fondo (imagen escala de grises)
    tras_morfologia = aplicar_morfologia(sframe)
    removed_bg1 = background_subtr_KNN.apply(tras_morfologia)
    cv.imshow('MORFOLOGIA ANTES DE ELIMINAR FONDO', removed_bg1)
    """
    # La documentacion de OpenCV recomienda procesar despues. Ángela lo hacía antes. 
    # La diferencia no parece demasiado significativa, pero si se hace antes hay como un poco más de grano.
    """
    # Aplicar morfologia despue de eliminar fondo
    removed_bg2 = background_subtr_KNN.apply(sframe)
    processed = aplicar_morfologia(removed_bg2)
    cv.imshow('MORFOLOGIA DESPUES DE ELIMINAR FONDO', processed)
    """
    
    # Sin aplicar procesamiento ninguno, es el que mejor mantiene brazos.
    # Parece que tolera algo más el chapoteo, pero habria que afinarlo más. (Posible suaviado antes o cambiar structuringElement)
    fg_gsoc = background_subtr_GSOC.apply(sframe)
    # gsoc_post_processed = aplicar_morfologia(fg_gsoc)
    cv.imshow('FOREGROUND MASK GSOC',fg_gsoc)

    # La otra manera con KNN con emborronamiento y thresholding, no parece haber casi diferencia
    # cv.imshow('KNN with blur',knn_post_processed(sframe))


    # Sin aplicar morfología parece que es el que menos ruido introduce por chapoteo. Puede que sea mas facil afirnalo con morfologia.
    fg_cnt = background_subtr_CNT.apply(sframe)
    #cnt_post_processed = aplicar_morfologia(fg_cnt)
    cv.imshow('FOREGROUND MASK CNT',fg_cnt)


    # Otro método mas. (No consigo instalar paquete)
    #foreground_mask_subsense = background_subtr_GSOC_subsense.apply(frame)
    #subsense_post_processed = aplicar_morfologia(foreground_mask_subsense)
    #cv.imshow('FOREGROUND MASK SUBSENSE', subsense_post_processed)

    actual_frame = actual_frame + 1
    
    # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
    key = cv.waitKey(1) & 0xff
    if actual_frame == FRAMES or key == 27: break
    
video.release()
cv.destroyAllWindows()
