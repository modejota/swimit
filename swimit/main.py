# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:02:20 2022

@author: modejota   |   José Alberto Gómez García
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog, simpledialog
from constants.swimmer_constants import SwimmerValues
from constants.pool_constants import PoolValues
from auxiliary_functions import calculate_splits, countours_union

Tk().withdraw()
try:
    vid = filedialog.askopenfilename(initialdir = "./",title = "Seleccione fichero",filetypes = [
        ("Video files", ".avi .mp4 .flv"),
        ("AVI file", ".avi"),                    
        ("MP4 file", ".mp4"),
        ("FLV file", ".flv"),
    ])
except(OSError,FileNotFoundError):
    print(f'No se ha podido abrir el fichero seleccionado.')
except Exception as error:
    print(f'Ha ocurrido un error: <{error}>')

if len(vid) == 0 or vid is None:
    print(f'No se ha seleccionado ningún archivo.')
    quit()
video = cv2.VideoCapture(vid)

# Pendiente reconocer el tipo de prueba
splits = calculate_splits(vid)
if splits == 0:
    print("No se especifica la longitud de la prueba en el nombre del fichero.")
    quit()

try:
    lane = simpledialog.askinteger(title="Calle", prompt="Calle (1-8):", minvalue=1, maxvalue=8)
except ValueError:
    print("Se ha introducido un valor incorrecto.")
    quit()

if lane is None:
    print("No se ha seleccionado calle a analizar.")
    quit()

# Estadísticas sobre el vídeo
frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_frame = 0
no_contour_detected = 0

# Valores en funcion de la resolucion
PV = PoolValues(original_width,original_height)
SV = SwimmerValues(original_width,original_height)
bottom = PV.LANES_Y.get(str(lane))
top = bottom + PV.LANE_HEIGHT

# Vectores para calculo de estadisticas
height_contour = np.full(frames,None)
x_coordinates = np.full(frames,None)

# Algoritmo GSOC (Google Summer of Code de 2017)
background_subtr_GSOC = cv2.bgsegm.createBackgroundSubtractorGSOC()

while video.isOpened():

    ok, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[bottom:top,:][...,1] 
    cv2.imshow('Banda Cr de YCrCb', frame)

    fg_gsoc = background_subtr_GSOC.apply(frame)

    contours = cv2.findContours(fg_gsoc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    valid_contours = []

    for c in contours:
        if SV.SWIMMER_MIN_AREA <= cv2.contourArea(c) <= SV.SWIMMER_MAX_AREA:
            [x,y,w,h] = cv2.boundingRect(c)
            # Discriminar corchera en base a su altura y obviar la zona del trampolin.
            if(h>PV.CORCHES_HEIGHT and x>PV.TRAMPOLIN_WIDTH and y > PV.MINIMUM_Y_ROI_LANE and y < PV.MAXIMUM_Y_ROI_LANE):            
                valid_contours.append(c)

    # Contornos por area descendente, el nadador será el más grande o la unión de los dos más grandes.
    cntsSorted = sorted(valid_contours, key=lambda c: cv2.contourArea(c), reverse=True)
    
    # Si tengo mas de dos contornos, me quedare con los dos de mayor área.
    if len(cntsSorted) >= 2:
        [x,y,w,h] = cv2.boundingRect(cntsSorted[0])
        [x2,y2,w2,h2] = cv2.boundingRect(cntsSorted[1])
        
        # Compruebo si dichos contornos corresponden a tronco y piernas del nadador. 
        if( abs(x-x2) < SV.SWIMSUIT_MAX_WIDTH and abs(y-y2) < SV.SWIMMER_HEIGHT_DIFFERENCE and w+w2 < SV.SWIMMER_MAX_WIDTH): 
            [x,y,w,h] = countours_union([x,y,w,h],[x2,y2,w2,h2])
        
        # Si el contorno está en el área de interés, guardamos sus posiciones para posterior análsis.
        if ((y>PV.MINIMUM_Y_ROI_LANE and y<PV.MAXIMUM_Y_ROI_LANE) 
                or (y>PV.MINIMUM_Y_ROI_LANE and y<PV.MAXIMUM_Y_ROI_LANE and x>PV.LEFT_T_X_POSITION)):
            x_coordinates[actual_frame] = x
            height_contour[actual_frame] = h
        
        cv2.rectangle(fg_gsoc, (x, y), (x + w, y + h), (255,255,255), 1)   
        cv2.drawContours(fg_gsoc, [c], 0, (255,255,255), -1)
    
    # Si tengo un único contorno tengo al nadador perfectamente reconocido, o por lo menos el tronco superior.
    elif len(cntsSorted) == 1:
        [x,y,w,h] = cv2.boundingRect(cntsSorted[0])
        cv2.rectangle(fg_gsoc, (x, y), (x + w, y + h), (255,255,255), 1)   
        cv2.drawContours(fg_gsoc, [c], 0, (255,255,255), -1)
       
        # Si el contorno está en el área de interés, guardamos sus posiciones para posterior análsis.
        if ((y>PV.MINIMUM_Y_ROI_LANE and y<PV.MAXIMUM_Y_ROI_LANE) 
                or (y>PV.MINIMUM_Y_ROI_LANE and y<PV.MAXIMUM_Y_ROI_LANE and x>PV.LEFT_T_X_POSITION)):
            x_coordinates[actual_frame] = x
            height_contour[actual_frame] = h

    else:
        no_contour_detected+=1

    cv2.imshow('Contornos tras procesamiento',fg_gsoc)

    actual_frame+=1
    
    key = cv2.waitKey(1) & 0xff
    # El bucle se pausa al pulsar P, y se reanuda al pulsar cualquier otra tecla
    if key == 112: cv2.waitKey(-1)
    # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
    if actual_frame == frames or key == 27: break

# Estadísticas y obtención de resultados
print('Frames sin contornos detectados: %d' % no_contour_detected)

video.release()
cv2.destroyAllWindows()