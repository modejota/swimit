# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:02:20 2022

@author: modejota   |   José Alberto Gómez García
"""

import cv2
import numpy as np
from tkinter import Tk, filedialog, simpledialog
from auxiliary_functions import calculate_splits, countours_union
from constants.analisis_constants import AnalisisValues

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
    video = cv2.VideoCapture(vid)

splits = calculate_splits(vid)
if splits == 0:
    raise ValueError("No se especifica la longitud de la prueba en el nombre del fichero.")

try:
    lane = simpledialog.askinteger(title="Calle", prompt="Calle (1-8):", minvalue=1, maxvalue=8)
except:
    print(f'Número de calle inválido. Calle 3 por defecto.') # Valor provisional 
finally:    
    lane = 3    # En el doc de Ángela dice que casi todas las calles 3 son buenas.

# Estadísticas sobre el vídeo
frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_frame = 0
no_contour_detected = 0

# Valores en funcion de la resolucion
AV = AnalisisValues(original_width,original_height)
bottom = AV.LANES_Y.get(str(lane))
top = bottom + AV.LANE_HEIGHT

threshold_brazadas = fps/2  # Freestyle -> Minimo 0.5" entre brazadas

# Vectores para calculo de estadisticas
height_contour = np.full(frames,None)
x_coordinates = np.full(frames,None)
y_coordinates = np.full(frames,None)

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
        if AV.SWIMMER_MIN_AREA <= cv2.contourArea(c) <= AV.SWIMMER_MAX_AREA:
            [x,y,w,h] = cv2.boundingRect(c)
            # Intentar discriminar corchera en base a su altura y obviar la zona del trampolin. Me guardo contornos validos
            if(h>AV.CORCHES_HEIGHT and x>AV.TRAMPOLIN_WIDTH and y > AV.MINIMUM_Y_ROI_LANE and y < AV.MAXIMUM_Y_ROI_LANE):            
                valid_contours.append(c)

    # Ordeno los contornos por area descendente, el nadador será el más grande o la unión de los dos más grandes
    cntsSorted = sorted(valid_contours, key=lambda c: cv2.contourArea(c), reverse=True)
    
    # Si tengo mas de dos contornos, me quedare con los dos de mayor área
    if len(cntsSorted) >= 2:
        [x,y,w,h] = cv2.boundingRect(cntsSorted[0])
        [x2,y2,w2,h2] = cv2.boundingRect(cntsSorted[1])
        # Compruebo si están muy cerca para unirlos, porque en función del color del bañador, a veces se separa tronco y piernas
        
        # Mandar parámetros fuera y ajustar longitud maxima. Probablemente sea mejor ajustar en funcion de diferencia de altura
        if( abs(x-x2) < 20 and abs(y-y2) < 10 and w+w2 < 120): 
            [x,y,w,h] = countours_union([x,y,w,h],[x2,2,w2,h2])
        
        if (y>AV.MINIMUM_Y_ROI_LANE and y<AV.MAXIMUM_Y_ROI_LANE and x>AV.AFTER_JUMPING_X) or (y>AV.MINIMUM_Y_ROI_LANE and y<AV.MAXIMUM_Y_ROI_LANE and x>AV.LEFT_T_X_POSITION):
            x_coordinates[actual_frame] = x
            y_coordinates[actual_frame] = y
            height_contour[actual_frame] = h
        
        cv2.rectangle(fg_gsoc, (x, y), (x + w, y + h), (255,255,255), 1)   
        cv2.drawContours(fg_gsoc, [c], 0, (255,255,255), -1)
    
    # Si tengo un único contorno tengo al nadador perfectamente reconocido, o por lo menos el tronco superior.
    # Salvo quizás durante el salto al agua, que el gran chapoteo puede reconocerse
    elif len(cntsSorted) == 1:
        [x,y,w,h] = cv2.boundingRect(cntsSorted[0])
        cv2.rectangle(fg_gsoc, (x, y), (x + w, y + h), (255,255,255), 1)   
        cv2.drawContours(fg_gsoc, [c], 0, (255,255,255), -1)

        if (y>AV.MINIMUM_Y_ROI_LANE and y<AV.MAXIMUM_Y_ROI_LANE and x>AV.AFTER_JUMPING_X) or (y>AV.MINIMUM_Y_ROI_LANE and y<AV.MAXIMUM_Y_ROI_LANE and x>AV.LEFT_T_X_POSITION):
            x_coordinates[actual_frame] = x
            y_coordinates[actual_frame] = y
            height_contour[actual_frame] = h

    # Si no hay contornos no detecta nadador, ni nada, obviamente
    else:
        no_contour_detected = no_contour_detected + 1

    cv2.imshow('Contornos tras procesamiento',fg_gsoc)

    actual_frame = actual_frame + 1
    
    key = cv2.waitKey(1) & 0xff
    # El bucle se pausa al pulsar P, y se reanuda al pulsar cualquier otra tecla
    if key == 112: cv2.waitKey(-1)
    # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
    if actual_frame == frames or key == 27: break


# Estadísticas y obtención de resultados
print('Frames sin contornos detectados: %d' % no_contour_detected)

# sentido_nado(x_coordinates,fps)


video.release()
cv2.destroyAllWindows()