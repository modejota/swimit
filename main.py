# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:02:20 2022

@author: modejota
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.signal import find_peaks, savgol_filter
from tkinter import Tk, filedialog, simpledialog

def countours_union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

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

pattern = '_([\d]*)m_'
match = re.search(pattern,vid)
if match:
    distance = match.group(1)           # Conseguir distancia de la prueba  
    splits = int(distance) // 25        # Splits, para posterior procesamiento
else:
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

# A diferenciar en función de la resolución del vídeo
alto_calle = 120
lane_xy = {"1": 690,"2": 590,"3": 490,"4": 390,"5": 290}    # Completar resto de calles
top = lane_xy[str(lane)]
bottom = top + alto_calle

actual_frame = 0
area_min = 200      # Valores provisionales según Ángela
area_max = 2500
T_izda = 160
T_dcha = 1140

altura_corchera = 18    # Estos son mios
ancho_trampolin = 50
minimo_ROI_calle = 15
maximo_ROI_calle = 105
no_contour_detected = 0

threshold_brazadas = fps/2  # Freestyle -> Minimo 0.5" entre brazadas
# Variables globales para estadísticas
height_contour = [None] * frames
x_coordinates = [None] * frames
y_coordinates = [None] * frames

# Algoritmo GSOC (Google Summer of Code de 2017)
background_subtr_GSOC = cv2.bgsegm.createBackgroundSubtractorGSOC()

while video.isOpened():

    ok, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[top:bottom,:][...,1] 

    # cv2.imshow('Banda Cr de YCrCb', frame)

    fg_gsoc = background_subtr_GSOC.apply(frame)

    contours = cv2.findContours(fg_gsoc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    valid_contours = []

    for c in contours:
        if area_min <= cv2.contourArea(c) <= area_max:
            [x,y,w,h] = cv2.boundingRect(c)
            # Intentar discriminar corchera en base a su altura y obviar la zona del trampolin. Me guardo contornos validos
            if(h>altura_corchera and x>ancho_trampolin and y > minimo_ROI_calle and y < maximo_ROI_calle):            
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
        
        if (actual_frame <= fps*10 and y>15 and y<105 and x>2*T_izda) or (y>15 and y<105 and x>T_izda):
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

        if (actual_frame <= fps*10 and y>15 and y<105 and x>2*T_izda) or (y>15 and y<105 and x>T_izda):
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

sentido = [i for i in x_coordinates if i != None]   # Quedarme con los valores de los frames que detectaron algo
plt.plot(sentido)                                   # Representar sentido (ascendente, hacia derecha; descendente, hacia izquierda)
sentido_s = savgol_filter(sentido, (2*round(fps))+1, 1)   # Suavizar curva

# Meramente informativo para mí
plt.plot(sentido_s)
plt.title('Sentido del movimiento')
plt.xlabel('Frame del vídeo')
plt.ylabel('X del contorno, en píxeles')
plt.show() 


video.release()
cv2.destroyAllWindows()