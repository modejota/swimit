# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:02:20 2022

@author: modejota
"""

import cv2
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

altura_corchera = 18    # Estos son mios
ancho_trampolin = 50
minimo_ROI_calle = 15
maximo_ROI_calle = 105
no_contour_detected = 0

threshold_brazadas = fps/2  # Freestyle -> Minimo 0.5" entre brazadas
# Variables globales que es muy probable utilice para estadísticas
width_contour = []
height_contour = []
x_coordinates = []
y_coordinates = []

# Algoritmo GSOC (Google Summer of Code de 2017)
background_subtr_GSOC = cv2.bgsegm.createBackgroundSubtractorGSOC()

while video.isOpened():

    ok, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[top:bottom,:][...,1] 

    cv2.imshow('Banda Cr de YCrCb', frame)

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
        if( abs(x-x2) < 20 and abs(y-y2) < 10): # Mandar parámetros fuera
            [x,y,w,h] = countours_union([x,y,w,h],[x2,2,w2,h2])
        cv2.rectangle(fg_gsoc, (x, y), (x + w, y + h), (255,255,255), 1)   
        cv2.drawContours(fg_gsoc, [c], 0, (255,255,255), -1)
    # Si tengo un único contorno tengo al nadador perfectamente reconocido.
    # Salvo quizás durante el salto al agua, que el gran chapoteo puede reconocerse
    elif len(cntsSorted) == 1:
        [x,y,w,h] = cv2.boundingRect(cntsSorted[0])
        cv2.rectangle(fg_gsoc, (x, y), (x + w, y + h), (255,255,255), 1)   
        cv2.drawContours(fg_gsoc, [c], 0, (255,255,255), -1)
    # Si no hay contornos no detecta nadador, obviamente
    else:
        no_contour_detected = no_contour_detected + 1

    cv2.imshow('Contornos tras procesamiento',fg_gsoc)

    actual_frame = actual_frame + 1
    
    key = cv2.waitKey(1) & 0xff
    # El bucle se pausa al pulsar P, y se reanuda al pulsar cualquier otra tecla
    if key == 112: cv2.waitKey(-1)
    # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
    if actual_frame == frames or key == 27: break
    
video.release()
cv2.destroyAllWindows()

