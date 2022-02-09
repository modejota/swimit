# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:02:20 2022

@author: modejota
"""

import cv2
from tkinter import Tk, filedialog, simpledialog

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
    print(f'Número de calle inválido.')

# Estadísticas sobre el vídeo
FRAMES = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))             

# A diferenciar en función de la resolución del vídeo
alto_calle = 120
lane_xy = {"1": 690,"2": 590,"3": 490,"4": 390,"5": 290}    # Completar resto de calles
top = lane_xy[str(lane)]
bottom = top + alto_calle

actual_frame = 0

while video.isOpened():

    ok, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    hframe = frame[...,0]   # Mostrar canal de tono
    hframe = hframe[top:bottom,:]
    cv2.imshow('Hue from [HSV]', hframe)

    sframe = frame[...,1]     # Mostrar canal de saturación
    sframe = sframe[top:bottom,:]
    cv2.imshow('Saturation from [HSV]', sframe)

    vframe = frame[...,2]    # Mostrar canal de valor/brillo
    vframe = vframe[top:bottom,:]
    cv2.imshow('Value from [HSV]', vframe)

    actual_frame = actual_frame + 1
    
    # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
    key = cv2.waitKey(1) & 0xff
    if actual_frame == FRAMES or key == 27:
        break
    
video.release()
cv2.destroyAllWindows()