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
    print(f'Número de calle inválido. Calle 3 por defecto.') # Valor provisional 
    lane = 3    # En el doc de Ángela dice que casi todas las calles 3 son buenas.

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

# Algoritmo GSOC (Google Summer of Code de 2017)
background_subtr_GSOC = cv2.bgsegm.createBackgroundSubtractorGSOC()

while video.isOpened():

    ok, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[top:bottom,:][...,1] 
    
    cv2.imshow('Banda Cr de YCrCb', frame)

    fg_gsoc = background_subtr_GSOC.apply(frame)
    cv2.imshow('Eliminacion de fondo con GSOC',fg_gsoc)

    contours = cv2.findContours(fg_gsoc.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    actual_frame = actual_frame + 1
    
    # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
    key = cv2.waitKey(1) & 0xff
    if actual_frame == FRAMES or key == 27: break
    
video.release()
cv2.destroyAllWindows()

