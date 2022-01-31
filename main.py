# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:02:20 2022

@author: modejota
"""

import cv2
from tkinter import Tk, filedialog

Tk().withdraw()
vid = filedialog.askopenfilename(initialdir = "./",title = "Seleccione fichero",filetypes = [
    ("AVI file", ".avi"),                    
    ("MP4 file", ".mp4"),
    ("FLV file", ".flv"),
])
video = cv2.VideoCapture(vid)

actual_frame = 0

# Estadísticas sobre el vídeo
FRAMES = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))             

while video.isOpened():

    ok, frame = video.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[...,1]     # Mostrar canal de saturación
    cv2.imshow('Saturation from [HSV]', frame)
    
    actual_frame = actual_frame + 1
    
    # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
    key = cv2.waitKey(1) & 0xff
    if actual_frame == FRAMES or key == 27:
        break
    
video.release()
cv2.destroyAllWindows()