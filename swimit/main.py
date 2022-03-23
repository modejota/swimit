# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:02:20 2022

@author: modejota   |   José Alberto Gómez García
"""

from tkinter import Tk, filedialog, simpledialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks

from auxiliary_functions import calculate_splits, countours_union
from constants.pool_constants import PoolValues
from constants.split_constants import SplitValues
from constants.swimmer_constants import SwimmerValues

Tk().withdraw()
try:
    vid = filedialog.askopenfilename(initialdir="./", title="Seleccione fichero", filetypes=[
        ("Video files", ".avi .mp4 .flv"),
        ("AVI file", ".avi"),
        ("MP4 file", ".mp4"),
        ("FLV file", ".flv"),
    ])
except(OSError, FileNotFoundError):
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
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_frame = 0
no_contour_detected = 0

# Valores en funcion de la resolucion
PV = PoolValues()
SWV = SwimmerValues()
SPV = SplitValues("freestyle", fps)
bottom = PV.LANES_Y.get(str(lane))
top = bottom + PV.LANE_HEIGHT

# Vectores para calculo de estadisticas
height_contour = np.full(frames, None)
x_coordinates = np.full(frames, None)

# Algoritmo GSOC (Google Summer of Code de 2017)
background_subtr_GSOC = cv2.bgsegm.createBackgroundSubtractorGSOC()

while video.isOpened():

    ok, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[bottom:top, :][..., 1]
    cv2.imshow('Banda Cr de YCrCb', frame)

    fg_gsoc = background_subtr_GSOC.apply(frame)

    contours = cv2.findContours(fg_gsoc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    valid_contours = []

    for c in contours:
        if SWV.SWIMMER_MIN_AREA <= cv2.contourArea(c) <= SWV.SWIMMER_MAX_AREA:
            [x, y, w, h] = cv2.boundingRect(c)
            # Discriminar corchera en base a su altura y obviar la zona del trampolin.
            if (h > PV.CORCHES_HEIGHT and x > PV.TRAMPOLIN_WIDTH and PV.MINIMUM_Y_ROI_LANE < y < PV.MAXIMUM_Y_ROI_LANE):
                valid_contours.append(c)

    # Contornos por area descendente, el nadador será el más grande o la unión de los dos más grandes.
    cntsSorted = sorted(valid_contours, key=lambda c: cv2.contourArea(c), reverse=True)

    # Si tengo mas de dos contornos, me quedare con los dos de mayor área.
    if len(cntsSorted) >= 2:
        [x, y, w, h] = cv2.boundingRect(cntsSorted[0])
        [x2, y2, w2, h2] = cv2.boundingRect(cntsSorted[1])

        # Compruebo si dichos contornos corresponden a tronco y piernas del nadador. 
        if ( abs(x - x2) < SWV.SWIMSUIT_MAX_WIDTH and 
             abs(y - y2) < SWV.SWIMMER_HEIGHT_DIFFERENCE and 
             w + w2 < SWV.SWIMMER_MAX_WIDTH):
            [x, y, w, h] = countours_union([x, y, w, h], [x2, y2, w2, h2])

        # Si el contorno está en el área de interés, guardamos sus posiciones para posterior análsis.
        if (PV.MINIMUM_Y_ROI_LANE < y < PV.MAXIMUM_Y_ROI_LANE):
            x_coordinates[actual_frame] = x
            height_contour[actual_frame] = h

        cv2.rectangle(fg_gsoc, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv2.drawContours(fg_gsoc, [c], 0, (255, 255, 255), -1)

    # Si tengo un único contorno tengo al nadador perfectamente reconocido, o por lo menos el tronco superior.
    elif len(cntsSorted) == 1:
        [x, y, w, h] = cv2.boundingRect(cntsSorted[0])
        cv2.rectangle(fg_gsoc, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv2.drawContours(fg_gsoc, [c], 0, (255, 255, 255), -1)

        # Si el contorno está en el área de interés, guardamos sus posiciones para posterior análsis.
        if (PV.MINIMUM_Y_ROI_LANE < y < PV.MAXIMUM_Y_ROI_LANE):
            x_coordinates[actual_frame] = x
            height_contour[actual_frame] = h

    else:
        no_contour_detected += 1

    cv2.imshow('Contornos tras procesamiento', fg_gsoc)

    actual_frame += 1

    key = cv2.waitKey(1) & 0xff
    # El bucle se pausa al pulsar P, y se reanuda al pulsar cualquier otra tecla
    if key == 112: cv2.waitKey(-1)
    # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
    if actual_frame == frames or key == 27: break

# Estadísticas y obtención de resultados
print('Frames sin contornos detectados: %d' % no_contour_detected)

# 1. Procesar coordenadas en las que no se detectó al nadador.
x_coordinates_detected = np.array([0 if i is None else i for i in x_coordinates], dtype=np.int32)
height_contour_detected = np.array([0.0 if i is None else float(i) for i in height_contour], dtype=np.float32)

# 2. Hallar índices de los frames en los que se cambia de sentido con respecto del vídeo original.
# (En el suavizado eliminamos los Nones, el find_peaks "puede proporcionar resultados imprevistos", si están)
sentido = [i for i in x_coordinates if i is not None]
sentidoS = savgol_filter(sentido, (2 * round(fps)) + 1, 1)
peaks, _ = find_peaks(sentidoS, distance=SPV.SPLIT_MIN_FRAMES, width=11)
peaks_I, _ = find_peaks(sentidoS * -1, distance=SPV.SPLIT_MIN_FRAMES, width=11)

magnitud = [sentidoS[peaks[i]] for i in range(0, len(peaks))]
plt.figure()
plt.plot(sentido)
plt.plot(np.arange(0, frames)[peaks], magnitud, marker='o')
"""
"""  # Graficas "no estrictamente necesarias", por ahora son para mostrar
magnitud_I = [sentidoS[peaks_I[i]] for i in range(0, len(peaks_I))]
plt.figure()
plt.plot(sentido)
plt.plot(np.arange(0, frames)[peaks_I], magnitud_I, marker='o')

indexes = []
first_index = 0
last_index = 0
peaks_coordinates_sentido = list(sentido[p] for p in peaks) + list(sentido[p] for p in peaks_I)
for i, x in enumerate(x_coordinates):
    if x in peaks_coordinates_sentido and not any(
            t in indexes for t in range(i - SPV.SPLIT_MIN_FRAMES // 2, i + SPV.SPLIT_MIN_FRAMES // 2)):
        indexes.append(i)
for i, x in enumerate(x_coordinates_detected):
    if x > PV.AFTER_JUMPING_X:
        first_index = i
        break
for i, x in reversed(list(enumerate(x_coordinates_detected))):
    if x > 0:
        last_index = i
        break
indexes = np.append(np.insert(np.sort(indexes), 0, first_index), last_index)

# En principio, tdo esto no lo necesito, para mostrar por ahora los graficos y aclararme
magnitud_G = [x_coordinates[indexes[i]] for i in range(0, len(indexes))]
plt.figure()
plt.plot(x_coordinates)
plt.plot(np.arange(0, frames)[indexes], magnitud_G, marker='o')

brazadas_min_total = 0.0
# 3. Cálculos en función del split.
for i in range(1, splits + 1):
    # 3.1- Extraer las coordenadas X y alturas.
    xs = x_coordinates_detected[indexes[i - 1]:indexes[i]]
    hs = height_contour_detected[indexes[i - 1]:indexes[i]]
    # 3.2. Establecer condiciones para saber frames extremo de la ROI. # (TO DO -> Ajustar parámetros basándome en más vídeos, a veces falla)
    # Idealmente, habria que considerar de forma distinta el primer split, ya que el salto recorre mas espacio y "falsea" el primer split
    if i % 2:  # Derecha a izquierda
        first_ind = min(element for element in xs if element >= PV.RIGHT_T_X_POSITION - 200 and element > 0)
        last_ind = min(element for element in xs if PV.LEFT_T_X_POSITION + 200 >= element > 0)
    else:  # Izquierda a derecha
        first_ind = min(element for element in xs if element >= PV.LEFT_T_X_POSITION + 200 and element > 0)
        last_ind = min(element for element in xs if element >= PV.RIGHT_T_X_POSITION - 200 and element > 0)

    hs_significativa = 1.3 * np.mean(hs)
    # 3.3. Hallar brazadas a partir de variaciones significativas en la altura de la caja que rodea al nadador
    hs_suavizado = savgol_filter(hs, 17, 2)
    peaks, _ = find_peaks(hs_suavizado, distance=SPV.THRESHOLD_BRAZADAS)
    true_peaks = [peak for peak in peaks if
                  hs_suavizado[peak] >= hs_significativa]  # Ajustar parametro, aunque parece funcionar bien así

    magnitud_vuelta = [hs_suavizado[true_peaks[i]] for i in range(len(true_peaks))]
    plt.figure()
    plt.plot(np.arange(0, frames)[indexes[i - 1]:indexes[i]], hs, label='H original')
    plt.plot(np.arange(0, frames)[indexes[i - 1]:indexes[i]], hs_suavizado, label='H suavizada')
    plt.plot(np.arange(0, frames)[indexes[i - 1]:indexes[i]][true_peaks], magnitud_vuelta, marker="o", ls="", ms=3,
             label='Brazada')
    plt.title('Variación de H en T25_01')
    plt.ylabel('H (píxeles)')
    plt.xlabel('Frame')
    plt.show()

    # 3.4- Calcular tiempo en ROI.
    time_ROI = abs(last_ind - first_ind) / fps
    # 3.5. Calcular brazadas por minuto en funcion de picos y tiempo en ROI
    brazadas_min = "{:.3f}".format((len(true_peaks) * 60) / time_ROI)
    print('T25: ' + str(brazadas_min) + ' brazadas/minuto')
    # 3.6. Acumular brazadas por minuto calculadas para hacer media entre splits.
    brazadas_min_total += float(brazadas_min)

# 4. Hacer media entre splits de las brazadas por minuto
brazadas_min_total /= splits
print('TTotal: ' + str("{:.3f}".format(brazadas_min_total)) + ' brazadas/minuto')

video.release()
cv2.destroyAllWindows()
