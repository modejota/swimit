# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 18:02:20 2022

@author: modejota   |   José Alberto Gómez García
"""
import math
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog, simpledialog
from scipy.signal import savgol_filter, find_peaks
from auxiliary_functions import calculate_splits, countours_union
from constants.pool_constants import PoolValues
from constants.split_constants import SplitValues
from constants.swimmer_constants import SwimmerValues
from constants.resolution_constants import ResolutionValues

Tk().withdraw()
vid = None
try:
    vid = filedialog.askopenfilename(initialdir="../sample_videos", title="Seleccione fichero", filetypes=[
        ("Video files", ".avi .mp4 .flv"),
        ("AVI file", ".avi"),
        ("MP4 file", ".mp4"),
        ("FLV file", ".flv"),
    ])
except(OSError, FileNotFoundError):
    sys.exit(f'No se ha podido abrir el fichero seleccionado.')
except Exception as error:
    sys.exit(f'Ha ocurrido un error: <{error}>')
if len(vid) == 0 or vid is None:
    sys.exit(f'No se ha seleccionado ningún archivo.')

video = cv2.VideoCapture(vid)

splits = calculate_splits(vid)
if splits == 0:
    sys.exit(f'No se especifica la longitud de la prueba en el nombre del fichero.')

lane = simpledialog.askinteger(title="Calle", prompt="Calle (1-8):", minvalue=1, maxvalue=8)
if lane is None:
    sys.exit(f'No se ha seleccionado calle a analizar.')
# Pendiente reconocer el tipo de prueba, puede que ni llegue a hacer falta.

# Estadísticas sobre el vídeo
frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_processed = 0
frames_read = 0
frames_with_no_contour = 0

# Valores provenientes de las dataclasses
PV = PoolValues()
SWV = SwimmerValues()
SPV = SplitValues("freestyle", fps)  # Testeando butterfly en verdad, temporal

need_to_resize = False
total_time = 0.0
fps_factor = 1
if width != ResolutionValues.HALF_WIDTH or height != ResolutionValues.HALF_HEIGHT:
    need_to_resize = True
if fps >= ResolutionValues.NORMAL_FRAME_RATE:
    fps_factor = math.ceil(fps / ResolutionValues.NORMAL_FRAME_RATE)
    frames = int(frames // fps_factor)

# Coordenadas Y de la calle a analizar
bottom = PV.LANES_Y.get(str(lane))
top = bottom + PV.LANE_HEIGHT

# Vectores para calculo de estadisticas
height_contour = np.full(frames, np.NaN)
x_coordinates = np.full(frames, np.NaN)

# Algoritmo GSOC (Google Summer of Code de 2017)
background_subtr_GSOC = cv2.bgsegm.createBackgroundSubtractorGSOC()

start_time = time.time()
# Lectura de frames. Eliminación de fondo y detección de contornos
while video.isOpened():

    ok, frame = video.read()
    if ok:
        # Reducir el framerate antes de procesar nada mas, para reducir tiempo de ejecucion y conservar memoria
        if frames_read % fps_factor != 0:
            frames_read += 1
        else:
            if need_to_resize:
                frame = cv2.resize(frame, (ResolutionValues.HALF_WIDTH, ResolutionValues.HALF_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[bottom:top, :][..., 1]
            # cv2.imshow('Frame original', frame)

            fg_gsoc = background_subtr_GSOC.apply(frame)

            contours = cv2.findContours(fg_gsoc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            valid_contours = []

            for c in contours:
                if SWV.SWIMMER_MIN_AREA <= cv2.contourArea(c) <= SWV.SWIMMER_MAX_AREA:
                    [x, y, w, h] = cv2.boundingRect(c)
                    # Discriminar corchera en base a su altura y obviar la zona del trampolin.
                    if h > PV.CORCHES_HEIGHT and x > PV.TRAMPOLIN_WIDTH and \
                            PV.MINIMUM_Y_ROI_LANE < y < PV.MAXIMUM_Y_ROI_LANE:
                        valid_contours.append(c)

            # Contornos por area descendente, el nadador será el más grande o la unión de los dos más grandes.
            cntsSorted = sorted(valid_contours, key=lambda contour: cv2.contourArea(contour), reverse=True)

            # Si tengo mas de dos contornos, me quedare con los dos de mayor área.
            if len(cntsSorted) >= 2:
                [x, y, w, h] = cv2.boundingRect(cntsSorted[0])
                [x2, y2, w2, h2] = cv2.boundingRect(cntsSorted[1])

                # Compruebo si dichos contornos corresponden a tronco y piernas del nadador.
                if (abs(x - x2) < SWV.SWIMSUIT_MAX_WIDTH and
                        abs(y - y2) < SWV.SWIMMER_HEIGHT_DIFFERENCE and
                        w + w2 < SWV.SWIMMER_MAX_WIDTH):
                    [x, y, w, h] = countours_union([x, y, w, h], [x2, y2, w2, h2])

                # Si el contorno está en el área de interés, guardamos sus posiciones para posterior análsis.
                if PV.MINIMUM_Y_ROI_LANE < y < PV.MAXIMUM_Y_ROI_LANE:
                    x_coordinates[frames_processed] = x
                    height_contour[frames_processed] = h

                cv2.rectangle(fg_gsoc, (x, y), (x + w, y + h), (255, 255, 255), 1)

            # Si tengo un único contorno tengo al nadador perfectamente reconocido, o por lo menos el tronco superior.
            elif len(cntsSorted) == 1:
                [x, y, w, h] = cv2.boundingRect(cntsSorted[0])
                cv2.rectangle(fg_gsoc, (x, y), (x + w, y + h), (255, 255, 255), 1)

                # Si el contorno está en el área de interés, guardamos sus posiciones para posterior análsis.
                if PV.MINIMUM_Y_ROI_LANE < y < PV.MAXIMUM_Y_ROI_LANE:
                    x_coordinates[frames_processed] = x
                    height_contour[frames_processed] = h

            else:
                frames_with_no_contour += 1

            # cv2.imshow('Contornos tras procesamiento', fg_gsoc)
            frames_processed += 1
            frames_read += 1
        print(f'\rProcesando frame {frames_read} de {frames}.', end='')
        key = cv2.waitKey(1) & 0xff
        # El bucle se pausa al pulsar P, y se reanuda al pulsar cualquier otra tecla
        if key == 112:
            cv2.waitKey(-1)
        # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
        if frames_read == frames or key == 27:
            break

# Estadísticas y obtención de resultados
print('\nFrames sin contornos detectados: %d' % frames_with_no_contour)
print('Tiempo total de ejecución: %.2f segundos\n' % (time.time() - start_time))

# Análisis de los resultados

# 1. Procesar coordenadas en las que no se detectó al nadador.
for i in range(frames):
    if np.isnan(x_coordinates[i]):
        if i > 0:
            x_coordinates[i] = x_coordinates[i - 1]
        else:
            x_coordinates[i] = 0
    if np.isnan(height_contour[i]):
        if i > 0:
            height_contour[i] = height_contour[i - 1]
        else:
            height_contour[i] = 0.0

# 2. Hallar índices y Xs de los frames en los que se cambia de sentido con respecto del vídeo original.
x_coordinates_smooth = savgol_filter(x_coordinates, (2 * round(fps)) + 1, 1)
peaks = list(np.concatenate((
    find_peaks(x_coordinates_smooth, distance=SPV.SPLIT_MIN_FRAMES, width=11)[0],
    find_peaks(x_coordinates_smooth * -1, distance=SPV.SPLIT_MIN_FRAMES, width=11)[0]
)))
x_coordinates_of_peaks = [x_coordinates[p] for p in peaks]

# El suavizado del final del vídeo puede crear un falso pico. (Hecho para detectar seguidos, aunque no me ha pasado)
if len(x_coordinates_of_peaks) > splits:
    for i in range(len(x_coordinates_of_peaks) - splits):
        if abs(x_coordinates_of_peaks[-1] - x_coordinates_of_peaks[-2]) < 20:
            if abs(x_coordinates_of_peaks[-1]) > abs(x_coordinates_of_peaks[-2]):
                x_coordinates_of_peaks.pop()
                peaks.pop()
            else:
                x_coordinates_of_peaks.pop(-2)
                peaks.pop(-2)

#  El nadador no tiene porqué completar el split, hay grabaciones en las que se corta.
if len(x_coordinates_of_peaks) != splits:
    iterations = len(x_coordinates_of_peaks)+1
else:
    iterations = splits+1
valid_splits = iterations-1

indexes = []
first_index = np.where(x_coordinates > PV.AFTER_JUMPING_X)[0][0]
indexes.append(first_index)
for i, x in enumerate(x_coordinates):
    if x in x_coordinates_of_peaks and \
            not any(t in indexes for t in range(i - SPV.SPLIT_MIN_FRAMES // 2, i + SPV.SPLIT_MIN_FRAMES // 2)):
        indexes.append(i)
last_index = np.where(x_coordinates > 0)[0][-1]
indexes = np.append(np.sort(indexes), last_index)

brazadas_min_total = 0.0
x_axis = np.arange(0, frames)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_axis, x_coordinates, 'b', label="Coordenadas X")
ax.plot(x_axis, x_coordinates_smooth, 'r', label="Coordenadas X suavizadas")
ax.plot(x_axis[peaks], x_coordinates_smooth[peaks], 'ro', markersize=5, label='Cambio de sentido')
ax.set_xlabel('Frame')
ax.set_ylabel('Coordenadas X')
ax.set_title('Sentido del movimiento')
ax.legend(loc='upper right')
ax.grid(True)
plt.show()

# 3. Cálculos en función del split.
for i in range(1, iterations):
    # 3.1- Extraer las coordenadas X y alturas del split en cuestión.
    try:
        xs = x_coordinates[indexes[i - 1]:indexes[i]]
        hs = height_contour[indexes[i - 1]:indexes[i]]
        # 3.2 Esteblecer los frames extremos de la región de interés.
        if i == 1:  # Primer split (izquierda a derecha)
            first_index = xs[np.where(xs >= PV.AFTER_JUMPING_X)[0][0]]
            last_index = xs[np.where(xs <= PV.RIGHT_T_X_POSITION)[0][-1]]
        elif i % 2 == 0:  # Splits pares (derecha a izquierda)
            first_index = xs[np.where(xs <= PV.RIGHT_T_X_POSITION)[0][0]]
            last_index = xs[np.where(xs >= PV.LEFT_T_X_POSITION)[0][-1]]
        else:  # Splits impares (izquierda a derecha)
            first_index = xs[np.where(xs >= PV.LEFT_T_X_POSITION)[0][0]]
            last_index = xs[np.where(xs <= PV.RIGHT_T_X_POSITION)[0][-1]]

        # 3.3- Hallar brazadas a partir de las variaciones significativas de las alturas.
        hs_significativa = np.mean(hs)
        # La clave está en este este suavizado de la curva de alturas.
        hs_suavizado = savgol_filter(hs, 55, 2)
        peaks, _ = find_peaks(hs_suavizado, distance=SPV.THRESHOLD_BRAZADAS, width=9)
        true_peaks = [peak for peak in peaks if hs_suavizado[peak] >= hs_significativa]
        # Ahora mismo esto parece funcionar correctamente para butterfly, sin probar para el resto.

        magnitud = [hs_suavizado[true_peaks[i]] for i in range(len(true_peaks))]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_axis[indexes[i-1]:indexes[i]], hs_suavizado, 'b', label="Altura suavizada")
        ax.plot(x_axis[indexes[i-1]:indexes[i]][true_peaks], magnitud, 'ro', ls="", markersize=4, label="Brazadas")
        ax.set_xlabel('Frame')
        ax.set_ylabel('Altura en píxeles')
        ax.set_title('Variación de altura de la brazada. Spit %d' % i)
        ax.legend(loc='upper right')
        ax.grid(True)
        plt.show()

        # 3.4. Calcular tiempo en ROI.
        time_ROI = abs(last_index - first_index) / ResolutionValues.NORMAL_FRAME_RATE
        # 3.5. Calcular brazadas por minuto en función de true_peaks y time_ROI
        brazadas_min = len(true_peaks) * 60 / time_ROI
        # 3.6. Calcular brazadas totales.
        brazadas_min_total += brazadas_min

        # Imprimir indices de los splits y brazadas por minuto.
        print('Split %d: %.2f brazadas por minuto.' % (i, brazadas_min))
        print('Split %d: %d brazadas en %.2f segundos. \n' % (i, len(true_peaks), time_ROI))

    except IndexError as e:
        valid_splits -= 1
        print("Error en split %d. No hay nadador detectado en zona central de la piscina. " % i)
        continue
    # Find_peaks no levanta excepciones, pero avisa de que algunos picos pueden ser "menos importantes" de lo esperado.
    # except RuntimeWarning as w:

# 4. Hacer media entre las brazadas por minuto de todos los splits válidos.
brazadas_min_total /= valid_splits
print('Media a lo largo de la prueba: %.2f brazadas por minuto' % brazadas_min_total)

video.release()
cv2.destroyAllWindows()
