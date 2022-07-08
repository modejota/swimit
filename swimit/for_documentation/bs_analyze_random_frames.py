import os
import re
import cv2
import math
import timeit
import argparse
import numpy as np

from swimit.UI import UI
from swimit.constants.pool_constants import PoolValues as PV
from swimit.constants.swimmer_constants import SwimmerValues as SV
from swimit.constants.resolution_constants import ResolutionValues as RV
from swimit.auxiliary_functions import union_de_contornos, yolobboxtocv, analizar_datos_video
from swimit.for_documentation.segmentation_metrics import iou, f1score

# Utilizado de cara a la documentación del trabajo.

parser = argparse.ArgumentParser(description='Analizar fotogramas aleatorios para documentacion')
parser.add_argument('--show', action='store_true', help='Mostrar procesamiento')
parser.add_argument('--analyze', action='store_true', help='Analizar fotogramas')
args = parser.parse_args()

the_path = os.path.join(os.path.dirname(__file__), 'random_frames')
the_files = os.listdir(the_path)
txt_files = [file for file in the_files if file.endswith('.txt')]
txt_files.sort(key=lambda var: [int(e) if e.isdigit() else e for e in re.findall(r'\D|\d+', var)])
txt_numbers = [int(file.split('.')[0]) for file in txt_files]
txt_numbers.sort()

# Para nuestros experimentos Competition_2016_10_15_18_15_16_Freestyle_50m_Male_Series_01_Scale_2_41.6667fps y calle 3.
videoname = UI.askforvideofile("../../sample_videos/")
lane = UI.askforlanenumber()
borde_abajo_calle = PV.LANES_BOTTOM_PIXEL.get(str(lane))
borde_arriba_calle = borde_abajo_calle + PV.LANE_HEIGHT
os.chdir('..')
os.chdir(the_path)

spaces = {
    "Saturacion de HSV": [cv2.COLOR_BGR2HSV, 1],
    "Cromimancia roja de YCbCr": [cv2.COLOR_BGR2YCrCb, 1],
    "Crominancia azul de YCbCr": [cv2.COLOR_BGR2YCrCb, 2]
}
ops = {
    "KNN": cv2.createBackgroundSubtractorKNN(detectShadows=False),
    "MOG": cv2.bgsegm.createBackgroundSubtractorMOG(),
    "MOG2": cv2.createBackgroundSubtractorMOG2(detectShadows=False),
    "GSOC": cv2.bgsegm.createBackgroundSubtractorGSOC()
}


iterations = len(spaces.keys()) * len(ops.keys())
current_iteration = 0
default_value = np.empty((), dtype=object)
default_value[()] = [0, 0, 0, 0]

open('../bs_random_frames_results.txt', 'w').close()

for s in range(len(spaces.keys())):
    space = list(spaces.values())[s]  # Pareja de espacio de color y banda
    with open('../bs_random_frames_results.txt', 'a') as results_file:
        results_file.write(f'{list(spaces.keys())[s]}\n')
    for i in range(len(ops.keys())):
        current_iteration += 1
        bs = list(ops.values())[i]
        # OpenCV fuerza a reabrir para mantener el booleano y correcto procesamiento.
        video = cv2.VideoCapture(videoname)
        fps = video.get(cv2.CAP_PROP_FPS)
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        processing_frames = frames
        alto, ancho = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Variables para posterior variación de resolución del vídeo y/o framerate
        necesita_redimension = (ancho != RV.HALF_WIDTH or alto != RV.HALF_HEIGHT)
        fps_factor = 1
        if fps >= RV.NORMAL_FRAME_RATE:
            fps_factor = math.ceil(fps / RV.NORMAL_FRAME_RATE)
            processing_frames = int(frames // fps_factor)

        frames_leidos, frames_guardar, frames_analizados, index_txt_numbers = 0, 0, 0, 0
        iou_values = np.full(45, 0.0, dtype=np.float64)
        f1_scores = np.full(45, 0.0, dtype=np.float64)

        coordenadas = np.full(processing_frames, np.nan, dtype=np.float64)
        anchuras_values = np.full(processing_frames, np.nan, dtype=np.float64)
        longitud_values = np.full(processing_frames, np.nan, dtype=np.float64)
        exec_time = 0.0
        while video.isOpened():
            start = timeit.default_timer()
            ok, original_frame = video.read()
            if ok:

                if frames_leidos % fps_factor != 0:
                    frames_leidos += 1
                else:
                    if necesita_redimension:
                        original_frame = cv2.resize(original_frame, (RV.HALF_WIDTH, RV.HALF_HEIGHT))
                    original_frame = original_frame[borde_abajo_calle:borde_arriba_calle, :, :]
                    pre_frame = cv2.cvtColor(original_frame, space[0])
                    frame = pre_frame[..., space[1]]

                    fg = bs.apply(frame)

                    start_read = timeit.default_timer()
                    gtbbox = None
                    if args.analyze and frames_leidos in txt_numbers:
                        with open(txt_files[index_txt_numbers]) as f:
                            yolo_data = f.read()
                        index_txt_numbers += 1
                        gtbbox = yolobboxtocv(yolo_data)
                    end_read = timeit.default_timer() - start_read

                    contornos = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contornos = contornos[0] if len(contornos) == 2 else contornos[1]
                    contornos_validos = []

                    for c in contornos:
                        [x, y, _, h] = cv2.boundingRect(c)
                        if h > PV.CORCHES_HEIGHT and x > PV.TRAMPOLIN_WIDTH and \
                                PV.MINIMUM_Y_ROI_LANE <= y <= PV.MAXIMUM_Y_ROI_LANE and \
                                SV.SWIMMER_MIN_AREA <= cv2.contourArea(c) <= SV.SWIMMER_MAX_AREA:
                            contornos_validos.append(c)

                    # Contornos por area descendente, el nadador será el más grande o la unión de los dos más grandes.
                    cnts_ord = sorted(contornos_validos, key=lambda contorno: cv2.contourArea(contorno), reverse=True)

                    # Si tengo mas de dos contornos, me quedare con los dos de mayor área.
                    if len(cnts_ord) >= 2:
                        [x, y, w, h] = cv2.boundingRect(cnts_ord[0])
                        [x2, y2, w2, h2] = cv2.boundingRect(cnts_ord[1])

                        # Compruebo si dichos contornos corresponden a tronco y piernas del nadador.
                        if (abs(x - x2) < SV.SWIMSUIT_MAX_WIDTH and
                                abs(y - y2) < SV.SWIMMER_HEIGHT_DIFFERENCE and
                                w + w2 < SV.SWIMMER_MAX_WIDTH):
                            [x, y, w, h] = union_de_contornos([x, y, w, h], [x2, y2, w2, h2])

                    elif len(cnts_ord) == 1:
                        [x, y, w, h] = cv2.boundingRect(cnts_ord[0])

                    if len(cnts_ord) >= 1:
                        coordenadas[frames_leidos] = x
                        anchuras_values[frames_leidos] = h
                        longitud_values[frames_leidos] = w

                    end_timer = timeit.default_timer() - start
                    exec_time += end_timer
                    exec_time -= end_read

                    if args.analyze and len(cnts_ord) >= 1:
                        if frames_leidos in txt_numbers:
                            gtdrawbox = np.array(gtbbox)
                            predbox = np.array([x, y, w, h])
                            gtbbox[2] -= gtbbox[0]     # Para el calculo de IOU utiliza ancho y alto, lo corrijo.
                            gtbbox[3] -= gtbbox[1]
                            iou_values[frames_analizados] = iou(gtbbox, predbox)[0]
                            f1_scores[frames_analizados] = f1score(gtbbox, predbox)
                            frames_analizados += 1
                            if args.show:
                                cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                                cv2.rectangle(original_frame,
                                              (gtdrawbox[0], gtdrawbox[1]), (gtdrawbox[2], gtdrawbox[3]),
                                              (0, 255, 0), 1)
                                cv2.imshow('Processed frame', original_frame)
                                cv2.waitKey(-1)
                    frames_guardar += 1
                    if args.show:
                        if len(cnts_ord) >= 1:
                            cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        cv2.imshow('Processed frame', original_frame)
                        cv2.waitKey(2)
                    frames_leidos += 1

                    print(f'\rProcesando frame {frames_leidos} de {frames} ({current_iteration}/{iterations})', end='')
                key = cv2.waitKey(1) & 0xff
                # El bucle se pausa al pulsar P, y se reanuda al pulsar cualquier otra tecla
                if key == 112:
                    cv2.waitKey(-1)
                # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
                if frames_leidos == frames or key == 27:
                    break
        video.release()
        cv2.destroyAllWindows()

        iou_mean = "{:0.5f}".format(np.mean(iou_values))
        iou_min, iou_max = "{:0.5f}".format(np.min(iou_values)), "{:0.5f}".format(np.max(iou_values))
        iou_std = "{:0.5f}".format(np.std(iou_values))
        f1_mean = "{:0.5f}".format(np.mean(f1_scores))
        f1_min, f1_max = "{:0.5f}".format(np.min(f1_scores)), "{:0.5f}".format(np.max(f1_scores))
        f1_std = "{:0.5f}".format(np.std(f1_scores))
        frame_exec_time = "{:0.3f}".format(exec_time / frames * 1000)
        exec_time = "{:0.3f}".format(exec_time)

        # Tiempo de cómputo del método completo a lo largo del vídeo.
        # Comprobado con el profiler lo que cuesta cada parte del código. Waitkey parece intentar balancear a 30 FPS.
        anchura_diff = np.diff(anchuras_values)
        anchura_diff_mean = np.nanmean(anchura_diff)
        anchura_diff_std = np.nanstd(anchura_diff)
        x_diff = np.diff(coordenadas)
        x_diff_mean = np.nanmean(x_diff)
        x_diff_std = np.nanstd(x_diff)

        with open('../bs_random_frames_results.txt', 'a') as results_file:
            results_file.write(f'\n{list(ops.keys())[i]}\n')
            results_file.write(f'IOU medio: {iou_mean} \t F1Score medio: {f1_mean}'
                               f'\nIOU mínimo: {iou_min} \t IOU máximo: {iou_max} \t IOU std: {iou_std}'
                               f'\nF1Score mínimo: {f1_min} \t F1Score máximo: {f1_max} \t F1Score std: {f1_std}'
                               f'\nTiempo cómputo total: {exec_time} segs. '
                               f'\tTiempo cómputo por fotograma: {frame_exec_time} milisegs.'
                               f'\nMedia de las diferencias de anchura: {anchura_diff_mean}'
                               f'\nDesviacion estándar de las diferencias de anchura: {anchura_diff_std}'
                               f'\nMedia de las differencias de x: {x_diff_mean}'
                               f'\nDesviacion estándar de las diferencias de x: {x_diff_std}\n')
    with open('../bs_random_frames_results.txt', 'a') as rf:
        rf.write("\n\n\n")

    analizar_datos_video(processing_frames, fps, True,
                         videoname, lane, coordenadas, anchuras_values, "ANALISIS_GSoC")
