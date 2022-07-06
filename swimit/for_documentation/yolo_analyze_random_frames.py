import os
import re
import cv2
import math
import timeit
import argparse
import numpy as np

from swimit.UI import UI

from swimit.auxiliary_functions import rotateyolobboxtocv
from swimit.constants.pool_constants import PoolValues as PV
from swimit.constants.resolution_constants import ResolutionValues as RV
from swimit.for_documentation.segmentation_metrics import iou, f1score
from swimit.auxiliary_functions import analizar_datos_video

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
# No se ha contemplado el uso de vídeos de resolución completa para estas pruebas, todos los que tenemos son 1/2
videoname = UI.askforvideofile("../../sample_videos")
lane = UI.askforlanenumber()
borde_abajo_calle = PV.LANES_BOTTOM_PIXEL_ROTATED.get(str(lane))
borde_arriba_calle = borde_abajo_calle + PV.LANE_HEIGHT
os.chdir('..')
cfg = UI.askforcfgfile()
weights = UI.askforweightsfile()
os.chdir(the_path)

default_value = np.empty((), dtype=object)
default_value[()] = [0, 0, 0, 0]

combs = {
    # "1": [416, 416],
    # "2": [416, 2112],
    # "3": [416, 2528],
    # "4": [416, 3232],
    # "5": [416, 4480],
    "6": [256, 2688]
}

net = cv2.dnn_DetectionModel(weights, cfg)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
net.setInputScale(1.0 / 255.0)
net.setInputSwapRB(True)

current_iteration, iterations = 0, len(combs.keys())

with open('../yolo_random_frames_results.txt', 'w') as f:
    f.write("YOLOv4 \n")

for s in range(len(combs.keys())):
    current_iteration += 1
    comb = list(combs.values())[s]
    net.setInputSize(comb[0], comb[1])
    exec_time = 0.0

    video = cv2.VideoCapture(videoname)  # Fuerza la reapetura para mantener correcto procesamiento.
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
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
    confidences_values = np.full(45, 0.0, dtype=np.float64)
    coordenadas = np.full(processing_frames, np.nan, dtype=np.float64)
    anchuras_values = np.full(processing_frames, np.nan, dtype=np.float64)

    while video.isOpened():
        ok, original_frame = video.read()
        if ok:

            if frames_leidos % fps_factor != 0:
                frames_leidos += 1
            else:
                frame = cv2.rotate(original_frame, cv2.ROTATE_90_CLOCKWISE)
                frame = frame[:, borde_abajo_calle:borde_arriba_calle, :]

                gtbbox = None
                if args.analyze and frames_leidos in txt_numbers:
                    with open(txt_files[index_txt_numbers]) as f:
                        yolo_data = f.read()
                    index_txt_numbers += 1
                    gtbbox = rotateyolobboxtocv(yolo_data, 270)

                start_time = timeit.default_timer()
                classes, confidences, boxes = net.detect(frame, confThreshold=0.3, nmsThreshold=0.4)
                if len(classes) != 0:
                    box_detected = True
                    confidence_and_boxes = zip(confidences.flatten(), boxes)
                    confidence_and_boxes = sorted(confidence_and_boxes,
                                                  key=lambda det: det[1][2]*det[1][3], reverse=True)

                    confidence = confidence_and_boxes[0][0]
                    box = confidence_and_boxes[0][1]
                    x, y, w, h = box
                    if PV.MINIMUM_Y_ROI_LANE <= x <= PV.MAXIMUM_Y_ROI_LANE and y > PV.TRAMPOLIN_WIDTH:
                        coordenadas[frames_guardar] = y
                        anchuras_values[frames_guardar] = w
                        predbox = np.array(box)
                        if args.analyze and frames_leidos in txt_numbers:
                            gtdrawbox = np.array(gtbbox)
                            gtbbox[2] -= gtbbox[0]  # Para métricas requiere ancho y alto, corrijo.
                            gtbbox[3] -= gtbbox[1]
                            iou_values[frames_analizados] = iou(gtbbox, predbox)[0]
                            f1_scores[frames_analizados] = f1score(gtbbox, predbox)
                            confidences_values[frames_analizados] = confidence
                            # print("IOU: {} | F1: {} | Confianza: {}".format(frames_leidos,
                            # iou_values[frames_analizados], f1_scores[frames_analizados],
                            # confidences_values[frames_analizados]))
                            frames_analizados += 1
                            if args.show:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                                cv2.rectangle(frame,
                                              (gtdrawbox[0], gtdrawbox[1]),
                                              (gtdrawbox[2], gtdrawbox[3]),
                                              (0, 255, 0), 1)
                                cv2.imshow('Processed frame', cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
                                cv2.waitKey(-1)
                else:
                    box_detected = False
                end_timer = timeit.default_timer() - start_time
                exec_time += end_timer

                if args.show:
                    if box_detected:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.imshow('Processed frame', cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
                    cv2.waitKey(10)

                frames_guardar += 1
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
    conf_mean = "{:0.5f}".format(np.mean(confidences_values))
    conf_min, conf_max = "{:0.5f}".format(np.min(confidences_values)), "{:0.5f}".format(np.max(confidences_values))
    frame_exec_time = "{:0.3f}".format(exec_time / frames * 1000)
    exec_time = "{:0.3f}".format(exec_time)

    anchura_diff = np.diff(anchuras_values)
    anchura_diff_mean = np.nanmean(anchura_diff)
    anchura_diff_std = np.nanstd(anchura_diff)
    x_diff = np.diff(coordenadas)
    x_diff_mean = np.nanmean(x_diff)
    x_diff_std = np.nanstd(x_diff)

    with open('../yolo_random_frames_results.txt', 'a') as results_file:
        results_file.write(f'Redimensión en la entrada a {comb[0]}x{comb[1]}\n\n'
                           f'IOU medio: {iou_mean} \t F1Score medio: {f1_mean} \t Confianza media: {conf_mean}'
                           f'\nIOU minimo: {iou_min} \t IOU maximo: {iou_max} \t Desviacion estandar: {iou_std}'
                           f'\nF1Score minimo: {f1_min} \t F1Score maximo: {f1_max} \t Desviacion estandar: {f1_std}'
                           f'\nConfianza mínima: {conf_min} \t Confianza máxima: {conf_max} '
                           f'\nTiempo cómputo: {exec_time} segs'
                           f'\t Tiempo cómputo por fotograma: {frame_exec_time} milisegs'
                           f'\nMedia de las diferencias de anchura: {anchura_diff_mean}'
                           f'\nDesviacion estándar de las diferencias de anchura: {anchura_diff_std}'
                           f'\nMedia de las diferencias de x: {x_diff_mean}'
                           f'\nDesviacion estándar de las diferencias de x: {x_diff_std}\n\n\n')

    analizar_datos_video(processing_frames, fps, True,
                         videoname, lane, coordenadas, anchuras_values, "ANALISIS_YOLO")
