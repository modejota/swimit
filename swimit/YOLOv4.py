import cv2
import sys
import time
import math
import argparse
import numpy as np

from swimit.UI import UI
from swimit.constants.pool_constants import PoolValues as PV
from swimit.constants.resolution_constants import ResolutionValues as RV
from swimit.constants.yolov4_paths_constants import YOLOv4Paths as YV4P
from swimit.auxiliary_functions import calcular_splits, analizar_datos_video, redimensionar_imagen


class YOLOv4:
    """
    Clase que implementa el procesamiento de un video con red neuronal YOLOv4

    Attributes
    ----------
    net: cv2.dnn_DetectionModel
        Red neuronal de YOLOv4
    args: argparse.Namespace
        Argumentos del programa.
        --gui marca el uso de interfaz gráfica, el resto pueden obtenerse por línea de comandos o por la propia GUI.
    video: cv2.VideoCapture
        Video a procesar
    """

    def __init__(self):
        """ Método al que se llama cuando se crea un objeto """

        self.net = None
        self.args = None
        self.video = None

        self.parse_arguments()
        if self.args is not None and self.args.gui:
            self.args.video = UI.askforvideofile("../sample_videos")
            self.args.calle = UI.askforlanenumber()
            self.args.guardar = UI.askforsavegraphs()
            self.args.mostrar = UI.askforshowprocessing()
            self.args.cfg = UI.askforcfgfile()
            self.args.weights = UI.askforweightsfile()
            self.args.gpu = UI.askforgpu()
        self.initialize_network_parameters()
        self.process_video()

    def parse_arguments(self):
        """ Método para parsear argumentos """

        parser = argparse.ArgumentParser(description='Deteccion de nadadores usando YOLOv4')

        parser.add_argument('--gui', action='store_true', help='Usar GUI')
        parser.add_argument('-v', '--video', type=str, help='Vídeo a procesar')
        parser.add_argument('-c', '--calle', type=int, help='Calle a analizar')
        parser.add_argument('-m', '--mostrar', action='store_true', help='Mostrar procesamiento')
        parser.add_argument('-g', '--guardar', action='store_true', help='Guardar gráficas')
        parser.add_argument('--cfg', help='Archivo de configuración de YOLO', default=YV4P.DEFAULT_CFG_FILE)
        parser.add_argument('--weights', help='Archivo de pesos de YOLO', default=YV4P.DEFAULT_WEIGHTS_FILE)
        parser.add_argument('--gpu', action='store_true', help='Usar NVIDIA CUDA')

        args = parser.parse_args()
        if args.gui is False:
            if args.video is None or args.calle is None:
                print('No se han especificado argumentos obligatorios [-v] [-c] '
                      'o el uso de la interfaz gráfica [--gui].')
                sys.exit(121)
        else:
            if not (args.video is None or args.calle is None or args.guardar is None or args.mostrar is None
                    or args.cfg is None or args.weights is None or args.gpu is None):
                print('Los valores de los argumentos adicionales serán sobreescritos por la interfaz gráfica.')
        self.args = args

    def initialize_network_parameters(self):
        """ Método para inicializar parámetros de la red neuronal, cargando el modelo. """

        self.net = cv2.dnn_DetectionModel(self.args.cfg, self.args.weights)
        if self.args.gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)    # Puede llegar a ser casi 200 veces más lento.
        self.net.setInputSize(256, 2688)    # Doble del tamaño de la calle en ambos ejes.
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)

    def process_video(self):
        """ Método para procesar un video """

        self.video = cv2.VideoCapture(self.args.video)
        video_name = self.args.video.split('/')[-1]

        # Estadísticas sobre el vídeo.
        frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video.get(cv2.CAP_PROP_FPS)
        ancho = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        processing_frames = frames
        frames_leidos, frames_procesados, frames_sin_contorno = 0, 0, 0

        # Variables para posterior variación de resolución del vídeo y/o framerate
        necesita_redimension = (ancho != RV.HALF_WIDTH or alto != RV.HALF_HEIGHT)
        fps_factor = 1
        if fps >= RV.NORMAL_FRAME_RATE:
            fps_factor = math.ceil(fps / RV.NORMAL_FRAME_RATE)
            processing_frames = int(frames // fps_factor)

        # Coordenadas frontera de la calle a analizar.
        # Al rotarse la primera calle es la de abajon si se ve el vídeo en horizontal.
        borde_abajo_calle = PV.LANES_BOTTOM_PIXEL_ROTATED.get(str(self.args.calle))
        borde_arriba_calle = borde_abajo_calle + PV.LANE_HEIGHT

        # Vectores con coordenadas para posterior cálculo de estadísticas
        coordenadas = np.full(processing_frames, np.NaN)
        altura_contorno = np.full(processing_frames, np.NaN)

        start_time = time.time()

        while self.video.isOpened():
            ok, frame = self.video.read()
            if ok:

                if frames_leidos % fps_factor != 0:
                    frames_leidos += 1
                else:
                    if necesita_redimension:
                        frame = cv2.resize(frame, (RV.HALF_WIDTH, RV.HALF_HEIGHT))
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    frame = frame[:, borde_abajo_calle:borde_arriba_calle, :]

                    timer = time.time()
                    classes, confidences, boxes = self.net.detect(frame, confThreshold=0.3, nmsThreshold=0.4)

                    if len(classes) != 0:
                        # La confianza se mantiene por si se quisiese mostrar en interfaz
                        confidence_and_boxes = zip(confidences.flatten(), boxes)
                        confidence_and_boxes = sorted(confidence_and_boxes,
                                                      key=lambda det: det[1][2] * det[1][3], reverse=True)

                        # confidence = confidence_and_boxes[0][0]
                        box = confidence_and_boxes[0][1]
                        x, y, w, h = box
                        if PV.MINIMUM_Y_ROI_LANE <= x <= PV.MAXIMUM_Y_ROI_LANE and y > PV.TRAMPOLIN_WIDTH:
                            coordenadas[frames_procesados] = y
                            altura_contorno[frames_procesados] = w
                            if self.args.mostrar:
                                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)

                    else:
                        frames_sin_contorno += 1

                    frames_leidos += 1
                    frames_procesados += 1
                    end_timer = "{:0.4f}".format((time.time() - timer))
                    print(f'\rProcesando frame {frames_leidos} de {frames} en {end_timer} segundos.', end='')
                    if self.args.mostrar:
                        cv2.imshow(video_name, cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
                key = cv2.waitKey(1) & 0xff
                # El bucle se pausa al pulsar P, y se reanuda al pulsar cualquier otra tecla
                if key == 112:
                    cv2.waitKey(-1)
                # El bucle se ejecuta hasta el último frame o hasta que se presiona la tecla ESC
                if frames_leidos == frames or key == 27:
                    break

        self.video.release()
        cv2.destroyAllWindows()

        print('\nFrames sin contornos detectados: %d.' % frames_sin_contorno)
        print('Tiempo total de procesamiento del vídeo: %.2f segundos.\n' % (time.time() - start_time))

        analizar_datos_video(processing_frames, fps, self.args.guardar,
                             self.args.video, self.args.calle, coordenadas, altura_contorno, "YOLOv4")


# Ejecutar desde este mismo script
# yolov4 = YOLOv4()
