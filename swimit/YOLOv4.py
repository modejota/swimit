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

    def __init__(self):
        """ Método al que se llama cuando se crea un objeto """

        self.net = None
        self.args = None
        self.video = None
        self.names = None

        self.parse_arguments()
        if self.args is not None and self.args.gui:
            videofilename = UI.askforvideofile()
            lanenumber = UI.askforlanenumber()
            save = UI.askforsavegraphs()
            show = UI.askforshowprocessing()
            cfg = UI.askforcfgfile()
            weights = UI.askforweightsfile()
            names = UI.askfornamesfile()
            gpu = UI.askforgpu()
            if cfg is not None and weights is not None and names is not None and gpu is not None:
                self.initialize_network_parameters(cfg, weights, names, gpu)
            if videofilename is not None and lanenumber is not None and save is not None and show is not None:
                self.process_video(videofilename, lanenumber, save, show)
        elif self.args is not None:
            self.initialize_network_parameters(self.args.cfg, self.args.weights, self.args.names, self.args.gpu)
            self.process_video(self.args.video, self.args.calle, self.args.guardar, self.args.mostrar)

    def parse_arguments(self):
        """ Método para parsear argumentos """

        parser = argparse.ArgumentParser(description='Deteccion de nadadores usando YOLOv4')
        parser.add_argument('-g', '--gui', action='store_true', help='Usar GUI', default=False)
        if not parser.parse_args().gui:
            parser.add_argument('-v', '--video', type=str, help='Nombre del video', required=True)
            parser.add_argument('-c', '--calle', type=int, help='Número de calle', required=True)
            parser.add_argument('--mostrar', action='store_true',
                                help='Mostrar vídeo durante el procesamiento', default=False)
            parser.add_argument('--guardar', action='store_true',
                                help='Guardar gráficas tras procesamiento', default=False)
            parser.add_argument('--cfg', help='Archivo de configuración de YOLO',
                                required=False, default=YV4P.DEFAULT_CFG_FILE)
            parser.add_argument('--weights', help='Archivo de pesos de YOLO',
                                required=False, default=YV4P.DEFAULT_WEIGHTS_FILE)
            parser.add_argument('--names', help='Archivo de nombres de YOLO',
                                required=False, default=YV4P.DEFAULT_NAMES_FILE)
            parser.add_argument('--gpu', action='store_true', help='Usar NVIDIA CUDA', default=False)
        self.args = parser.parse_args()

    def initialize_network_parameters(self, cfg=None, weights=None, names=None, use_gpu=None):
        """
        Método para inicializar parámetros de la red neuronal, cargando el modelo.

        Parameters
        ----------
        cfg: str
            Ruta del archivo de configuración de YOLOv4
        weights: str
            Ruta del archivo de pesos de YOLOv4
        names: str
            Ruta del archivo de nombres de YOLOv4
        use_gpu: bool
            Usar NVIDIA CUDA (procesamiento por GPU)
        """

        self.net = cv2.dnn_DetectionModel(cfg, weights)
        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net.setInputSize(416, 1984)
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)
        with open(names, 'rt') as f:
            self.names = f.read().rstrip('\n').split('\n')

    def process_video(self, videofilename=None, lanenumber=None, save=False, show=False):
        """
        Método para procesar el vídeo.

        Parameters
        ----------
        videofilename: str
            Ruta del archivo de vídeo
        lanenumber: int
            Número de calle
        save: bool
            Guardar gráficas tras procesamiento
        show: bool
            Mostrar vídeo durante el procesamiento
        """

        if self.args.gui and videofilename is not None:
            self.video = cv2.VideoCapture(videofilename)
        elif self.args.video is not None:
            self.video = cv2.VideoCapture(self.args.video)
        else:
            print("No se ha especificado ningún video")
            sys.exit(102)

        # Estadísticas sobre el vídeo.
        frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video.get(cv2.CAP_PROP_FPS)
        ancho = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_leidos, frames_procesados, frames_sin_contorno = 0, 0, 0
        splits_esperados = calcular_splits(videofilename)

        # Variables para posterior variación de resolución del vídeo y/o framerate
        necesita_redimension = (ancho != RV.HALF_WIDTH or alto != RV.HALF_HEIGHT)
        fps_factor = 1
        if fps >= RV.NORMAL_FRAME_RATE:
            fps_factor = math.ceil(fps / RV.NORMAL_FRAME_RATE)
            frames = int(frames // fps_factor)

        # Coordenadas frontera de la calle a analizar.
        # Al rotar, coge una calle que no es.
        borde_abajo_calle = PV.LANES_BOTTOM_PIXEL_ROTATED.get(str(lanenumber))
        borde_arriba_calle = borde_abajo_calle + PV.LANE_HEIGHT

        # Vectores con coordenadas para posterior cálculo de estadísticas
        coordenadas = np.full(frames, np.NaN)
        altura_contorno = np.full(frames, np.NaN)

        # Color para dibujar el contorno
        b, g, r = 51, 204, 51

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

                    # Los umbrales puede que los cambie
                    timer = time.time()
                    classes, confidences, boxes = self.net.detect(frame, confThreshold=0.3, nmsThreshold=0.4)

                    if len(classes) != 0:
                        # No necesito nada más que las boxes en verdad. Si acaso confianza para calcular medias.
                        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                            x, y, w, h = box
                            if PV.MINIMUM_Y_ROI_LANE <= x <= PV.MAXIMUM_Y_ROI_LANE and y > PV.TRAMPOLIN_WIDTH:
                                coordenadas[frames_procesados] = y
                                altura_contorno[frames_procesados] = w
                                if show:
                                    cv2.rectangle(frame, box, color=(b, g, r), thickness=2)

                    else:
                        frames_sin_contorno += 1

                    frames_leidos += 1
                    frames_procesados += 1
                    end_timer = "{:0.4f}".format(1 / (time.time() - timer))
                print(f'\rProcesando frame {frames_leidos} de {frames}. Velocidad: {end_timer} FPS.', end='')
                if show:
                    # Redimensionamos ya que en vertical se sale de una pantalla Full HD (1292>1080)
                    cv2.imshow('Video', redimensionar_imagen(frame, height=900))
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

        analizar_datos_video(frames, fps, splits_esperados, save,
                             videofilename, coordenadas, altura_contorno)


# if __name__ == "__main__":
#     yolov4 = YOLOv4().__new__(YOLOv4)
#     yolov4.__init__()
