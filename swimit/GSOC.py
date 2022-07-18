import sys
import cv2
import time
import math
import argparse
import numpy as np

from swimit.UI import UI
from swimit.constants.pool_constants import PoolValues as PV
from swimit.constants.swimmer_constants import SwimmerValues as SV
from swimit.constants.resolution_constants import ResolutionValues as RV
from swimit.auxiliary_functions import calcular_splits, union_de_contornos, analizar_datos_video


class GSOC:
    """
    Clase que implementa el procesamiento de un video con algoritmo de substracción de fondos GSOC.

    Attributes
    ----------
    args: argparse.Namespace
        Argumentos del programa.
        --gui marca el uso de interfaz gráfica, el resto pueden obtenerse por línea de comandos o por la propia GUI.
    gsocbs: cv2.bgsegm.BackgroundSubtractorGSOC
        Algoritmo de substracción de fondos GSOC
    video: cv2.VideoCapture
        Video a procesar
    """

    def __init__(self):
        """ Método llamado cuando se crea un objeto"""

        self.args = None
        self.gsocbs = cv2.bgsegm.createBackgroundSubtractorGSOC()
        self.video = None

        self.parse_arguments()
        if self.args is not None and self.args.gui:
            self.args.video = UI.askforvideofile("../sample_videos")
            self.args.calle = UI.askforlanenumber()
            self.args.guardar = UI.askforsavegraphs()
            self.args.mostrar = UI.askforshowprocessing()
        self.process_video()

    def parse_arguments(self):
        """ Método para parsear argumentos """

        parser = argparse.ArgumentParser(description='Deteccion de nadadores usando GSOC')

        parser.add_argument('--gui', action='store_true', help='Usar interfaz gráfica')
        parser.add_argument('-v', '--video', type=str, help='Video a procesar')
        parser.add_argument('-c', '--calle', type=int, help='Calle a analizar', choices=range(1, 9))
        parser.add_argument('-g', '--guardar', action='store_true', help='Guardar gráficas')
        parser.add_argument('-m', '--mostrar', action='store_true', help='Mostrar procesamiento')

        args = parser.parse_args()
        if args.gui is False:
            if args.video is None or args.calle is None:
                print('No se han especificado argumentos obligatorios [-v] [-c] '
                      'o el uso de la interfaz gráfica [--gui].')
                sys.exit(121)
        else:
            if not (args.video is None or args.calle is None or args.guardar is None or args.mostrar is None):
                print('Los valores de los argumentos adicionales serán sobreescritos por la interfaz gráfica.')
        self.args = args

    def process_video(self):
        """ Método para procesar video """

        # Abrir vídeo
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
            fps_factor = math.ceil(fps / RV.NORMAL_FRAME_RATE)  # Ignorar algunos fotogramas
            processing_frames = int(frames // fps_factor)

        if self.args.calle < 1 or self.args.calle > 8:
            print('El número de calle debe estar entre 1 y 8')
            sys.exit(104)

        # Coordenadas frontera de la calle a analizar
        borde_abajo_calle = PV.LANES_BOTTOM_PIXEL.get(str(self.args.calle))
        borde_arriba_calle = borde_abajo_calle + PV.LANE_HEIGHT

        # Vectores con coordenadas para posterior cálculo de estadísticas
        coordenadas = np.full(processing_frames, np.NaN)
        altura_contorno = np.full(processing_frames, np.NaN)

        start_time = time.time()

        # Comenzamos procesamiento del vídeo
        while self.video.isOpened() and frames_leidos < 7000:
            ok, original_frame = self.video.read()
            if ok:

                if frames_leidos % fps_factor != 0:
                    frames_leidos += 1
                else:
                    if necesita_redimension:
                        original_frame = cv2.resize(original_frame, (RV.HALF_WIDTH, RV.HALF_HEIGHT))
                    original_frame = original_frame[borde_abajo_calle:borde_arriba_calle, :, :]
                    # Sin crear copia, OpenCV impide dibujar sobre el resultado de un slice
                    frame = np.copy(cv2.cvtColor(original_frame, cv2.COLOR_BGR2YCrCb)[..., 1])

                    # Aplicar substracción de fondo, lo que nos devuelve la variación de la imagen.
                    timer = time.time()
                    gsocfg = self.gsocbs.apply(frame)

                    contornos = cv2.findContours(gsocfg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contornos = contornos[0] if len(contornos) == 2 else contornos[1]
                    contornos_validos = []

                    for c in contornos:
                        [x, y, _, h] = cv2.boundingRect(c)
                        # Discriminar corchera por su altura, obviar la zona del trampolín y extremos verticales.
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

                    else:
                        frames_sin_contorno += 1

                    if len(cnts_ord) >= 1:
                        coordenadas[frames_procesados] = x
                        altura_contorno[frames_procesados] = h

                        # Mostrar contornos y rectángulos de interés dependerá del flag mostrar
                        if self.args.mostrar:
                            cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
                            cv2.rectangle(gsocfg, (x, y), (x + w, y + h), (255, 255, 255), 1)

                    # Necesita que los frames tengan el mismo número de canales. Dado que mostramos el RGB, 3.
                    if self.args.mostrar:
                        vertical_concat = np.vstack((
                            original_frame,
                            cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB),
                            cv2.cvtColor(gsocfg, cv2.COLOR_GRAY2RGB)
                        ))
                        cv2.imshow(video_name, vertical_concat)

                    frames_leidos += 1
                    frames_procesados += 1
                    end_timer = "{:0.4f}".format((time.time() - timer))
                    print(f'\rProcesando frame {frames_leidos} de {frames} en {end_timer} segundos.', end='')

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
                             self.args.video, self.args.calle, coordenadas, altura_contorno, "GSoC")


# Ejecutar desde este mismo script
# gsoc = GSOC()
