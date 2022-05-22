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

    def __init__(self):
        """ Método llamado cuando se crea un objeto"""

        self.args = None
        self.gsocbs = cv2.bgsegm.createBackgroundSubtractorGSOC()
        self.video = None

        self.parse_arguments()
        if self.args is not None and self.args.gui:
            videofilename = UI.askforvideofile()
            lanenumber = UI.askforlanenumber()
            save = UI.askforsavegraphs()
            show = UI.askforshowprocessing()
            if videofilename is not None and lanenumber is not None and save is not None and show is not None:
                self.process_video(videofilename, lanenumber, save, show)
        elif self.args is not None:
            self.process_video(self.args.video, self.args.calle, self.args.guardar, self.args.mostrar)

    def parse_arguments(self):
        """ Método para parsear argumentos """

        parser = argparse.ArgumentParser(description='Deteccion de nadadores usando GSOC')
        parser.add_argument('-g', '--gui', action='store_true', help='Usar GUI', default=False)
        if not parser.parse_args().gui:
            parser.add_argument('-v', '--video', type=str, help='Nombre del video', required=True)
            parser.add_argument('-c', '--calle', type=int, help='Número de calle', required=True)
            parser.add_argument('--mostrar', action='store_true',
                                help='Mostrar vídeo durante el procesamiento', default=False)
            parser.add_argument('--guardar', action='store_true',
                                help='Guardar gráficas tras procesamiento', default=False)
        self.args = parser.parse_args()

    def process_video(self, videofilename=None, lanenumber=None, save=False, show=False):
        """ Método para procesar video """

        # Abrir vídeo en función de cómo se haya especificado
        if self.args.gui and videofilename is not None:
            self.video = cv2.VideoCapture(videofilename)
        elif self.args.video is not None:
            self.video = cv2.VideoCapture(self.args.video)
        else:
            sys.exit('No se ha especificado ningún video')

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

        # Coordenadas frontera de la calle a analizar
        borde_abajo_calle = PV.LANES_BOTTOM_PIXEL.get(str(lanenumber))
        borde_arriba_calle = borde_abajo_calle + PV.LANE_HEIGHT

        # Vectores con coordenadas para posterior cálculo de estadísticas
        coordenadas = np.full(frames, np.NaN)
        altura_contorno = np.full(frames, np.NaN)

        start_time = time.time()

        # Comenzamos procesamiento del vídeo
        while self.video.isOpened():
            ok, original_frame = self.video.read()
            if ok:

                if frames_leidos % fps_factor != 0:
                    frames_leidos += 1
                else:
                    if necesita_redimension:
                        original_frame = cv2.resize(original_frame, (RV.HALF_WIDTH, RV.HALF_HEIGHT))
                    original_frame = original_frame[borde_abajo_calle:borde_arriba_calle, :, :]
                    frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2YCrCb)[..., 1]

                    # Aplicar substracción de fondo, lo que nos devuelve la variación de la imagen.
                    gsocfg = self.gsocbs.apply(frame)

                    contornos = cv2.findContours(gsocfg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contornos = contornos[0] if len(contornos) == 2 else contornos[1]
                    contornos_validos = []

                    for c in contornos:
                        if SV.SWIMMER_MIN_AREA <= cv2.contourArea(c) <= SV.SWIMMER_MAX_AREA:
                            [x, y, _, h] = cv2.boundingRect(c)
                            # Discriminar corchera por su altura y obviar la zona del trampolín.
                            if h > PV.CORCHES_HEIGHT and x > PV.TRAMPOLIN_WIDTH and \
                                    PV.MINIMUM_Y_ROI_LANE <= y <= PV.MAXIMUM_Y_ROI_LANE:
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

                        # Si el contorno está en el área de interés, guardamos sus posiciones para posterior análsis.
                        if PV.MINIMUM_Y_ROI_LANE <= y <= PV.MAXIMUM_Y_ROI_LANE:
                            coordenadas[frames_procesados] = x
                            altura_contorno[frames_procesados] = h

                        # Mostrar contornos y rectángulos de interés dependerá del flag show
                        if show:
                            cv2.rectangle(gsocfg, (x, y), (x + w, y + h), (255, 255, 255), 1)

                    elif len(cnts_ord) == 1:
                        [x, y, w, h] = cv2.boundingRect(cnts_ord[0])
                        if PV.MINIMUM_Y_ROI_LANE <= y <= PV.MAXIMUM_Y_ROI_LANE:
                            coordenadas[frames_procesados] = x
                            altura_contorno[frames_procesados] = h

                        # Mostrar contornos y rectángulos de interés dependerá del flag show
                        if show:
                            cv2.rectangle(gsocfg, (x, y), (x + w, y + h), (255, 255, 255), 1)

                    else:
                        frames_sin_contorno += 1

                    # Necesita que los frames tengan el mismo número de canales. Dado que mostramos el RGB, 3.
                    if show:
                        vertical_concat = np.vstack((
                            original_frame,
                            cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB),
                            cv2.cvtColor(gsocfg, cv2.COLOR_GRAY2RGB)
                        ))
                        cv2.imshow('Video', vertical_concat)

                    frames_leidos += 1
                    frames_procesados += 1
                print(f'\rProcesando frame {frames_leidos} de {frames}.', end='')

                key = cv2.waitKey(1) & 0xff
                # El bucle se pausa al pulsar P, y se reanuda al pulsar cualquier otra tecla
                if key == 112:
                    cv2.waitKey(-1)
                # El bucle se ejecuta hasta el último frame o hasta qu se presiona la tecla ESC
                if frames_leidos == frames or key == 27:
                    break

        self.video.release()
        cv2.destroyAllWindows()

        print('\nFrames sin contornos detectados: %d.' % frames_sin_contorno)
        print('Tiempo total de procesamiento del vídeo: %.2f segundos.\n' % (time.time() - start_time))

        analizar_datos_video(frames, fps, splits_esperados, save,
                             videofilename, coordenadas, altura_contorno)

# if __name__ == "__main__":
#    gsoc = GSOC.__new__(GSOC)
#    gsoc.__init__()
