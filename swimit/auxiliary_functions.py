import re
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from swimit.constants.pool_constants import PoolValues as PV
from swimit.constants.swimmer_constants import SwimmerValues as SV
from swimit.constants.resolution_constants import ResolutionValues as RV


def analizar_datos_video(frames: int, fps: float, save: bool,
                         videofilename: str, lanenumber: int,
                         coordenadas: np.ndarray, anchuras: np.ndarray, detector: str):
    """
    Función que nos permite analizar los datos obtenidos de un vídeo.
    Se generará un fichero de texto con los datos obtenidos, y opcionalmente, gráficas.

    Parameters
    ----------
    frames: int
        Número de frames del vídeo.
    fps: float
        Frames por segundo del vídeo.
    save: boolean
        Indica si se deben generar y guardar gráficas.
    videofilename: str
        Ruta del vídeo que se ha procesado y cuyos datos se desean analizar
    lanenumber: int
        Número de la calle que se ha procesado
    coordenadas: np.ndarray
        Array con las coordenadas X o Y del nadador en cada frame del vídeo.
    anchuras: np.ndarray
        Array con las anchuras de los contornos del nadador en cada frame del vídeo.
    detector: str
        Nombre del detector que se ha utilizado para detectar los contornos. GSoC o YOLOv4
    """

    video = cv2.VideoCapture(videofilename)

    splits_esperados = calcular_splits(videofilename)
    splits_por_sentido = splits_esperados // 2
    # Utilizado para cálculos que impliquen conocer datos del eje X de la gráfica. (Índice de fotograma)
    axis = np.arange(0, frames, dtype=np.int32)

    # 1. Procesar coordenadas en las que no se detectó al nadador.
    for i in range(frames):
        if np.isnan(coordenadas[i]):
            coordenadas[i] = coordenadas[i - 1] if i > 0 else 0
        if np.isnan(anchuras[i]):
            anchuras[i] = anchuras[i - 1] if i > 0 else 0.0

    # 2. Hallar índices y coordenadas de los frames en los que se cambia de sentido con respecto del vídeo original.

    coordenadas_suavizadas = savgol_filter(coordenadas, (2 * round(fps)) + 1, polyorder=1)

    picos_izq_to_der = find_peaks(coordenadas_suavizadas, distance=350, width=7)[0]
    picos_der_to_izq = find_peaks(-1 * coordenadas_suavizadas, distance=350, width=7)[0]
    picos_izq_to_der_reales = [int(axis[pico]) for pico in picos_izq_to_der
                               if coordenadas_suavizadas[pico] > 1080][:splits_por_sentido]
    # Algo menos de la posición de la segunda T
    if splits_esperados > 2:
        picos_der_to_izq_reales = [int(axis[pico]) for pico in picos_der_to_izq
                                   if 85 < coordenadas_suavizadas[pico] < 120][:splits_por_sentido]
        # LEFT_T_X_POSITION - longitud del nadador (corregir esquina inf izq a inf der). Rango pues es variable.

    else:
        picos_der_to_izq_reales = []

    # El punto donde comienza ROI primer split, y donde acaba la ROI del último necesitan ser
    # calculados a mano ya que no son considerados extremos relativos.
    picos_izq_to_der_reales = [int(np.where(coordenadas > PV.AFTER_JUMPING_X)[0][0])] + picos_izq_to_der_reales
    picos_der_to_izq_reales = [int(np.where(coordenadas > PV.LEFT_T_X_POSITION)[0][-1])] + picos_der_to_izq_reales

    picos_sentido = np.sort(np.concatenate((picos_der_to_izq_reales, picos_izq_to_der_reales)))
    splits_reales = len(picos_sentido) - 1
    # Hallar nombre del vídeo y convertir calle a string para conformar despúes nombre de otros ficheros.
    videofilename = Path(videofilename).stem
    lanenumber = str(lanenumber)
    # Crear directorio separado para resultados.
    if not os.path.exists("../results"):
        os.mkdir("../results")
    os.chdir("../results")

    path_results_directory = detector + "_results_calle_" + lanenumber + '_' + videofilename
    path_results_file = detector + "_analisis_calle_" + lanenumber + '_' + videofilename + ".txt"

    # Asegurar el directorio para guardar gráficas y resultado del análisis.
    if not os.path.exists(path_results_directory):
        os.makedirs(path_results_directory)
    os.chdir(path_results_directory)

    brazadas_por_minuto_media = 0.0
    puntos_interes = picos_sentido.copy()
    i = 0
    # 3. Cálculos en función del split. (Necesito "una iteración más" por manejo de índices)
    with open(path_results_file, "w") as f:
        while i < splits_reales:
            i += 1
            try:
                # 3.1 Extraer las coordenadas y anchuras correspondientes a ese split.
                coordenadas_split = coordenadas[picos_sentido[i - 1]:picos_sentido[i]]
                anchuras_split = anchuras[picos_sentido[i - 1]:picos_sentido[i]]

                # 3.2 Establecer frames extremo de la región de interés.
                # Se corrige con longitud media allá donde sea necesario esquina inf der en lugar de inf izq.
                if i == 1:  # Primer split (izquierda a derecha), diferenciado por ser cuando salta el nadador.
                    indice_inicio = np.where(coordenadas_split >= PV.AFTER_JUMPING_X)[0][0]
                    indice_final = np.where(coordenadas_split <=
                                            PV.RIGHT_T_X_POSITION + SV.SWIMMER_AVERAGE_LENGTH)[0][-1]

                elif i % 2 == 0:  # Splits pares (derecha a izquierda)
                    indice_inicio = np.where(coordenadas_split <= PV.RIGHT_T_X_POSITION)[0][0]
                    indice_final = np.where(coordenadas_split >=
                                            PV.LEFT_T_X_POSITION - SV.SWIMMER_AVERAGE_LENGTH)[0][-1]

                else:  # Splits impares (izquierda a derecha)
                    indice_inicio = np.where(coordenadas_split >= PV.LEFT_T_X_POSITION)[0][0]
                    indice_final = np.where(coordenadas_split <=
                                            PV.RIGHT_T_X_POSITION + SV.SWIMMER_AVERAGE_LENGTH)[0][-1]

                np.append(puntos_interes, (indice_inicio, indice_final))

                if abs(indice_inicio - indice_final) > 1800:
                    splits_reales -= 1
                    continue

                # 3.3 Hallar brazadas a partir de las variaciones significativas de las anchuras.
                anchura_significativa = np.mean(anchuras_split)
                anchuras_suavizada = savgol_filter(anchuras_split, 55, 2)
                picos, _ = find_peaks(anchuras_suavizada, distance=(fps / 2), width=9)
                picos_relevantes = [p for p in picos if anchuras_suavizada[p] >= anchura_significativa]

                if save:
                    path_brazadas_file = 'brazadas_split%d_' % i + videofilename + '.png'
                    magnitud = [anchuras_suavizada[picos_relevantes[i]] for i in range(len(picos_relevantes))]
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(axis[picos_sentido[i - 1]:picos_sentido[i]],
                            anchuras_suavizada, 'b', label="Anchura suavizada")
                    ax.plot(axis[picos_sentido[i - 1]:picos_sentido[i]][picos_relevantes],
                            magnitud, 'ro', ls="", markersize=5,
                            label="Brazadas")
                    plt.ylim([10, 60])  # Ningún dato experimental útil se ha movido fuera de estos umbrales.
                    ax.set_xlabel('Fotograma')
                    ax.set_ylabel('Anchura del nadador en píxeles')
                    ax.set_title('Variación de la anchura del nadador. Split %d' % i)
                    ax.legend(loc='upper right')
                    ax.grid(True)
                    # Asegurar el guardado de la gráfica.
                    if os.path.exists(path_brazadas_file):
                        os.remove(path_brazadas_file)
                    plt.savefig(path_brazadas_file)

                # 3.4 Calcular tiempo en ROI.
                tiempo_en_roi = abs(indice_final - indice_inicio) / RV.NORMAL_FRAME_RATE
                # 3.5. Calcular brazadas por minuto en función de true_peaks y time_ROI
                brazadas_por_minuto = len(picos_relevantes) * 60 / tiempo_en_roi
                # 3.6. Calcular brazadas totales.
                brazadas_por_minuto_media += brazadas_por_minuto

                f.write('Split %d: %.2f brazadas por minuto.\n' % (i, brazadas_por_minuto))
                f.write('Split %d: %d brazadas en %.2f segundos. \n\n' % (i, len(picos_relevantes), tiempo_en_roi))

            except IndexError:
                splits_reales -= 1
                print("\nError en split %d. No hay nadador detectado en zona central de la piscina. "
                      "Puede que el vídeo no abarque la totalidad de la prueba. " % i)
                continue
            # Find_peaks no levanta excepción sino un warning, avisa de que algunos picos
            # pueden ser "menos importantes de lo inicialmente esperado" al tener menos anchura de la especificada.
            except RuntimeWarning:
                print("\nAviso en split %d. Puede que haya picos 'menos relevantes' de lo esperado" % i)
                continue

        # 4. Hacer media entre las brazadas por minuto de todos los splits válidos.
        brazadas_por_minuto_media /= splits_reales
        f.write('Media a lo largo de la prueba: %.2f brazadas por minuto' % brazadas_por_minuto_media)

    if save:
        path_cambio_sentido_file = 'sentido_movimiento_calle_' + lanenumber + '_' + videofilename + '.png'
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(axis, coordenadas, 'b', label="Coordenadas ")
        ax.plot(axis, coordenadas_suavizadas, 'r', label="Coordenadas suavizadas")
        ax.plot(axis[puntos_interes], coordenadas_suavizadas[puntos_interes],
                'go', markersize=6, label='Puntos de interés')
        ax.set_xlabel('Fotograma')
        ax.set_ylabel('Coordenadas a lo largo de la piscina')
        ax.set_title('Puntos de interés')
        ax.legend(loc='upper right')
        ax.grid(True)
        # Asegurar el guardado de la gráfica.
        if os.path.exists(path_cambio_sentido_file):
            os.remove(path_cambio_sentido_file)
        plt.savefig(path_cambio_sentido_file)


def union_de_contornos(a: list, b: list) -> tuple:
    """
    Obtiene el rectángulo que contiene a otros dos rectángulos dados.
    Parameters
    ----------
    a : list
        Primer rectángulo, compuesto por coordenadas X, Y, ancho y alto.
    b: list
        Segundo rectángulo, compuesto por coordenadas X, Y, ancho y alto.
    Returns
    -------
    x,y,w,h : list
        Rectángulo que contiene a los pasados por parámetro, compuesto por coordenadas X, Y, ancho y alto.
    """
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return x, y, w, h


def calcular_splits(videoname: str) -> int:
    """
    Obtiene el número de splits de la prueba.
    Parameters
    ----------
    videoname : str
        Nombre del fichero con la grabación de la prueba.
    Returns
    -------
    splits : int
        Número de splits de la prueba.
    """
    splits = 0
    match = re.search(r"_(\d*)m_", videoname)
    if match:
        total_distance = match.group(1)
        splits = int(total_distance) // PV.REAL_POOL_LENGTH
    return splits


def redimensionar_imagen(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Función para redimensionar una imagen de manera que mantenga la proporcion.
    Parameters
    ---------
    image: numpy.ndarray
        Imagen a redimensionar.
    width: int
        Ancho de la imagen que se desea que tenga la imagen redimensionada.
    height: int
        Alto de la imagen que se desea que tenga la imagen redimensionada.
    inter: int
        Tipo de interpolación a utilizar.

    Returns
    ------
    resized: numpy.ndarray
        Imagen redimensionada.
    """

    (h, w) = image.shape[:2]

    # No se indican dimensiones, devolvemos original
    if width is None and height is None:
        return image

    # Comprobamos que dimensión se especifica y la otra se calcula manteniendo la proporción
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def yolobboxtocv(yolo_data: str, H=120, W=1292):
    """
    Función para convertir una bounding box en formato YOLO a formato OpenCV.
    Parameters
    ---------
    yolo_data: list
        Información de la segmentación en formato YOLO, conforme se extrae del fichero.
        Es una lista de 5 elementos: [class_id, center_x, center_y, width, height].
    H: int
        Alto de la imagen.
    W: int
        Ancho de la imagen.
    Returns
    ------
    bbox: list
        Bounding box en formato OpenCV. Es una lista de 4 elementos: [x1, y1, x2, y2].
        Los elementos con subíndice 1 hacen referencia al punto de la esquina superior izquierda de la bounding box.
        Los elementos con subíndice 2 hacen referencia al punto de la esquina inferior derecha de la bounding box.
    """
    bbox = yolo_data.strip('\n').split(' ')[1:5]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    bbox_width = x2 * W
    bbox_height = y2 * H
    center_x = x1 * W
    center_y = y1 * H

    voc = [center_x - (bbox_width / 2), center_y - (bbox_height / 2),
           center_x + (bbox_width / 2), center_y + (bbox_height / 2)]

    return [int(v) for v in voc]


def rotateyolobboxtocv(yolo_data: str, angle: int, old_height=120, old_width=1292, new_height=1292, new_width=120):
    """
    Función para rotar una bounding box a partir de la información en formato YOLO.
    Las dimensiones se pueden obtener con el atributo shape de la imagen leída, antes y después de aplicar cv2.rotate().
    Parameters
    ----------
    yolo_data: list
        Información de la segmentación en formato YOLO, conforme se extrae del fichero.
        Es una lista de 5 elementos: [class_id, center_x, center_y, width, height].
    old_height:
        Alto de la imagen original.
    old_width:
        Ancho de la imagen original.
    new_height:
        Alto de la imagen destino.
    new_width:
        Ancho de la imagen destino.
    angle:
        Ángulo de rotación, en grados, en el sentido contrario de las agujas del reloj.

    Returns
    -------
    new_bbox:
        Bounding box en formato OpenCV rotada. Es una lista de 4 elementos: [x1, y1, x2, y2].
        Los elementos con subíndice 1 hacen referencia al punto de la esquina superior izquierda de la bounding box.
        Los elementos con subíndice 2 hacen referencia al punto de la esquina inferior derecha de la bounding box.
    """
    rotation_angle = angle * np.pi / 180
    rot_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                           [np.sin(rotation_angle), np.cos(rotation_angle)]])

    center_x, center_y, bbox_width, bbox_height = yolobboxtocv(yolo_data, old_height, old_width)

    upper_left_corner_shift = (center_x - old_width / 2, -old_height / 2 + center_y)
    upper_right_corner_shift = (bbox_width - old_width / 2, -old_height / 2 + center_y)
    lower_left_corner_shift = (center_x - old_width / 2, -old_height / 2 + bbox_height)
    lower_right_corner_shift = (bbox_width - old_width / 2, -old_height / 2 + bbox_height)

    new_lower_right_corner = [-1, -1]
    new_upper_left_corner = []

    for i in (upper_left_corner_shift, upper_right_corner_shift, lower_left_corner_shift,
              lower_right_corner_shift):
        new_coords = np.matmul(rot_matrix, np.array((i[0], -i[1])))
        x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
        if new_lower_right_corner[0] < x_prime:
            new_lower_right_corner[0] = x_prime
        if new_lower_right_corner[1] < y_prime:
            new_lower_right_corner[1] = y_prime

        if len(new_upper_left_corner) > 0:
            if new_upper_left_corner[0] > x_prime:
                new_upper_left_corner[0] = x_prime
            if new_upper_left_corner[1] > y_prime:
                new_upper_left_corner[1] = y_prime
        else:
            new_upper_left_corner.append(x_prime)
            new_upper_left_corner.append(y_prime)

    return [int(new_upper_left_corner[0]), int(new_upper_left_corner[1]),
            int(new_lower_right_corner[0]), int(new_lower_right_corner[1])]
