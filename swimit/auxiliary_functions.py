import re
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from swimit.constants.pool_constants import PoolValues as PV
from swimit.constants.resolution_constants import ResolutionValues as RV



def analizar_datos_video(frames: int, fps: float, splits_esperados: int, save: bool,
                         videofilename: str, coordenadas: np.ndarray, anchuras: np.ndarray):
    """
    Función que nos permite analizar los datos obtenidos de un vídeo.
    Se generará un fichero de texto con los datos obtenidos, y opcionalmente, gráficas.

    Parameters
    ----------
    frames: int
        Número de frames del vídeo.
    fps: float
        Frames por segundo del vídeo.
    splits_esperados: int
        Número de splits que se espera que nade el nadador
    save: boolean
        Indica si se deben generar y guardar gráficas.
    videofilename: str
        Ruta del vídeo que se ha procesado y cuyos datos se desean analizar
    coordenadas: np.ndarray
        Array con las coordenadas X o Y del nadador en cada frame del vídeo.
    anchuras: np.ndarray
        Array con las anchuras de los contornos del nadador en cada frame del vídeo.
    """

    # 1. Procesar coordenadas en las que no se detectó al nadador.
    for i in range(frames):
        if np.isnan(coordenadas[i]):
            coordenadas[i] = coordenadas[i - 1] if i > 0 else 0
        if np.isnan(anchuras[i]):
            anchuras[i] = anchuras[i - 1] if i > 0 else 0.0

    # 2. Hallar índices y coordenadas de los frames en los que se cambia de sentido con respecto del vídeo original.
    coordenadas_suavizadas = savgol_filter(coordenadas, (2 * round(fps)) + 1, 1)
    picos = list(np.concatenate((
        find_peaks(coordenadas_suavizadas, distance=PV.SPLIT_MIN_FRAMES, width=11)[0],
        find_peaks(-1 * coordenadas_suavizadas, distance=PV.SPLIT_MIN_FRAMES, width=11)[0]
    )))
    coordenadas_picos = [coordenadas[p] for p in picos]

    # El suavizado del final del vídeo puede crear falsos picos, los eliminamos.
    if len(coordenadas_picos) > splits_esperados:
        for i in range(len(coordenadas_picos) > splits_esperados):
            if abs(coordenadas_picos[-1] - coordenadas_picos[-2]) < 20:
                if abs(coordenadas_picos[-1] > abs(coordenadas_picos[-2])):
                    coordenadas_picos.pop()
                    picos.pop()
                else:
                    coordenadas_picos.pop(-2)
                    picos.pop(-2)

    # Hallamos el número real de splits que se han nadado, ya que hay vídeos en los que se corta a mitad.
    splits_reales = len(coordenadas_picos) if len(coordenadas_picos) != splits_esperados else splits_esperados

    indices = []
    try:
        primer_indice = np.where(coordenadas > PV.AFTER_JUMPING_X)[0][0]
        indices.append(primer_indice)
        # Correspondencia entre coordenada del pico y número de frame donde se produce.
        # El pico y el número de frame donde se produce nos sirven si no hubo ninguno cerca.
        for i, x in enumerate(coordenadas):
            if x in coordenadas_picos and \
                    not any(t in indices for t in range(i - PV.SPLIT_MIN_FRAMES // 2, i + PV.SPLIT_MIN_FRAMES // 2)):
                indices.append(i)
        ultimo_indice = np.where(coordenadas > 0)[0][-1]
        indices = np.append(np.sort(indices), ultimo_indice)
    except IndexError:
        print(f"No se ha podido encontrar al nadador en la región de interés durante todo el vídeo.\n"
              f"¿Está seguro de que alguien está nadando en la calle seleccionada?")
        sys.exit(109)

    # Hallar nombre del vídeo para conformar despúes nombre de otros ficheros.
    videofilename = Path(videofilename).stem
    # Crear directorio separado para resultados.
    if not os.path.exists("../results"):
        os.mkdir("../results")
    os.chdir("../results")

    if save:
        # Asegurar el directorio para guardar gráficas y resultado del análisis.
        if not os.path.exists("results_" + videofilename):
            os.makedirs("results_" + videofilename)
        os.chdir("results_" + videofilename)

        axis = np.arange(0, frames)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(axis, coordenadas, 'b', label="Coordenadas ")
        ax.plot(axis, coordenadas_suavizadas, 'r', label="Coordenadas suavizadas")
        ax.plot(axis[picos], coordenadas_suavizadas[picos], 'ro', markersize=5, label='Cambio de sentido')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Coordenadas a lo largo de la piscina')
        ax.set_title('Sentido del movimiento')
        ax.legend(loc='upper right')
        ax.grid(True)
        plt.savefig('sentido_movimiento_' + videofilename + '.png')

    brazadas_por_minuto_media = 0.0
    # 3. Cálculos en función del split. (Necesito "una iteración más" por manejo de índices)
    with open("analisis_" + videofilename + ".txt", "w") as f:
        for i in range(1, splits_reales + 1):
            try:
                # 3.1 Extraer las coordenadas y anchuras correspondientes a ese split.
                coordenadas_split = coordenadas[indices[i - 1]:indices[i]]
                anchuras_split = anchuras[indices[i - 1]:indices[i]]

                # 3.2 Establecer frames extremo de la región de interés.
                if i == 1:  # Primer split (izquierda a derecha), diferenciado por ser cuando salta el nadador.
                    indice_inicio = coordenadas_split[np.where(coordenadas_split >= PV.AFTER_JUMPING_X)[0][0]]
                    indice_final = coordenadas_split[np.where(coordenadas_split <= PV.RIGHT_T_X_POSITION)[0][-1]]
                elif i % 2 == 0:  # Splits pares (derecha a izquierda)
                    indice_inicio = coordenadas_split[np.where(coordenadas_split <= PV.RIGHT_T_X_POSITION)[0][0]]
                    indice_final = coordenadas_split[np.where(coordenadas_split >= PV.LEFT_T_X_POSITION)[0][-1]]
                else:  # Splits impares (izquierda a derecha)
                    indice_inicio = coordenadas_split[np.where(coordenadas_split >= PV.LEFT_T_X_POSITION)[0][0]]
                    indice_final = coordenadas_split[np.where(coordenadas_split <= PV.RIGHT_T_X_POSITION)[0][-1]]

                # 3.3 Hallar brazadas a partir de las variaciones significativas de las anchuras.
                anchura_significativa = np.mean(anchuras_split)
                anchuras_suavizada = savgol_filter(anchuras_split, 55, 2)
                picos, _ = find_peaks(anchuras_suavizada, distance=(fps / 2), width=9)
                picos_relevantes = [p for p in picos if anchuras_suavizada[p] >= anchura_significativa]

                if save:
                    axis = np.arange(0, frames)
                    magnitud = [anchuras_suavizada[picos_relevantes[i]] for i in range(len(picos_relevantes))]
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(axis[indices[i - 1]:indices[i]], anchuras_suavizada, 'b', label="Anchura suavizada")
                    ax.plot(axis[indices[i - 1]:indices[i]][picos_relevantes], magnitud, 'ro', ls="", markersize=4,
                            label="Brazadas")
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Anchura del nadador en píxeles')
                    ax.set_title('Variación de la anchura del nadador. Split %d' % i)
                    ax.legend(loc='upper right')
                    ax.grid(True)
                    plt.savefig('brazadas_split%d_' % i + videofilename + '.png')

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
                print("Error en split %d. No hay nadador detectado en zona central de la piscina. " % i)
                continue
            # Find_peaks no levanta excepción sino un warning, avisa de que algunos picos
            # pueden ser "menos importantes de lo inicialmente esperado" al tener menos anchura de la especificada.
            except RuntimeWarning:
                print("Aviso en split %d. Puede que haya picos 'menos relevantes' de lo esperado" % i)
                continue

        # 4. Hacer media entre las brazadas por minuto de todos los splits válidos.
        brazadas_por_minuto_media /= splits_reales
        f.write('Media a lo largo de la prueba: %.2f brazadas por minuto' % brazadas_por_minuto_media)


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
