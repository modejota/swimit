from dataclasses import dataclass


@dataclass(frozen=True)
class YOLOv4Paths:
    """
    Objeto que almacenará las rutas por defecto a los ficheros importantes de YOLOv4

    Attributes
    ---------
    DEFAULT_CFG_FILE: str
        Ruta relativa al archivo de configuración de YOLOv4 utilizado para la detección de nadadores.
    DEFAULT_WEIGHTS_FILE: str
        Ruta relativa al archivo de pesos de YOLOv4 utilizado para la detección de nadadores.
    DEFAULT_NAMES_FILE: str
        Ruta relativa al archivo de YOLOv4 con los nombres de las posibles clases.
    DEFAULT_CFG_DIRECTORY: str
        Ruta relativa al directorio donde se encuentran los archivos de configuración de YOLOv4.
    DEFAULT_WEIGHTS_DIRECTORY: str
        Ruta relativa al directorio donde se encuentran los archivos de pesos de YOLOv4.
    DEFAULT_NAMES_DIRECTORY: str
        Ruta relativa al directorio donde se encuentra el fichero con los nombres de las posibles clases.
    """

    DEFAULT_CFG_FILE = '../yolov4/darknet/cfg/yolov4-custom-detection.cfg'
    DEFAULT_WEIGHTS_FILE = '../yolov4/training/yolov4-custom_best.weights'
    DEFAULT_NAMES_FILE = '../yolov4/darknet/data/obj.names'
    DEFAULT_CFG_DIRECTORY = '../yolov4/darknet/cfg'
    DEFAULT_WEIGHTS_DIRECTORY = '../yolov4/training'
    DEFAULT_NAMES_DIRECTORY = '../yolov4/darknet/data'
