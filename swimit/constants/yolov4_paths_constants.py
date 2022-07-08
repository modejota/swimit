from dataclasses import dataclass


@dataclass(frozen=True)
class YOLOv4Paths:
    """
    Objeto que almacenará las rutas por defecto a los ficheros importantes de YOLOv4

    Attributes
    ---------
    DEFAULT_YOLO_DIRECTORY: str
        Ruta relativa al directorio donde se encuentran los archivos de YOLOv4 necesarios para la ejecución.
    DEFAULT_CFG_FILE: str
        Ruta relativa al archivo de configuración de YOLOv4 utilizado para la detección de nadadores.
    DEFAULT_WEIGHTS_FILE: str
        Ruta relativa al archivo de pesos de YOLOv4 utilizado para la detección de nadadores.
    DEFAULT_NAMES_FILE: str
        Ruta relativa al archivo de YOLOv4 con los nombres de las posibles clases.
    """

    DEFAULT_YOLO_DIRECTORY = 'yolov4_files'
    DEFAULT_CFG_FILE = f'{DEFAULT_YOLO_DIRECTORY}/yolov4-custom-detection.cfg'
    DEFAULT_WEIGHTS_FILE = f'{DEFAULT_YOLO_DIRECTORY}/yolov4-custom_best.weights'
    DEFAULT_NAMES_FILE = f'{DEFAULT_YOLO_DIRECTORY}/obj.names'
