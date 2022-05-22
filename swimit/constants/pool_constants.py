from dataclasses import dataclass


@dataclass(frozen=True)
class PoolValues:
    """
    Objeto que almacenará diferentes parámetros propios a la piscina, sus medidas y posiciones relevantes de la misma.

    Attributes
    ----------
    LANE_HEIGHT: int
        Altura en píxeles de cada calle de la piscina.
    LANES_BOTTOM_PIXEL : dictionary
        Correspondencia entre el número de calle y número de píxel en el que comienzan.
    LEFT_T_X_POSITION: int
        Posición en el eje X en la que empieza el dibujo bajo la calle (en los extremos es una T).
    RIGHT_T_X_POSITION: int
        Posicion en el eje X en la que termina el dibujo bajo la calle (en los extremos es una T).
    CORCHES_HEIGHT: int
        Altura en píxeles de las corcheras que delimitan las calles de la piscina.
    TRAMPOLIN_WIDTH: int
        Ancho en píxeles del trampolín desde el que salta el nadador a la piscina.
    AFTER_JUMPING_X: int
        Posición en el eje X mínima en la que un nadador bracea por primera vez tras saltar desde el trampolín.
    MINIMUM_Y_ROI_LANE: int
        Posicion en el eje Y mínima para considerar que el nadador está en la región de interés, relativa a la calle.
    MAXIMUM_Y_ROI_LANE: int
        Posicion en el eje Y máxima para considerar que el nadador está en la región de interés, relativa a la calle.
    REAL_POOL_LENGTH: int
        Longitud real de la piscina en metros.
    SPLIT_MIN_FRAMES: int
        Número mínimo de frames que dura un split, para vídeos en framerate normal.
    """

    LANE_HEIGHT = 120
    LANES_BOTTOM_PIXEL = {"1": 670, "2": 580, "3": 485, "4": 387, "5": 285, "6": 190, "7": 100, "8": 0}
    LEFT_T_X_POSITION = 170
    RIGHT_T_X_POSITION = 1140
    CORCHES_HEIGHT = 10
    TRAMPOLIN_WIDTH = 60
    AFTER_JUMPING_X = 300
    MINIMUM_Y_ROI_LANE = 12
    MAXIMUM_Y_ROI_LANE = 108
    REAL_POOL_LENGTH = 25
    SPLIT_MIN_FRAMES = 350
