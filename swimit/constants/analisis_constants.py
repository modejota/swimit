from dataclasses import dataclass
from constants.resolution_constants import ResolutionValues

@dataclass
class AnalisisValues:
    """
    Objeto que almacenará los parámetros dependientes de la resolución del vídeo 
    utilizados para la detección del nadador y obtención de estadísticas.

    Attributes
    ----------
    LANE_HEIGHT: int
        Altura en píxeles de cada calle de la piscina.
    LANES_Y : dictionary
        Correspondencia entre el número de calle y su posición de inicio en el eje Y.
    SWIMMER_MIN_AREA: int
        Mínima área para considerar válido el contorno.
    SWIMMER_MAX_AREA: int
        Máxima área para considerar válido el contorno.
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
        Posicion en el eje Y mínima para considerar que el nadador está en la región de interés, relativa a la calle. (No debe pegarse a la corchera)
    MAXIMUM_Y_ROI_LANE: int
        Posicion en el eje Y máxima para considerar que el nadador está en la región de interés, relativa a la calle. (No debe pegarse a la corchera)
    """

    """
    Constructor de la clase
    Parameters
    ----------
    width : int
        Ancho en píxeles del vídeo.
    height : int
        Alto en píxeles del vídeo.
    """

    def __init__(self,width,height):
        if width == ResolutionValues.HALF_WIDTH and height == ResolutionValues.HALF_HEIGHT:
            self.LANE_HEIGHT = 120                
            self.LANES_Y = {"1": 670,"2": 580,"3": 485,"4": 387,"5": 285,"6":190,"7":100,"8":0}
            self.SWIMMER_MIN_AREA = 200
            self.SWIMMER_MAX_AREA = 2500
            self.LEFT_T_X_POSITION = 170         
            self.RIGHT_T_X_POSITION = 1140       
            self.CORCHES_HEIGHT = 10         
            self.TRAMPOLIN_WIDTH = 60
            self.AFTER_JUMPING_X = 300       
            self.MINIMUM_Y_ROI_LANE = 15     
            self.MAXIMUM_Y_ROI_LANE = 105
            # Quedan pendientes parametros para la union de contornos


