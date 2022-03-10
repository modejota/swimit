from dataclasses import dataclass
from constants.resolution_constants import ResolutionValues


@dataclass
class SwimmerValues:
    """
    Objeto que almacenará diferentes parámetros propios de un nadador.
    
    Attributes
    ----------
    SWIMSUIT_MAX_WIDTH: int
        Ancho máximo del bañador.
    SWIMMER_HEIGHT_DIFFERENCE: int
        Diferencia en altura entre el tronco superior e inferior del nadador
    SWIMMER_MAX_WIDTH: int
        Ancho máximo del nadador. (Altura si estuviera de pie)
    SWIMMER_MIN_AREA: int
        Mínima área para considerar válido el contorno.
    SWIMMER_MAX_AREA: int
        Máxima área para considerar válido el contorno.
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
            self.SWIMSUIT_MAX_WIDTH = 22
            self.SWIMMER_HEIGHT_DIFFERENCE = 12
            self.SWIMMER_MAX_WIDTH = 115
            self.SWIMMER_MIN_AREA = 200
            self.SWIMMER_MAX_AREA = 2500