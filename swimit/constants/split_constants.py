
from dataclasses import dataclass
from constants.resolution_constants import ResolutionValues

@dataclass
class SplitValues:
    """
    Objeto que almacenará las constantes relativas a los splits.

    Attributes
    ----------
    REGEX_PATTERN: str
        Patrón para reconocer la longitud de la prueba, especificada en el título del vídeo.
    DISTANCE : int
        Distancia de un split en metros, correspondiente al largo de una piscina semiolímpica.
    THRESHOLD_BRAZADAS: int
        Número de frames mínimo entre dos posibles brazadas del nadador. Dependiente de la resolución del vídeo.
    SPLIT_MIN_FRAMES: int
        Número mínimo de frames que dura un split. Dependiente del frame rate del vídeo.
    """

    REGEX_PATTERN = '_([\d]*)m_'
    DISTANCE = 25

    def __init__(self,tipo_prueba,fps):
        self.THRESHOLD_BRAZADAS = 0
        self.SPLIT_MIN_FRAMES = 0
        if tipo_prueba == "freestyle":
            self.THRESHOLD_BRAZADAS = fps/2
        self.SPLIT_MIN_FRAMES = 350


