
from dataclasses import dataclass

@dataclass(frozen=True)
class SplitValues:
    """
    Objeto que almacenará las constantes relativas a los splits.

    Attributes
    ----------
    REGEX_PATTERN: str
        Patrón para reconocer la longitud de la prueba, especificada en el título del vídeo.
    DISTANCE : int
        Distancia de un split en metros, correspondiente al largo de una piscina semiolímpica.
    """

    REGEX_PATTERN = '_([\d]*)m_'
    DISTANCE = 25                 

